# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import asyncio
import getpass
import inspect
import logging
import os
import pickle
import re
import socket
import time
from contextlib import contextmanager
from dataclasses import asdict
from types import MethodType
from typing import Any, Generator

import numpy as np
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from omegaconf import ListConfig
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
from vllm.config import CompilationConfig, CompilationLevel, LoRAConfig
from vllm.lora.request import LoRARequest

try:
    from vllm.worker.worker_base import WorkerWrapperBase
except ModuleNotFoundError:
    # https://github.com/vllm-project/vllm/commit/6a113d9aed8221a9c234535958e70e34ab6cac5b
    from vllm.v1.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL
from verl.utils.device import is_npu_available
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


# ============================================================================
# SID Reasoner customization: helpers for vLLM beam search over semantic-ID
# tokens that follow the model's reasoning (</think>) span.
# Ported from the fork's verl 0.4.1 vllm_rollout_spmd.py.
# ============================================================================
def truncate_at_end_think(tokens, marker=[151668, 271], clip_chars=20):
    """
    Truncate a token sequence at the first occurrence of `marker` within the
    last `clip_chars` tokens. If the marker is found, keep tokens up to and
    including the marker. If not found, return the original sequence.

    Args:
        tokens (List[int]): The original token sequence.
        marker (List[int]): The token sequence indicating the </think>\n boundary.
        clip_chars (int): Number of tokens from the end to search for the marker.

    Returns:
        List[int]: The truncated token sequence, or the original sequence
                   if the marker is not found.
    """
    m = len(marker)
    # Limit search to the last `clip_chars` tokens for efficiency
    search_start = max(0, len(tokens) - clip_chars - m + 1)

    for i in range(search_start, len(tokens) - m + 1):
        if tokens[i : i + m] == marker:
            return tokens[: i + m]  # Keep marker included

    # Marker not found -> return original (format reward will be 0)
    return tokens


def prepare_reasoning_prefix(
    tokens: list[int],
    end_think_marker: list[int],
    reasoning_separator: list[int],
    eos_token_id: int,
    max_length: int,
) -> tuple[list[int], int]:
    """Keep sampled reasoning, normalize its separator, and report sampled length."""
    marker_length = len(end_think_marker)
    for start in range(len(tokens) - marker_length + 1):
        if tokens[start : start + marker_length] == end_think_marker:
            reasoning = tokens[: start + marker_length]
            break
    else:
        reasoning = list(tokens)
        while reasoning and reasoning[-1] == eos_token_id:
            reasoning.pop()
        reasoning = reasoning[: max_length - len(reasoning_separator)]
        if not reasoning:
            raise RuntimeError("Reasoning rollout ended before producing any trainable token")
        return reasoning + reasoning_separator, len(reasoning)

    separator_suffix = reasoning_separator[marker_length:]
    normalized = reasoning + separator_suffix
    if len(normalized) > max_length:
        raise ValueError("Sampled reasoning leaves no room for the constrained SID")
    return normalized, len(reasoning)


def build_sid_token_trie(tokenizer, sid_sequences, depth: int) -> dict[tuple[int, ...], list[int]]:
    """Build token-ID prefix constraints from catalog SID paths."""
    trie: dict[tuple[int, ...], set[int]] = {}
    sequence_count = 0

    for sid_sequence in sid_sequences:
        if len(sid_sequence) != depth:
            raise ValueError(f"Expected {depth} SID tokens, got {sid_sequence}")

        token_ids = []
        for sid_token in sid_sequence:
            encoded = tokenizer.encode(sid_token, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"SID token {sid_token!r} maps to {len(encoded)} tokenizer tokens")
            token_ids.append(encoded[0])

        if tokenizer.encode("".join(sid_sequence), add_special_tokens=False) != token_ids:
            raise ValueError(f"SID path does not tokenize atomically: {sid_sequence}")

        for position, token_id in enumerate(token_ids):
            trie.setdefault(tuple(token_ids[:position]), set()).add(token_id)
        sequence_count += 1

    if sequence_count == 0:
        raise ValueError("Cannot build constrained decoding trie from an empty SID catalog")

    return {prefix: sorted(token_ids) for prefix, token_ids in trie.items()}


def vllm_constrained_beam_search(
    llm,
    prompts_ids: list[list[int]],
    sid_token_trie: dict[tuple[int, ...], list[int]],
    depth: int,
    beam_width: int,
    lora_requests=None,
) -> list[list[list[int]]]:
    """Decode ordered SID beams within catalog constraints."""
    if beam_width < 2:
        raise ValueError("Constrained SID beam width must be at least 2")
    if isinstance(lora_requests, list) and len(lora_requests) != len(prompts_ids):
        raise ValueError("Expected one LoRA request per prompt")

    beams = [[([], 0.0)] for _ in prompts_ids]

    for _position in range(depth):
        allowed_token_ids = []
        sampling_params = []
        step_prompts = []
        beam_origins = []

        for prompt_index, (prompt_ids, prompt_beams) in enumerate(zip(prompts_ids, beams)):
            for sid_prefix, cumulative_logprob in prompt_beams:
                allowed = sid_token_trie.get(tuple(sid_prefix))
                if not allowed:
                    raise RuntimeError(f"No valid SID continuation for token prefix {sid_prefix}")
                allowed_token_ids.append(set(allowed))
                sampling_params.append(
                    SamplingParams(
                        n=1,
                        max_tokens=1,
                        temperature=0.0,
                        logprobs=min(beam_width, len(allowed)),
                        detokenize=False,
                        allowed_token_ids=allowed,
                    )
                )
                step_prompts.append({"prompt_token_ids": prompt_ids + sid_prefix})
                beam_origins.append((prompt_index, sid_prefix, cumulative_logprob))

        step_lora_requests = lora_requests
        if isinstance(lora_requests, list):
            step_lora_requests = [lora_requests[prompt_index] for prompt_index, _, _ in beam_origins]

        step_outputs = llm.generate(
            prompts=step_prompts,
            sampling_params=sampling_params,
            lora_request=step_lora_requests,
            use_tqdm=False,
        )
        if len(step_outputs) != len(step_prompts):
            raise RuntimeError("vLLM returned an unexpected constrained-beam batch size")

        candidates = [[] for _ in prompts_ids]
        for index, output in enumerate(step_outputs):
            sample = output.outputs[0]
            token_ids = sample.token_ids
            if len(token_ids) != 1 or token_ids[0] not in allowed_token_ids[index]:
                raise RuntimeError("vLLM emitted a token outside the SID catalog constraint")
            if not sample.logprobs or len(sample.logprobs) != 1:
                raise RuntimeError("vLLM did not return token log probabilities for constrained beam search")

            prompt_index, sid_prefix, cumulative_logprob = beam_origins[index]
            for token_id, token_logprob in sample.logprobs[0].items():
                if token_id in allowed_token_ids[index]:
                    candidates[prompt_index].append(
                        (sid_prefix + [token_id], cumulative_logprob + token_logprob.logprob)
                    )

        for prompt_index, prompt_candidates in enumerate(candidates):
            if not prompt_candidates:
                raise RuntimeError(f"Constrained beam search produced no candidates for prompt {prompt_index}")
            beams[prompt_index] = sorted(
                prompt_candidates,
                key=lambda candidate: candidate[1],
                reverse=True,
            )[:beam_width]

    return [[sid_tokens for sid_tokens, _score in prompt_beams] for prompt_beams in beams]


_SOLUTION_CLIP_CHARS = 100


def extract_content_after_think(output: str) -> str | None:
    if len(output) > _SOLUTION_CLIP_CHARS:
        output = output[-_SOLUTION_CLIP_CHARS:]

    match = re.search(r"</think>\s*(.*)", output, re.DOTALL)
    if not match:
        return None
    after_think = match.group(1).strip()
    return after_think if after_think else None


def vllm_beam_search_concat(
    llm,
    tokenizer,
    prompts_ids: list[list[int]],
    params,
    beam_width: int = 10,
    depth: int = 3,
    sep: str = "<|beam_sep|>\n",  # or any special marker
):
    beam_params = BeamSearchParams(
        beam_width=beam_width,
        max_tokens=depth,
        temperature=0.0,
        length_penalty=0.0,
    )
    prompts = [{"prompt_token_ids": _} for _ in prompts_ids]
    all_beams = llm.beam_search(
        prompts,
        beam_params,
    )

    concatenated = []
    for _pi, beam_list in enumerate(all_beams):
        # build one string per prompt: beam1 + sep + beam2 + sep + ...
        pieces = []
        for seq in beam_list.sequences:
            continuation = extract_content_after_think(seq.text)
            if continuation is None:
                continuation = ""
            else:
                continuation = continuation.strip()
            pieces.append(continuation)

        concatenated_text = sep.join(pieces)
        concatenated.append(concatenated_text)

    return concatenated


if is_version_ge(pkg="vllm", minver="0.7.3"):
    VLLMHijack.hijack()


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)

        if config.layered_summon:
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

        model_path = model_config.local_path
        tokenizer = model_config.tokenizer
        model_hf_config = model_config.hf_config
        trust_remote_code = model_config.trust_remote_code
        self.lora_kwargs = (
            {"enable_lora": True, "max_loras": 1, "max_lora_rank": model_config.lora_rank}
            if model_config.lora_rank > 0
            else {}
        )

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        # copy it to avoid secretly modifying the engine config
        engine_kwargs = config.get("engine_kwargs", {}).get("vllm", {}) or {}

        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        compilation_config = {}

        cudagraph_capture_sizes = config.get("cudagraph_capture_sizes")
        # enforce_eager must be False to use cudagraph
        if not config.enforce_eager and cudagraph_capture_sizes:
            if isinstance(cudagraph_capture_sizes, ListConfig):
                compilation_config["compilation_config"] = CompilationConfig(
                    level=CompilationLevel.PIECEWISE, cudagraph_capture_sizes=cudagraph_capture_sizes
                )
            else:
                logger.warning(f"cudagraph_capture_sizes must be a list, but got {cudagraph_capture_sizes}")

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            max_num_seqs=config.max_num_seqs,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=config.enable_prefix_caching,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **compilation_config,
            **self.lora_kwargs,
            **engine_kwargs,
        )

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
            repetition_penalty=config.get("repetition_penalty", 1.0),
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

        # === SID Reasoner: vLLM beam search over semantic-ID tokens ===
        # `sid_beam_size` / `sid_length` are optional RolloutConfig fields
        # (set via CLI override). Read with .get() to stay safe on the
        # structured BaseConfig (where `in` raises AttributeError).
        self.tokenizer = tokenizer
        self.truncate_marker = self.tokenizer.encode("</think>\n\n", add_special_tokens=False)
        _sid_beam_size = config.get("sid_beam_size", None)
        _sid_length = config.get("sid_length", None)
        self.activate_beam_search = (
            _sid_beam_size is not None and _sid_length is not None and _sid_beam_size > 1
        )
        if self.activate_beam_search:
            self.sid_beam_size = _sid_beam_size
            self.num_sid_tokens = _sid_length

        _sid_constrained_beam_size = config.get("sid_constrained_beam_size", None)
        self.activate_constrained_beam_search = _sid_constrained_beam_size is not None
        _sid_validation_beam_size = config.get("sid_validation_beam_size", None)
        self.activate_validation_beam_search = _sid_validation_beam_size is not None
        if self.activate_constrained_beam_search or self.activate_validation_beam_search:
            if self.activate_beam_search:
                raise ValueError("SID beam search modes cannot be enabled together")
            if self.activate_constrained_beam_search and _sid_constrained_beam_size < 2:
                raise ValueError("sid_constrained_beam_size must be at least 2")
            if self.activate_validation_beam_search and _sid_validation_beam_size < 2:
                raise ValueError("sid_validation_beam_size must be at least 2")
            if _sid_length is None or _sid_length < 1:
                raise ValueError("sid_length must be set when constrained beam search is enabled")
            sid_category = config.get("sid_category", None)
            if not sid_category:
                raise ValueError("sid_category must be set when constrained beam search is enabled")
            if self.config.calculate_log_probs:
                raise ValueError("Constrained beam search requires actor-side log-probability recomputation")

            import hf_data

            self.sid_constrained_beam_size = _sid_constrained_beam_size
            self.sid_validation_beam_size = _sid_validation_beam_size
            self.num_sid_tokens = _sid_length
            self.end_think_marker = self.tokenizer.encode("</think>", add_special_tokens=False)
            if self.truncate_marker[: len(self.end_think_marker)] != self.end_think_marker:
                raise ValueError("The </think> separator does not extend the tokenizer's </think> marker")
            sid_sequences = hf_data.load_sid_indices(sid_category).values()
            self.sid_token_trie = build_sid_token_trie(
                self.tokenizer,
                sid_sequences,
                depth=self.num_sid_tokens,
            )

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]
        # generation_config may expose eos_token_id as a list (e.g. Qwen3 -> [151645, 151643]);
        # the constrained-beam path needs a single scalar id to append as the terminal token.
        primary_eos_token_id = eos_token_id[0] if isinstance(eos_token_id, (list, tuple)) else eos_token_id

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        for input_data in vllm_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        use_constrained_beam_search = self.activate_constrained_beam_search
        constrained_beam_size = getattr(self, "sid_constrained_beam_size", None)
        if is_validate and self.activate_validation_beam_search:
            use_constrained_beam_search = True
            constrained_beam_size = self.sid_validation_beam_size
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        if use_constrained_beam_search:
            reserved_tokens = self.num_sid_tokens + len(self.truncate_marker) + 1
            max_reasoning_tokens = self.config.response_length - reserved_tokens
            if max_reasoning_tokens < 1:
                raise ValueError("response_length is too short for reasoning plus constrained SID")
            kwargs["max_tokens"] = max_reasoning_tokens
            kwargs["min_tokens"] = 1

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            response_reasonings = []
            sampled_reasoning_lengths = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if use_constrained_beam_search:
                        reasoning_ids, sampled_length = prepare_reasoning_prefix(
                            response_ids,
                            end_think_marker=self.end_think_marker,
                            reasoning_separator=self.truncate_marker,
                            eos_token_id=primary_eos_token_id,
                            max_length=self.config.response_length - self.num_sid_tokens - 1,
                        )
                        response_reasonings.append(reasoning_ids)
                        sampled_reasoning_lengths.append(sampled_length)
                    elif self.activate_beam_search:
                        # keep only the reasoning span (up to and including </think>)
                        response_ids_truncated = truncate_at_end_think(
                            response_ids, marker=self.truncate_marker, clip_chars=20
                        )
                        response_reasonings.append(response_ids_truncated)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            # === SID Reasoner: constrained beam search over catalog SID paths ===
            if use_constrained_beam_search:
                input_prompt_ids = [
                    vllm_inputs[i]["prompt_token_ids"] + response_reasonings[i] for i in range(batch_size)
                ]
                constrained_sid_beams = vllm_constrained_beam_search(
                    self.inference_engine,
                    prompts_ids=input_prompt_ids,
                    sid_token_trie=self.sid_token_trie,
                    depth=self.num_sid_tokens,
                    beam_width=constrained_beam_size,
                    lora_requests=lora_requests,
                )
                constrained_sids = [sid_beam[0] for sid_beam in constrained_sid_beams]
                # Build a 1-D object array of length batch_size where each cell is
                # itself a numpy array of decoded SID strings. A per-sample beam can
                # be shorter than beam_width when the catalog trie runs out of valid
                # continuations; building a plain np.array([[...], ...]) then yields a
                # ragged->1-D array on some ranks and a rectangular 2-D array on others,
                # so DataProto.concat's np.concatenate across data-parallel workers
                # crashes with mismatched ndim. Forcing a 1-D outer array keeps the
                # shape identical on every rank while `.tolist()` in the reward manager
                # still works (each cell is an ndarray).
                sid_beam_predictions = np.empty(batch_size, dtype=object)
                for _j, sid_beam in enumerate(constrained_sid_beams):
                    sid_beam_predictions[_j] = np.array(
                        [self.tokenizer.decode(sid_ids, skip_special_tokens=False) for sid_ids in sid_beam],
                        dtype=object,
                    )
                non_tensor_batch["sid_beam_predictions"] = sid_beam_predictions
                response = [
                    reasoning_ids + sid_ids + [primary_eos_token_id]
                    for reasoning_ids, sid_ids in zip(response_reasonings, constrained_sids)
                ]

            # === SID Reasoner: beam search over semantic-ID tokens after reasoning ===
            elif self.activate_beam_search:
                input_prompt_ids = [
                    vllm_inputs[i]["prompt_token_ids"] + response_reasonings[i] for i in range(batch_size)
                ]
                response_beam_search = vllm_beam_search_concat(
                    self.inference_engine,
                    self.tokenizer,
                    prompts_ids=input_prompt_ids,
                    params=self.sampling_params.__copy__(),
                    beam_width=self.sid_beam_size,
                    depth=self.num_sid_tokens,
                )
                non_tensor_batch["beam_search_results"] = np.array(response_beam_search)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if use_constrained_beam_search:
                response_mask = torch.zeros_like(response, dtype=attention_mask.dtype)
                for index, sampled_length in enumerate(sampled_reasoning_lengths):
                    response_mask[index, :sampled_length] = 1
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope (batch size, 4, seq len)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if use_constrained_beam_search:
            batch["response_mask"] = response_mask
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if not self.config.free_cache_engine:
            return

        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=tags)
        else:
            self.inference_engine.wake_up()

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        self.inference_engine.reset_prefix_cache()

        if not self.config.free_cache_engine:
            return

        self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_reqest = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="simon_lora_path",
                peft_config=asdict(peft_config),
                lora_tensors=dict(weights),
            )
            self.inference_engine.llm_engine.add_lora(lora_reqest)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            patch_vllm_moe_model_weight_loader(model)
            model.load_weights(weights)


# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout(BaseRollout):
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase, which is engine in single worker process."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.tokenizer = model_config.tokenizer
        self.inference_engine: WorkerWrapperBase = None
        self.address = self._init_zeromq()
        self.lora_config = (
            {"max_loras": 1, "max_lora_rank": model_config.lora_rank} if model_config.lora_rank > 0 else {}
        )

        # https://github.com/vllm-project/vllm/issues/25171
        if config.layered_summon or config.expert_parallel_size > 1:
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock(f"/tmp/verl_vllm_zmq_{getpass.getuser()}.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{getpass.getuser()}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.asyncio.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        loop = asyncio.get_running_loop()
        self.zmq_loop_task = loop.create_task(self._loop_forever())

        return address

    def _get_free_port(self):
        ip = ray.util.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    async def _loop_forever(self):
        while True:
            try:
                message = await self.socket.recv()
                method, args, kwargs = pickle.loads(message)
                result = await self._execute_method(method, *args, **kwargs)
                await self.socket.send(pickle.dumps(result))
            except Exception as e:
                logger.exception(f"vLLMAsyncRollout _loop_forever error: {e}")
                os._exit(-1)

    def _init_worker(self, all_kwargs: list[dict[str, Any]]):
        """Initialize worker engine."""
        if not torch.distributed.is_initialized():
            initialize_global_process_group_ray()
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        device_name = "NPU" if is_npu_available else "GPU"
        all_kwargs[0]["local_rank"] = (
            0
            if not ray_noset_visible_devices()
            else int(ray.get_runtime_context().get_accelerator_ids()[device_name][0])
        )
        self.vllm_config = all_kwargs[0]["vllm_config"]
        if self.lora_config:
            lora_dtype = getattr(torch, self.config.dtype)
            self.vllm_config.lora_config = LoRAConfig(lora_dtype=lora_dtype, **self.lora_config)
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def _load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)
        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    async def _execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self._init_worker(*args, **kwargs)
        elif method == "load_model":
            return self._load_model(*args, **kwargs)
        elif method == "sleep" or method == "wake_up":
            raise ValueError("wake_up and sleep should not be called through ZeroMQ")
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if self.config.free_cache_engine:
            self.inference_engine.wake_up(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.free_cache_engine:
            self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_reqest = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="simon_lora_path",
                peft_config=asdict(peft_config),
                lora_tensors=dict(weights),
            )
            self.inference_engine.worker.add_lora(lora_reqest)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            model = self.inference_engine.worker.model_runner.model
            patch_vllm_moe_model_weight_loader(model)
            model.load_weights(weights)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode."""
        raise NotImplementedError

    # ==================== server mode public methods ====================

    def get_zeromq_address(self):
        return self.address
