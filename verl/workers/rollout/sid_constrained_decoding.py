"""Shared SID-constrained decoding used by training validation and evaluation."""

from collections.abc import Callable
from typing import Any, Optional


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


def build_sid_token_trie(tokenizer: Any, sid_sequences: Any, depth: int) -> dict[tuple[int, ...], list[int]]:
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
    llm: Any,
    prompts_ids: list[list[int]],
    sid_token_trie: dict[tuple[int, ...], list[int]],
    depth: int,
    beam_width: int,
    lora_requests: Any = None,
    sampling_params_factory: Optional[Callable[..., Any]] = None,
) -> list[list[list[int]]]:
    """Decode ordered SID beams within catalog constraints."""
    if beam_width < 2:
        raise ValueError("Constrained SID beam width must be at least 2")
    if isinstance(lora_requests, list) and len(lora_requests) != len(prompts_ids):
        raise ValueError("Expected one LoRA request per prompt")

    if sampling_params_factory is None:
        from vllm import SamplingParams

        sampling_params_factory = SamplingParams

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
                    sampling_params_factory(
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