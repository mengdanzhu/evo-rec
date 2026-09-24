import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

_cpu_threads = _os.environ.get("SIDR_EVAL_CPU_THREADS", "1")
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ[_thread_var] = _cpu_threads

import json
import logging
import os
import random

import fire
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

import data_io
from data_Qwen3 import ReasoningEvalDataset
from verl.workers.rollout.sid_constrained_decoding import (
    build_sid_token_trie,
    prepare_reasoning_prefix,
    vllm_constrained_beam_search,
)


CATEGORY_NAMES = {
    "Industrial_and_Scientific": "industrial and scientific items",
    "Office_Products": "office products",
    "Toys_and_Games": "toys and games",
    "Sports": "sports and outdoors",
    "Books": "books",
    "Video_Games": "video games",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def batched(values, batch_size):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def extract_cot(tokenizer, response_ids, sampled_length):
    text = tokenizer.decode(response_ids[:sampled_length], skip_special_tokens=False).strip()
    end_think = text.find("</think>")
    if end_think >= 0:
        text = text[:end_think]
    return text.strip()


def main(
    base_model: str = "./checkpoint/Video_Games/stage3/final",
    info_file: str = "./data/Video_Games/catalog/",
    category: str = "Video_Games",
    test_data_path: str = "./data/Video_Games/seqrec/test/",
    item_file: str = "./data/Video_Games/catalog/",
    index_file: str = "./data/Video_Games/catalog/",
    result_json_data: str = "./temp/test_results_Qwen3.json",
    batch_size: int = 32,
    K: int = 0,
    seed: int = 42,
    length_penalty: float = 0.0,
    max_new_tokens: int = 1024,
    num_beams: int = 10,
    padding_side: str = "left",
    max_prompt_length: int = 1024,
    sid_length: int = 3,
    gpu_memory_utilization: float = 0.8,
    max_num_batched_tokens: int = 32768,
    max_num_seqs: int = 4096,
    enforce_eager: bool = False,
    logprobs_mode: str = "processed_logprobs",
):
    """Evaluate thinking-mode recommendations with the training validation path."""
    del info_file, K, padding_side
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if num_beams < 2:
        raise ValueError("num_beams must be at least 2")
    if sid_length != 3:
        raise ValueError("SID recommendation metrics require exactly three semantic tokens")
    if length_penalty != 0.0:
        raise ValueError("training-style fixed-depth SID beam search requires length_penalty=0.0")
    if max_num_batched_tokens < max_prompt_length + max_new_tokens:
        raise ValueError("max_num_batched_tokens must cover at least one full sequence")

    logging.basicConfig(level=logging.INFO)
    set_seed(seed)

    llm = LLM(
        model=base_model,
        max_model_len=max_prompt_length + max_new_tokens,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        logprobs_mode=logprobs_mode,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
        seed=seed,
        enforce_eager=enforce_eager,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    end_think_marker = tokenizer.encode("</think>", add_special_tokens=False)
    reasoning_separator = tokenizer.encode("</think>\n\n", add_special_tokens=False)
    if reasoning_separator[: len(end_think_marker)] != end_think_marker:
        raise ValueError("The </think> separator does not extend the tokenizer's </think> marker")
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token")

    reserved_tokens = sid_length + len(reasoning_separator) + 1
    max_reasoning_tokens = max_new_tokens - reserved_tokens
    if max_reasoning_tokens < 1:
        raise ValueError("max_new_tokens is too short for reasoning plus constrained SID decoding")

    sid_token_trie = build_sid_token_trie(
        tokenizer,
        data_io.load_sid_indices(category).values(),
        depth=sid_length,
    )

    prompt_category = CATEGORY_NAMES.get(category, "items")
    val_dataset = ReasoningEvalDataset(
        data_file=test_data_path,
        item_file=item_file,
        index_file=index_file,
        sample=-1,
        tokenizer=tokenizer,
        max_len=max_prompt_length + max_new_tokens,
        category=prompt_category,
        test=True,
        seed=seed,
    )
    encodings = [val_dataset[index] for index in range(len(val_dataset))]
    test_data = val_dataset.get_all()
    if len(encodings) != len(test_data):
        raise RuntimeError("Evaluation prompts and result rows have different lengths")
    overlong_prompts = [len(encoding["input_ids"]) for encoding in encodings if len(encoding["input_ids"]) > max_prompt_length]
    if overlong_prompts:
        raise ValueError(
            f"Found {len(overlong_prompts)} prompts longer than the training limit "
            f"of {max_prompt_length} tokens (maximum: {max(overlong_prompts)})"
        )

    reasoning_sampling_params = SamplingParams(
        n=1,
        logprobs=0,
        max_tokens=max_reasoning_tokens,
        min_tokens=1,
        repetition_penalty=1.0,
        detokenize=False,
        best_of=1,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        temperature=0.0,
    )

    predictions = []
    cots = []
    prompt_cots = []
    encoding_batches = list(batched(encodings, batch_size))
    for encoding_batch in tqdm(encoding_batches, desc="Generating reasoning and constrained SID beams"):
        prompt_ids = [list(encoding["input_ids"]) for encoding in encoding_batch]
        reasoning_outputs = llm.generate(
            prompts=[{"prompt_token_ids": ids} for ids in prompt_ids],
            sampling_params=reasoning_sampling_params,
            use_tqdm=False,
        )
        if len(reasoning_outputs) != len(prompt_ids):
            raise RuntimeError("vLLM returned an unexpected reasoning batch size")

        reasoning_prefixes = []
        batch_cots = []
        for output in reasoning_outputs:
            response_ids = list(output.outputs[0].token_ids)
            reasoning_ids, sampled_length = prepare_reasoning_prefix(
                response_ids,
                end_think_marker=end_think_marker,
                reasoning_separator=reasoning_separator,
                eos_token_id=tokenizer.eos_token_id,
                max_length=max_new_tokens - sid_length - 1,
            )
            reasoning_prefixes.append(reasoning_ids)
            batch_cots.append(extract_cot(tokenizer, response_ids, sampled_length))

        sid_beams = vllm_constrained_beam_search(
            llm,
            prompts_ids=[
                current_prompt_ids + reasoning_ids
                for current_prompt_ids, reasoning_ids in zip(prompt_ids, reasoning_prefixes)
            ],
            sid_token_trie=sid_token_trie,
            depth=sid_length,
            beam_width=num_beams,
        )
        short_beams = [len(sample_beams) for sample_beams in sid_beams if len(sample_beams) != num_beams]
        if short_beams and logprobs_mode == "processed_logprobs":
            raise RuntimeError(
                "Constrained SID beam search did not return the requested beam width "
                f"of {num_beams} (minimum returned: {min(short_beams)}). Ensure the "
                "vLLM engine uses logprobs_mode='processed_logprobs'."
            )
        if short_beams:
            # Only reachable with a non-default logprobs_mode, which exists purely as a
            # diagnostic knob for reproducing verl's in-training validation numbers.
            logging.warning(
                "logprobs_mode=%s produced %d samples with fewer than %d beams (minimum %d)",
                logprobs_mode, len(short_beams), num_beams, min(short_beams),
            )
        predictions.extend(
            [
                tokenizer.decode(sid_ids, skip_special_tokens=False)
                for sid_ids in sample_beams
            ]
            for sample_beams in sid_beams
        )
        cots.extend(batch_cots)
        prompt_cots.extend(
            tokenizer.decode(current_prompt_ids + reasoning_ids, skip_special_tokens=False)
            for current_prompt_ids, reasoning_ids in zip(prompt_ids, reasoning_prefixes)
        )

    output_lengths = {len(test_data), len(predictions), len(cots), len(prompt_cots)}
    if len(output_lengths) != 1:
        raise RuntimeError("Evaluation generated inconsistent numbers of result rows")

    for sample, prediction, cot, prompt_cot in zip(
        test_data,
        predictions,
        cots,
        prompt_cots,
    ):
        sample["prompt_cot"] = prompt_cot
        sample["predict"] = prediction
        sample["cot"] = cot
        sample.pop("dedup", None)

    result_dir = os.path.dirname(result_json_data)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    with open(result_json_data, "w", encoding="utf-8") as handle:
        json.dump(test_data, handle, indent=4)


if __name__ == "__main__":
    fire.Fire(main)
