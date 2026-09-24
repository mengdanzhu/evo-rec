from __future__ import annotations

import ast
import os
from pathlib import Path

from datasets import load_dataset
from torch.utils.data import Dataset

from data_Qwen3 import mask_assistant_response_only
from stage2_bon_sft import sft_reasoning_activation as stage2


# Fallback source when no selection output is given: one raw teacher-CoT sample,
# i.e. single-CoT SFT rather than best-of-N.
REASONING_CATEGORY = os.environ.get("COT_REASONING_CATEGORY", "Video_Games")
REASONING_DIR = os.environ.get("COT_REASONING_DIR", "").strip()
# When set, the reasoning traces are read from this local JSONL (one merged
# record per line). This is what the best-of-N pipeline produces, so it is the
# normal path; the directory above is only the single-CoT fallback.
REASONING_JSONL = os.environ.get("COT_REASONING_JSONL", "").strip()
os.environ.setdefault("WANDB_MODE", "online")

SYSTEM_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
Can you recommend the next item for the user based on their interaction history?
"""


def _parse_history(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"history_item_sid must be a list, got {type(parsed).__name__}")
    return [str(item) for item in parsed]


def _normalize_reasoning(value) -> str | None:
    if value is None:
        return None
    reasoning = str(value).strip()
    if not reasoning:
        return None
    if reasoning.startswith("<think>"):
        if "</think>" not in reasoning:
            return None
        reasoning = reasoning[len("<think>"):reasoning.index("</think>")].strip()
    return reasoning or None


class GeneratedReasoningDataset(Dataset):
    """Drop-in replacement for the original reasoning-activation train dataset."""

    def __init__(
        self,
        reasoning_train_file,
        item_file,
        index_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        dedup=False,
    ):
        del reasoning_train_file, item_file, index_file, test, category
        if REASONING_JSONL:
            if not os.path.isfile(REASONING_JSONL):
                raise FileNotFoundError(f"COT_REASONING_JSONL not found: {REASONING_JSONL}")
            source = REASONING_JSONL
            data = load_dataset("json", data_files=REASONING_JSONL, split="train")
        else:
            directory = Path(
                REASONING_DIR
                or f"./data/{REASONING_CATEGORY}/cot/sample_1"
            )
            shards = sorted(str(p) for p in directory.glob("*.parquet"))
            if not shards:
                raise FileNotFoundError(
                    f"no *.parquet shards in {directory}; set COT_REASONING_JSONL "
                    f"to a selection output or COT_REASONING_DIR to a CoT sample"
                )
            source = str(directory)
            data = load_dataset("parquet", data_files=shards, split="train")
        required = {"history_item_sid", "item_sid", "reasoning_path"}
        missing = required - set(data.column_names)
        if missing:
            raise ValueError(f"{source} is missing required columns: {sorted(missing)}")

        valid_indices = [
            index
            for index, reasoning in enumerate(data["reasoning_path"])
            if _normalize_reasoning(reasoning) is not None
        ]
        data = data.select(valid_indices)
        if sample > 0 and sample < len(data):
            data = data.shuffle(seed=seed).select(range(sample))

        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dedup = dedup

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        history = _parse_history(row["history_item_sid"])
        target_sid = str(row["item_sid"]).strip()
        reasoning = _normalize_reasoning(row["reasoning_path"])
        if reasoning is None:
            raise RuntimeError(f"Invalid reasoning_path survived filtering at row {index}")
        if self.dedup and history and history[-1] == target_sid:
            raise RuntimeError("dedup=True is unsupported because it changes dataset indexing")

        history_text = ", ".join(history)
        user_prompt = (
            f"The user has sequentially interacted with items {history_text}. "
            "Can you recommend the next item for him? Let's think step by step before "
            "making recommendation. Directly output the item SID after thinking."
        )
        assistant_response = f"<think>\n{reasoning}\n</think>\n\n{target_sid}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
        input_ids, attention_mask, labels = mask_assistant_response_only(
            tokenizer=self.tokenizer,
            messages=messages,
            assistant_response=assistant_response,
            max_len=self.max_len,
            mask_eos=False,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


stage2.ReasoningActivationDataset = GeneratedReasoningDataset


if __name__ == "__main__":
    stage2.main()
