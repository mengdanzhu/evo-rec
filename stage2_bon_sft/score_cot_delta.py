"""Score every CoT trace with a Stage-2 model: how much does it help the answer?

For each ``(history h, target SID y*)`` row and each candidate trace ``z_k`` we
compute, under one frozen scorer model, the summed token log-probability of the
answer span in two conditions

    log p(y* | h, z_k)   assistant = "<think>\\n{z_k}\\n</think>\\n\\n{y*}"   (think)
    log p(y* | h)        assistant = "<think>\\n</think>\\n\\n{y*}"           (non-think)

and report their difference::

    delta_k = log p(y* | h, z_k) - log p(y* | h)

``delta_k > 0`` means the trace made the gold SID more likely than not thinking
at all. The non-think condition uses the repo's own convention -- the empty
``"<think>\\n</think>\\n\\n"`` block from ``data_Qwen3.py`` -- and keeps the
*identical* system/user prompt, so the two conditions differ **only** by the
presence of ``z_k`` and the gap is attributable to the trace alone.

Only the ``{y*}`` tokens are scored (never the trace, never the closing
``<|im_end|>``), and their token ids are identical in both conditions, so
``delta`` is a clean per-trace scalar in nats.

Downstream, ``select_cot_best_of_n.py`` keeps the ``argmax_k delta_k`` trace of
each row, provided it clears ``--min_delta``. Because ``log p(y*|h)`` is shared
by all K candidates of a row, that argmax is equivalent to maximising the
rationale-conditioned likelihood of the target SID; the baseline is still
recorded because it is what makes ``delta`` interpretable across rows.

Run with one process per GPU::

    torchrun --nproc_per_node=8 stage2_bon_sft/score_cot_delta.py \\
        --scorer_model checkpoint/<CATEGORY>/stage2_scorer/final_checkpoint \\
        --category Video_Games \\
        --cot_dirs ./data/Video_Games/cot/sample_1 ... \\
        --output stage2_bon_sft/deltas/<name>.parquet
"""

import argparse
import json
import os
import sys
import time
from array import array
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[0]
for path in (str(REPO_ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cot_data import build_expanded_frame, load_cot_frames, slice_source_indices  
from data_Qwen3 import ReasoningActivationDataset  # noqa: E402

INSTRUCTION = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\nCan you recommend the next item for the user based on their "
    "interaction history?\n"
)
# The repo's non-think convention (data_Qwen3.py, SidNextItemEvalDataset).
NOTHINK_BLOCK = "<think>\n</think>\n\n"
BASELINE_SOURCE = -1  # _cot_source marker for the no-CoT condition


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scorer_model", required=True,
                   help="Stage-2 checkpoint used to score the traces.")
    p.add_argument("--category", default="Video_Games")
    p.add_argument("--cot_dirs", nargs="+", required=True,
                   help="./data/<Category>/cot/sample_N/ directories, in a "
                        "fixed order shared with select_cot_best_of_n.py.")
    p.add_argument("--output", required=True, help="Parquet file to write.")
    p.add_argument("--batch_tokens", type=int, default=24576,
                   help="Token budget per micro-batch (length-bucketed).")
    p.add_argument("--max_batch_size", type=int, default=48)
    p.add_argument("--cutoff_len", type=int, default=1024,
                   help="Must match the training cutoff so traces are scored on "
                        "the same text the trainer will see.")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--drop_incomplete_cot", action="store_true")
    p.add_argument("--limit", type=int, default=0,
                   help="Debug: keep only the first N expanded rows.")
    p.add_argument("--source_index_file", default=None,
                   help="Newline-delimited _source_index ids to score. "
                        "select_cot_best_of_n.py must be handed the same file "
                        "or its delta join will trip on unscored candidates.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def torch_dtype_from(name):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return dist.get_rank(), dist.get_world_size(), local_rank
    return 0, 1, 0


def log0(rank, *args):
    if rank == 0:
        print(*args, flush=True)


def build_dataset(args, tokenizer, cot_frame):
    """A ReasoningActivationDataset over the expanded frame, for its helpers."""
    cat = args.category
    dataset = ReasoningActivationDataset(
        reasoning_train_file=f"./data/{cat}/reasoning/",
        item_file=f"./data/{cat}/catalog/",
        index_file=f"./data/{cat}/catalog/",
        tokenizer=tokenizer,
        max_len=args.cutoff_len,
        sample=-1,
        seed=args.seed,
        category=cat,
    )
    dataset.data = cot_frame.reset_index(drop=True)
    return dataset


def make_prompt_ids(tokenizer, dataset, history_str):
    """Chat-template prefix up to and including ``<|im_start|>assistant\\n``."""
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": dataset.generate_prompt_title(history_str)},
    ]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors=None)


def think_block(reasoning):
    return f"<think>\n{reasoning.strip()}\n</think>\n\n"


def verify_against_dataset(tokenizer, dataset, rank, probes=8):
    """Hard guarantee that our rebuild reproduces the trainer's tokens.
    """
    checked = 0
    for idx in range(min(len(dataset.data), probes * 40)):
        sample = dataset.pre(idx)
        if sample is None:
            continue
        history = dataset.get_history(dataset.data.iloc[idx])
        prompt_ids = make_prompt_ids(tokenizer, dataset, history["history_str"])
        response = think_block(history["reasoning"]) + history["target_sid"]
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        rebuilt = prompt_ids + response_ids
        # +2 covers the "<|im_end|>\n" the full template appends after the answer.
        if len(rebuilt) + 2 > dataset.max_len:
            continue
        trainer = sample["input_ids"]
        if rebuilt != trainer[: len(rebuilt)]:
            raise RuntimeError(
                "rebuilt prompt does not match ReasoningActivationDataset.pre() at "
                f"row {idx}: rebuilt[{len(rebuilt)}] vs trainer[{len(trainer)}]"
            )
        first_label = next(i for i, v in enumerate(sample["labels"]) if v != -100)
        if first_label != len(prompt_ids):
            raise RuntimeError(
                f"row {idx}: assistant span starts at {first_label}, "
                f"expected {len(prompt_ids)}"
            )
        checked += 1
        if checked >= probes:
            break
    if checked == 0:
        raise RuntimeError("could not verify any row against the trainer dataset")
    log0(rank, f"[verify] {checked} rows match ReasoningActivationDataset token-for-token")


def build_tasks(args, tokenizer, dataset, cot_frame, rank, world_size):
    """Enumerate every scoring task; materialise token ids for this rank only.

    Returns ``(meta, shard)`` where ``meta[i] = (source_index, cot_source,
    n_target)`` for *all* tasks in a deterministic global order, and ``shard``
    maps the task indices owned by this rank to their token ids. Sharding at
    build time keeps peak memory at 1/world_size and skips the tokenisation of
    traces this rank will never score.
    """
    source_index = cot_frame["_source_index"].to_numpy()
    cot_source = cot_frame["_cot_source"].to_numpy()
    nothink_ids = tokenizer.encode(NOTHINK_BLOCK, add_special_tokens=False)

    # One prompt + one answer per unique (history, target) row.
    unique = cot_frame.drop_duplicates("_source_index")
    prompts, targets = {}, {}
    for i in range(len(unique)):
        row = unique.iloc[i]
        history = dataset.get_history(row)
        if history is None:
            raise RuntimeError(
                f"_source_index={row['_source_index']} has unusable reasoning; "
                "build_expanded_frame should have dropped it"
            )
        key = int(row["_source_index"])
        prompts[key] = array("i", make_prompt_ids(tokenizer, dataset, history["history_str"]))
        targets[key] = array("i", tokenizer.encode(history["target_sid"],
                                                   add_special_tokens=False))
    log0(rank, f"[tasks] cached {len(prompts)} prompts / answers")

    meta = []
    shard = {}
    seen = set()
    for i in range(len(cot_frame)):
        key = int(source_index[i])
        n_target = len(targets[key])
        if key not in seen:
            seen.add(key)
            index = len(meta)
            meta.append((key, BASELINE_SOURCE, n_target))
            if index % world_size == rank:
                shard[index] = prompts[key] + array("i", nothink_ids) + targets[key]
        index = len(meta)
        meta.append((key, int(cot_source[i]), n_target))
        if index % world_size == rank:
            reasoning = dataset.get_history(cot_frame.iloc[i])["reasoning"]
            ids = tokenizer.encode(think_block(reasoning), add_special_tokens=False)
            shard[index] = prompts[key] + array("i", ids) + targets[key]

    over = sum(1 for ids in shard.values() if len(ids) > args.cutoff_len)
    log0(rank, f"[tasks] {len(meta)} sequences ({len(seen)} baselines); this rank "
               f"owns {len(shard)}, {over} of them exceed cutoff_len="
               f"{args.cutoff_len} and are clipped from the left (as in training)")
    return meta, shard


def clip_left(ids, n_target, cutoff):
    """Trim context from the left, exactly like ReasoningActivationDataset."""
    if len(ids) <= cutoff:
        return ids
    return ids[-max(cutoff, n_target + 1):]


@torch.no_grad()
def score_shard(model, meta, shard, args, rank, device, pad_id):
    """Return (task indices, summed answer log-prob) for this rank's tasks."""
    # Length bucketing keeps padding waste near zero without changing results.
    ordered = sorted(shard.items(), key=lambda item: len(item[1]))
    order = np.empty(len(ordered), dtype=np.int64)
    values = np.empty(len(ordered), dtype=np.float64)

    written = 0
    cursor = 0
    start = time.time()
    while cursor < len(ordered):
        longest = len(ordered[cursor][1])
        size = max(1, min(args.max_batch_size, args.batch_tokens // max(longest, 1)))
        chunk = ordered[cursor:cursor + size]
        cursor += len(chunk)

        seqs = [clip_left(ids, meta[i][2], args.cutoff_len) for i, ids in chunk]
        width = max(len(s) for s in seqs)
        # Right padding: a plain forward derives positions as arange(seq_len),
        # so the real tokens must start at position 0.
        input_ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
        attention = torch.zeros((len(seqs), width), dtype=torch.long)
        labels = torch.full((len(seqs), width), -100, dtype=torch.long)
        for j, (seq, (task_index, _)) in enumerate(zip(seqs, chunk)):
            n = len(seq)
            input_ids[j, :n] = torch.tensor(seq, dtype=torch.long)
            attention[j, :n] = 1
            n_target = meta[task_index][2]
            labels[j, n - n_target:n] = input_ids[j, n - n_target:n]

        input_ids = input_ids.to(device, non_blocking=True)
        attention = attention.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(input_ids=input_ids, attention_mask=attention).logits
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        per_token = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shift_labels.shape)
        mask = (shift_labels != -100).to(per_token.dtype)
        logprob = -(per_token * mask).sum(dim=-1).double().cpu().numpy()

        for j, (task_index, _) in enumerate(chunk):
            order[written] = task_index
            values[written] = logprob[j]
            written += 1

        if rank == 0 and written // 8192 != (written - len(chunk)) // 8192:
            done = written / max(len(ordered), 1)
            elapsed = time.time() - start
            print(f"[score] {written}/{len(ordered)} ({done:.1%}) "
                  f"elapsed {elapsed / 60:.1f}m "
                  f"eta {(elapsed / max(done, 1e-9) - elapsed) / 60:.1f}m", flush=True)

    return order, values


def gather_to_rank0(order, values, rank, world_size, device):
    if world_size == 1:
        return order, values
    counts = torch.zeros(world_size, dtype=torch.long, device=device)
    counts[rank] = len(order)
    dist.all_reduce(counts)
    largest = int(counts.max())

    padded_order = torch.full((largest,), -1, dtype=torch.long, device=device)
    padded_value = torch.zeros(largest, dtype=torch.float64, device=device)
    padded_order[: len(order)] = torch.as_tensor(order, device=device)
    padded_value[: len(values)] = torch.as_tensor(values, device=device)

    all_order = [torch.empty_like(padded_order) for _ in range(world_size)]
    all_value = [torch.empty_like(padded_value) for _ in range(world_size)]
    dist.all_gather(all_order, padded_order)
    dist.all_gather(all_value, padded_value)

    merged_order, merged_value = [], []
    for o, v in zip(all_order, all_value):
        keep = o >= 0
        merged_order.append(o[keep].cpu().numpy())
        merged_value.append(v[keep].cpu().numpy())
    return np.concatenate(merged_order), np.concatenate(merged_value)


def summarise(out, args):
    """Delta distribution stats for the scored candidate set."""
    summary = {
        "scorer_model": args.scorer_model,
        "cot_dirs": list(args.cot_dirs),
        "rows": int(out["_source_index"].nunique()),
        "pairs": int(len(out)),
        "delta_mean": float(out["delta"].mean()),
        "delta_std": float(out["delta"].std()),
        "delta_p5": float(out["delta"].quantile(0.05)),
        "delta_median": float(out["delta"].median()),
        "delta_p95": float(out["delta"].quantile(0.95)),
        "frac_delta_positive": float((out["delta"] > 0).mean()),
        "logp_with_mean": float(out["logp_with"].mean()),
        "logp_base_mean": float(out["logp_base"].mean()),
        "mean_target_tokens": float(out["n_target_tokens"].mean()),
    }
    # Within-row spread is what drives selection: the common per-row level
    # cancels when we take the argmax over candidates.
    centred = out["delta"] - out.groupby("_source_index")["delta"].transform("mean")
    summary["delta_within_row_std"] = float(centred.std())
    summary["delta_within_row_range_mean"] = float(
        (out.groupby("_source_index")["delta"].max()
         - out.groupby("_source_index")["delta"].min()).mean())
    for k in sorted(out["_cot_source"].unique()):
        summary[f"delta_mean_source_{int(k)}"] = float(
            out[out["_cot_source"] == k]["delta"].mean())
    return summary


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    log0(rank, f"[config] scorer={args.scorer_model}")
    log0(rank, f"[config] cot_dirs={args.cot_dirs} world_size={world_size} "
               f"cutoff_len={args.cutoff_len} dtype={args.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.scorer_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    frames = load_cot_frames(args.cot_dirs)
    cot_frame = build_expanded_frame(
        frames, drop_incomplete=args.drop_incomplete_cot, verbose=(rank == 0))
    if args.limit or args.source_index_file:
        cot_frame = slice_source_indices(
            cot_frame, limit=args.limit,
            source_index_file=args.source_index_file,
            label="config", verbose=(rank == 0))

    dataset = build_dataset(args, tokenizer, cot_frame)
    verify_against_dataset(tokenizer, dataset, rank)
    meta, shard = build_tasks(args, tokenizer, dataset, cot_frame, rank, world_size)

    model = AutoModelForCausalLM.from_pretrained(
        args.scorer_model, torch_dtype=torch_dtype_from(args.dtype))
    vocab = model.get_input_embeddings().weight.shape[0]
    if vocab < len(tokenizer):
        raise RuntimeError(f"scorer vocab {vocab} < tokenizer {len(tokenizer)}: "
                           "this checkpoint lacks the SID tokens")
    model.to(device).eval()
    log0(rank, f"[model] loaded, vocab={vocab}")

    order, values = score_shard(
        model, meta, shard, args, rank, device, tokenizer.pad_token_id)
    order, values = gather_to_rank0(order, values, rank, world_size, device)

    if rank == 0:
        if len(order) != len(meta):
            raise RuntimeError(f"scored {len(order)} tasks, expected {len(meta)}")
        scored = pd.DataFrame({
            "_source_index": [meta[i][0] for i in order],
            "_cot_source": [meta[i][1] for i in order],
            "n_target_tokens": [meta[i][2] for i in order],
            "logprob": values,
        })
        baseline = (scored[scored["_cot_source"] == BASELINE_SOURCE]
                    .set_index("_source_index")["logprob"])
        out = scored[scored["_cot_source"] != BASELINE_SOURCE].copy()
        out["logp_base"] = out["_source_index"].map(baseline)
        if out["logp_base"].isna().any():
            raise RuntimeError("some traces have no matching baseline score")
        out = out.rename(columns={"logprob": "logp_with"})
        out["delta"] = out["logp_with"] - out["logp_base"]
        out = out.sort_values(["_source_index", "_cot_source"]).reset_index(drop=True)

        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(destination, index=False)

        summary = summarise(out, args)
        Path(str(destination) + ".summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        print(f"SCORING_DONE output={destination}", flush=True)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
