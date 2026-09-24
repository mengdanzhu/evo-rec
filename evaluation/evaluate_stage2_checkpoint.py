"""Score one Stage-2 checkpoint on the test split in thinking mode.

The model generates its reasoning trace, then trie-constrained beam search
decodes a catalog-valid SID; HR/NDCG@{1,5,10} are computed from the merged
predictions. 
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
METRIC_CUTOFFS = (1, 5, 10)


def _normalize_item(value):
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value).strip(' \n"')


def calculate_metrics(path, item_path):
    """Calculate Stage-2 HR/NDCG at 1, 5, and 10 from merged predictions."""
    import data_io

    known_sids = {
        line.split("\t")[0].strip()
        for line in data_io.load_info_lines(item_path)
    }
    with open(path, "r", encoding="utf-8") as handle:
        test_data = json.load(handle)

    max_beams = max((len(sample.get("predict", [])) for sample in test_data), default=0)
    if max_beams < max(METRIC_CUTOFFS):
        raise ValueError(f"predictions contain only {max_beams} beams; need at least 10")
    hits = {cutoff: 0.0 for cutoff in METRIC_CUTOFFS}
    ndcg = {cutoff: 0.0 for cutoff in METRIC_CUTOFFS}
    unknown_predictions = 0

    for sample in test_data:
        predictions = [_normalize_item(value) for value in sample.get("predict", [])]
        target = _normalize_item(sample.get("output", ""))
        unknown_predictions += sum(prediction not in known_sids for prediction in predictions)
        try:
            rank = predictions.index(target)
        except ValueError:
            continue
        for cutoff in METRIC_CUTOFFS:
            if rank < cutoff:
                hits[cutoff] += 1.0
                ndcg[cutoff] += 1.0 / math.log2(rank + 2)

    denominator = len(test_data)
    return {
        "rows": denominator,
        "num_beams": max_beams,
        "hr": {
            str(cutoff): hits[cutoff] / denominator if denominator else 0.0
            for cutoff in METRIC_CUTOFFS
        },
        "ndcg": {
            str(cutoff): ndcg[cutoff] / denominator if denominator else 0.0
            for cutoff in METRIC_CUTOFFS
        },
        "unknown_predictions": unknown_predictions,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Score a trained Stage-2 checkpoint on the test split in thinking "
            "mode, with trie-constrained beam search over the SID catalog."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cuda-list", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--sid-length", type=int, default=3)
    parser.add_argument("--thinking-batch-size", type=int, default=32)
    parser.add_argument("--thinking-max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upload-to-wandb", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-run-id")
    args = parser.parse_args()

    if args.num_beams < max(METRIC_CUTOFFS):
        parser.error(f"--num-beams must be at least {max(METRIC_CUTOFFS)}")
    if args.sid_length != 3:
        parser.error("--sid-length must be 3 for recommendation metrics")
    args.cuda_list = [value.strip() for value in args.cuda_list.split(",") if value.strip()]
    if not args.cuda_list:
        parser.error("--cuda-list must contain at least one GPU")
    if args.upload_to_wandb:
        missing = [
            name for name in ("wandb_project", "wandb_run_name", "wandb_run_id")
            if not getattr(args, name)
        ]
        if missing:
            parser.error(
                "--upload-to-wandb requires "
                + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )
    return args


def run_checked(command, env=None):
    print("+ " + " ".join(str(value) for value in command), flush=True)
    subprocess.run(command, env=env, check=True)


def data_locators(category):
    """Canonical ./data locators; the directories need not exist on disk."""
    root = REPO_ROOT / "data" / category
    return {
        "test": root / "seqrec" / "test",
        "info": root / "catalog",
        "item": root / "catalog",
        "index": root / "catalog",
    }


def split_test_data(test_locator, temp_dir, cuda_list, num_samples):
    temp_dir.mkdir(parents=True, exist_ok=True)
    for path in temp_dir.glob("*.csv"):
        path.unlink()
    for path in temp_dir.glob("*.json"):
        path.unlink()

    run_checked([
        sys.executable,
        str(SCRIPT_DIR / "split.py"),
        "--input_path", str(test_locator),
        "--output_path", str(temp_dir),
        "--cuda_list", ",".join(cuda_list),
        "--num_samples", str(num_samples),
    ])

    def has_data_row(path):
        if not path.exists():
            return False
        with open(path, "r", encoding="utf-8", newline="") as handle:
            rows = csv.reader(handle)
            next(rows, None)
            return next(rows, None) is not None

    return [
        gpu for gpu in cuda_list
        if has_data_row(temp_dir / f"{gpu}.csv")
    ]


def evaluator_command(args, gpu, paths, temp_dir):
    return [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_Qwen3_think.py"),
        "--base_model", str(Path(args.checkpoint).resolve()),
        "--info_file", str(paths["info"]),
        "--category", args.category,
        "--test_data_path", str(temp_dir / f"{gpu}.csv"),
        "--result_json_data", str(temp_dir / f"{gpu}.json"),
        "--num_beams", str(args.num_beams),
        "--sid_length", str(args.sid_length),
        "--length_penalty", "0.0",
        "--seed", str(args.seed),
        "--item_file", str(paths["item"]),
        "--index_file", str(paths["index"]),
        "--batch_size", str(args.thinking_batch_size),
        "--max_new_tokens", str(args.thinking_max_new_tokens),
    ]


def run_eval(args, output_root, paths):
    temp_dir = output_root / "temp" / "thinking"
    active_gpus = split_test_data(paths["test"], temp_dir, args.cuda_list, args.num_samples)
    if not active_gpus:
        raise RuntimeError("evaluation split produced no non-empty GPU shards")

    processes = []
    for gpu in active_gpus:
        cache_root = temp_dir / ".cache" / f"gpu{gpu}"
        cache_root.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": gpu,
            "TRITON_CACHE_DIR": str(cache_root / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "inductor"),
            "VLLM_CACHE_ROOT": str(cache_root / "vllm"),
        })
        command = evaluator_command(args, gpu, paths, temp_dir)
        print("+ " + " ".join(command), flush=True)
        processes.append((gpu, subprocess.Popen(command, env=env)))

    failures = []
    for gpu, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append(f"GPU {gpu}: exit {return_code}")
    if failures:
        raise RuntimeError(f"thinking evaluation failed ({', '.join(failures)})")

    merged_path = output_root / f"final_result_thinking_{args.category}.json"
    merged = []
    for gpu in active_gpus:
        with open(temp_dir / f"{gpu}.json", "r", encoding="utf-8") as handle:
            merged.extend(json.load(handle))
    with open(merged_path, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)

    full_metrics = calculate_metrics(str(merged_path), str(paths["info"]))
    metrics = {
        "rows": full_metrics["rows"],
        "num_beams": full_metrics["num_beams"],
        "hr": {str(cutoff): full_metrics["hr"][str(cutoff)] for cutoff in METRIC_CUTOFFS},
        "ndcg": {
            str(cutoff): full_metrics["ndcg"][str(cutoff)]
            for cutoff in METRIC_CUTOFFS
        },
        "unknown_predictions": full_metrics["unknown_predictions"],
        "predictions": str(merged_path),
    }
    print(
        f"[stage2/thinking] "
        f"HR@5={metrics['hr']['5']:.6f} HR@10={metrics['hr']['10']:.6f} | "
        f"NDCG@5={metrics['ndcg']['5']:.6f} NDCG@10={metrics['ndcg']['10']:.6f}",
        flush=True,
    )
    return metrics


def upload_metrics_to_wandb(args, report):
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        id=args.wandb_run_id,
        resume="must",
    )
    logged_metrics = {}
    metrics = report["metrics"]
    for cutoff in METRIC_CUTOFFS:
        logged_metrics[f"recsys_eval_thinking/hr_at_{cutoff}"] = metrics["hr"][str(cutoff)]
        logged_metrics[f"recsys_eval_thinking/ndcg_at_{cutoff}"] = metrics["ndcg"][str(cutoff)]
    run.log(logged_metrics)
    run.finish()
    print(f"Uploaded Stage-2 recommendation metrics to W&B run {args.wandb_run_id}")


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "metrics.json"
    if metrics_path.exists():
        metrics_path.unlink()
    paths = data_locators(args.category)
    report = {
        "mode": "thinking",
        "category": args.category,
        "checkpoint": str(checkpoint),
        "num_beams": args.num_beams,
        "num_samples": args.num_samples,
    }
    report["metrics"] = run_eval(args, output_root, paths)

    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Wrote Stage-2 evaluation metrics to {metrics_path}", flush=True)
    if args.upload_to_wandb:
        upload_metrics_to_wandb(args, report)


if __name__ == "__main__":
    main()