#!/usr/bin/env python3
"""Best-of-N rejection sampling: keep the highest-delta CoT trace per row.

The accept/reject signal is the predictive utility ``score_cot_delta.py``
computes under the frozen single-CoT Stage-2 scorer (Eq. 2 of the paper)::

    delta_k = log p(y* | h, z_k) - log p(y* | h)
              \\_______________/   \\____________/
               think mode, with     non-think mode,
               this trace           plain user history

``delta_k > 0`` means the trace made the gold SID *more* likely than emitting it
straight from the history, i.e. the reasoning paid for itself.

Selection is **argmax over the N candidates** (best-of-N), not first-accept: for
each row we score all N traces and keep the one with the largest delta, provided
it clears ``--min_delta``. Ties go to the lowest ``_cot_source``, which keeps the rule deterministic.

Rejection is the secondary effect: a row survives iff at least one candidate
clears the bar, and ``max_k delta_k > min_delta`` is exactly that condition.

Output is a JSONL with the three columns ``stage2_bon_sft/train_stage2.py``
reads (``history_item_sid``, ``item_sid``, ``reasoning_path``) plus provenance
(``_source_index``, ``_cot_source``, ``_cot_config``, ``_cot_delta``), so the
training side needs no new code path.

Usage::

    python stage2_bon_sft/select_cot_best_of_n.py \\
        --delta_file stage2_bon_sft/deltas/<name>.parquet \\
        --cot_dirs ./data/Video_Games/cot/sample_1 ./data/Video_Games/cot/sample_2 ... \\
        --output stage2_bon_sft/data/<name>.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[0]
for path in (str(REPO_ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cot_data import build_expanded_frame, load_cot_frames, slice_source_indices 
REQUIRED_COLUMNS = ("history_item_sid", "item_sid", "reasoning_path")
PROVENANCE_COLUMNS = ("_source_index", "_cot_source", "_cot_config", "_cot_delta")


def parse_args():

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--delta_file", required=True,
                   help="Parquet written by score_cot_delta.py.")
    p.add_argument("--cot_dirs", nargs="+", required=True,
                   help="./data/<Category>/cot/sample_N/ directories; index i "
                        "is _cot_source i. Order only breaks delta ties "
                        "(best-of-N ignores priority).")
    p.add_argument("--output", required=True, help="JSONL to write.")
    p.add_argument("--min_delta", type=float, default=0.0,
                   help="Keep the row only when its best delta > this. 0.0 is "
                        "the 'reasoning beats not reasoning' bar.")
    p.add_argument("--drop_incomplete_cot", action="store_true",
                   help="Require a usable trace in every subset (must match "
                        "whatever score_cot_delta.py was run with).")
    p.add_argument("--limit", type=int, default=0,
                   help="Debug: keep only the first N rows. Must match "
                        "score_cot_delta.py --limit, otherwise the unscored "
                        "rows trip the completeness check.")
    p.add_argument("--source_index_file", default=None,
                   help="Newline-delimited _source_index ids to select over. "
                        "Must match score_cot_delta.py --source_index_file, "
                        "otherwise the unscored rows trip the same check.")
    return p.parse_args()


def attach_deltas(expanded, delta_frame):
    """Join ``delta`` onto the aligned K-way candidate frame.

    Every candidate must carry a score -- a partial join would silently drop
    rows for the wrong reason (missing score, not a bad trace).
    """
    for column in ("_source_index", "_cot_source"):
        if column not in expanded.columns:
            raise ValueError(f"expanded frame is missing '{column}'")
        if column not in delta_frame.columns:
            raise ValueError(f"delta frame is missing '{column}'")
    if "delta" not in delta_frame.columns:
        raise ValueError("delta frame is missing 'delta'")

    keys = ["_source_index", "_cot_source"]
    lookup = delta_frame.drop_duplicates(keys).set_index(keys)["delta"]
    index = pd.MultiIndex.from_arrays([expanded["_source_index"], expanded["_cot_source"]])
    deltas = lookup.reindex(index).to_numpy()
    if np.isnan(deltas).any():
        missing = int(np.isnan(deltas).sum())
        raise ValueError(
            f"{missing}/{len(expanded)} candidate traces have no delta score; "
            "re-run score_cot_delta.py with the same --cot_dirs"
        )

    scored = expanded.copy()
    scored["_cot_delta"] = deltas
    return scored


def _first_accepted_source(scored, min_delta):
    """The trace the first-accept rule would have taken, per row (for stats)."""
    accepted = scored[scored["_cot_delta"] > min_delta]
    accepted = accepted.sort_values(["_source_index", "_cot_source"], kind="stable")
    first = accepted.drop_duplicates("_source_index", keep="first")
    return first.set_index("_source_index")[["_cot_source", "_cot_delta"]]


def select_best_of_n(expanded, delta_frame, n_configs, min_delta=0.0):
    """Keep, per row, the candidate with the largest delta if it clears the bar.

    ``expanded`` is the aligned K-way frame from ``build_expanded_frame``;
    ``delta_frame`` is joined on ``(_source_index, _cot_source)``.

    Returns ``(selected, stats)``. ``selected`` has one row per accepted
    ``_source_index``, carrying ``_cot_delta`` and the winning ``_cot_source``.
    A row whose *best* candidate still fails the bar is rejected outright, which
    is the same rejection criterion as first-accept -- only the kept trace
    differs.
    """
    scored = attach_deltas(expanded, delta_frame)

    # Descending delta, ascending source: the head of each row group is the
    # argmax, with ties resolved toward the earliest config.
    ranked = scored.sort_values(
        ["_source_index", "_cot_delta", "_cot_source"],
        ascending=[True, False, True], kind="stable")
    best = ranked.drop_duplicates("_source_index", keep="first")
    selected = best[best["_cot_delta"] > min_delta].reset_index(drop=True)

    total_rows = int(scored["_source_index"].nunique())
    positive = scored["_cot_delta"] > min_delta
    per_source = selected["_cot_source"].value_counts().sort_index()
    available = scored.groupby("_source_index")["_cot_source"].size()

    # Best-of-N always evaluates the whole budget, so the interesting numbers
    # are how much the argmax buys over a random draw and over first-accept.
    row_delta_mean = scored.groupby("_source_index")["_cot_delta"].mean()
    row_delta_range = (scored.groupby("_source_index")["_cot_delta"].max()
                       - scored.groupby("_source_index")["_cot_delta"].min())
    first_accept = _first_accepted_source(scored, min_delta)
    accepted_index = pd.Index(selected["_source_index"])
    winner = selected.set_index("_source_index")["_cot_source"]
    fa_source = first_accept["_cot_source"].reindex(accepted_index)
    fa_delta = first_accept["_cot_delta"].reindex(accepted_index)

    stats = {
        "selection_rule": "best_of_n_argmax_delta",
        "rows_total": total_rows,
        "rows_accepted": int(len(selected)),
        "rows_rejected": total_rows - int(len(selected)),
        "acceptance_rate": float(len(selected) / total_rows) if total_rows else 0.0,
        "candidates_total": int(len(scored)),
        "candidates_positive": int(positive.sum()),
        "frac_candidates_positive": float(positive.mean()) if len(scored) else 0.0,
        "min_delta": float(min_delta),
        "n_candidates_mean": float(available.mean()) if total_rows else 0.0,
        "selected_delta_mean": float(selected["_cot_delta"].mean()) if len(selected) else 0.0,
        "selected_delta_median": float(selected["_cot_delta"].median()) if len(selected) else 0.0,
        "selected_delta_p5": float(selected["_cot_delta"].quantile(0.05)) if len(selected) else 0.0,
        "row_delta_mean": float(row_delta_mean.reindex(accepted_index).mean())
                          if len(selected) else 0.0,
        "gain_over_random_pick": float(
            (selected.set_index("_source_index")["_cot_delta"]
             - row_delta_mean.reindex(accepted_index)).mean()) if len(selected) else 0.0,
        "gain_over_first_accept": float(
            (selected.set_index("_source_index")["_cot_delta"] - fa_delta).mean())
            if len(selected) else 0.0,
        "frac_rows_differ_from_first_accept": float(
            (winner != fa_source).mean()) if len(selected) else 0.0,
        "within_row_delta_range_mean": float(row_delta_range.reindex(accepted_index).mean())
                                       if len(selected) else 0.0,
        "selected_from_source": {int(k): int(per_source.get(k, 0)) for k in range(n_configs)},
    }
    return selected, stats


def main():
    args = parse_args()

    frames = load_cot_frames(args.cot_dirs)
    expanded = build_expanded_frame(
        frames, drop_incomplete=args.drop_incomplete_cot, verbose=True)
    # Same slicing as score_cot_delta.py, through the same function, so the two
    # stay joinable however they were narrowed.
    if args.limit or args.source_index_file:
        expanded = slice_source_indices(
            expanded, limit=args.limit,
            source_index_file=args.source_index_file, label="select")

    delta_frame = pd.read_parquet(args.delta_file)
    selected, stats = select_best_of_n(
        expanded, delta_frame, len(args.cot_dirs), min_delta=args.min_delta)

    missing = [c for c in REQUIRED_COLUMNS if c not in selected.columns]
    if missing:
        raise RuntimeError(f"selected frame is missing training columns: {missing}")

    names = {i: Path(name).name for i, name in enumerate(args.cot_dirs)}
    selected = selected.copy()
    selected["_cot_config"] = selected["_cot_source"].map(names)

    out = selected[list(REQUIRED_COLUMNS) + list(PROVENANCE_COLUMNS)]
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    out.to_json(destination, orient="records", lines=True, force_ascii=False)

    stats["cot_dirs"] = list(args.cot_dirs)
    stats["delta_file"] = args.delta_file
    stats["output"] = str(destination)
    stats["selected_from_config"] = {
        names[k]: v for k, v in stats["selected_from_source"].items()}
    Path(str(destination) + ".stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2), flush=True)
    print(f"SELECTION_DONE output={destination} rows={len(out)}", flush=True)


if __name__ == "__main__":
    main()
