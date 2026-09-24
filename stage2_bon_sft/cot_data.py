"""Multi-sample CoT loading for Stage-2 best-of-N selection.

Loads the K teacher-CoT subsets for one category, aligns them row-wise on
``(history_item_sid, item_sid)``, and expands them into a single frame with one
row per usable ``(history, target, trace)`` triple. ``score_cot_delta.py`` and
``select_cot_best_of_n.py`` both consume that frame: the former attaches a
predictive-utility delta to every row, the latter keeps at most one trace per
row (the argmax delta, and only when it is positive).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import data_io

# Columns that must agree across subsets for a row to be considered aligned.
ALIGNMENT_COLUMNS = ("history_item_sid", "item_sid")
ALIGNMENT_KEY = "_source_index"


def _valid_reasoning(value):
    """Mirror ReasoningActivationDataset's own validity filter."""
    if pd.isna(value):
        return False
    text = str(value).strip()
    if text == "":
        return False
    if text.startswith("<think>") and "</think>" not in text:
        return False
    return True


def load_cot_frames(cot_dirs):
    """Load the K independently sampled CoT subsets as DataFrames.

    ``cot_dirs`` are ``./data/<Category>/cot/sample_N/`` directories, e.g.
    ``["./data/Video_Games/cot/sample_1", "./data/Video_Games/cot/sample_2"]``.
    Position ``i`` in the list is the ``_cot_source`` ``i`` recorded downstream,
    so the order must stay fixed between scoring and selection.
    """
    return [data_io.load_cot(directory) for directory in cot_dirs]


def build_expanded_frame(frames, drop_incomplete=False, verbose=True):
    """Align K CoT frames row-wise and expand them into one candidate frame.

    Returns a DataFrame with the original columns plus ``_cot_source`` (which
    subset the trace came from). Each surviving ``(history, target)`` row
    contributes one candidate row per subset that supplied a usable trace.

    Rows whose trace is missing/unclosed in some subsets keep the subsets that
    worked, so a bad sample in one subset never drops that row entirely. Pass
    ``drop_incomplete=True`` to instead require all K.
    """
    if not frames:
        raise ValueError("need at least one CoT frame")

    n_sources = len(frames)
    keyed = all(ALIGNMENT_KEY in f.columns for f in frames)

    if keyed:
        indexed = [f.set_index(ALIGNMENT_KEY, drop=False) for f in frames]
        common = indexed[0].index
        for f in indexed[1:]:
            common = common.intersection(f.index)
        common = common.sort_values()
        if common.has_duplicates:
            raise ValueError(f"duplicate {ALIGNMENT_KEY} values in a CoT subset")
        aligned = [f.loc[common] for f in indexed]
    else:
        lengths = {len(f) for f in frames}
        if len(lengths) != 1:
            raise ValueError(
                f"CoT subsets have no '{ALIGNMENT_KEY}' column and differing "
                f"lengths {sorted(lengths)}; cannot align positionally"
            )
        aligned = [f.reset_index(drop=True) for f in frames]

    # Alignment must be semantic, not just positional: the K traces have to
    # describe the same (history, target) pair or the weights are meaningless.
    base = aligned[0]
    for i, other in enumerate(aligned[1:], start=1):
        for col in ALIGNMENT_COLUMNS:
            if col not in base.columns or col not in other.columns:
                continue
            mismatches = (base[col].to_numpy() != other[col].to_numpy()).sum()
            if mismatches:
                raise ValueError(
                    f"CoT subset {i} disagrees with subset 0 on '{col}' for "
                    f"{mismatches} rows -- subsets are not row-aligned"
                )

    valid = [f["reasoning_path"].map(_valid_reasoning).to_numpy() for f in aligned]
    valid_count = sum(v.astype(int) for v in valid)
    keep = valid_count == n_sources if drop_incomplete else valid_count > 0

    parts = []
    for k, frame in enumerate(aligned):
        take = keep & valid[k]
        if not take.any():
            continue
        part = frame.loc[take].copy()
        part["_cot_source"] = k
        parts.append(part)

    expanded = pd.concat(parts, ignore_index=True)

    if verbose:
        rows = int(keep.sum())
        dropped = int((~keep).sum())
        print(
            f"[cot-expand] {n_sources} subsets | {rows} aligned rows "
            f"({dropped} dropped) -> {len(expanded)} candidate traces"
        )
    return expanded


def read_source_index_file(path):
    """Read a newline-delimited ``_source_index`` manifest into a sorted array.

    Blank lines and ``#`` comments are ignored so a manifest can carry a header
    describing which chunk it is. Used by the self-evolution loop to train on an
    arbitrary subset of rows rather than the ``--limit`` prefix.
    """
    values = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values.append(int(line))
    if not values:
        raise ValueError(f"source index file is empty: {path}")
    unique = np.unique(np.asarray(values, dtype=np.int64))
    if len(unique) != len(values):
        raise ValueError(
            f"source index file has {len(values) - len(unique)} duplicate ids: {path}")
    return unique


def slice_source_indices(expanded, limit=0, source_index_file=None, label="",
                         verbose=True):
    """Restrict an aligned K-way frame to a subset of ``_source_index``.

    ``score_cot_delta.py`` and ``select_cot_best_of_n.py`` must slice
    *identically* or the delta join drops candidates for the wrong reason
    (unscored, not bad), so both call this one function.

    ``source_index_file`` selects the chunk and is validated against the full
    frame, so a manifest naming rows that do not exist fails loudly instead of
    silently shrinking the round. ``limit`` then truncates whatever is left,
    which makes ``--limit`` a cheap smoke test *within* a chunk. With no
    manifest, ``limit`` keeps the first N ids of the whole frame -- exactly the
    behaviour both scripts had before this helper existed.
    """
    note = []
    if source_index_file:
        wanted = read_source_index_file(source_index_file)
        present = expanded["_source_index"].unique()
        missing = np.setdiff1d(wanted, present, assume_unique=False)
        if len(missing):
            raise ValueError(
                f"{len(missing)}/{len(wanted)} ids in {source_index_file} are absent "
                f"from the aligned frame (first few: {missing[:5].tolist()}); the "
                "manifest was built against different --cot_dirs or "
                "--drop_incomplete_cot settings"
            )
        expanded = expanded[expanded["_source_index"].isin(set(wanted.tolist()))]
        expanded = expanded.reset_index(drop=True)
        note.append(f"--source_index_file {source_index_file} ({len(wanted)} ids)")
    if limit:
        keep = set(expanded["_source_index"].unique()[:limit])
        expanded = expanded[expanded["_source_index"].isin(keep)].reset_index(drop=True)
        note.append(f"--limit {limit}")
    if note:
        if verbose:
            print(f"[{label or 'slice'}] {' + '.join(note)} -> {len(expanded)} expanded rows",
                  flush=True)
    return expanded
