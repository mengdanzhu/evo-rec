"""Local dataset loader for Evo-Rec.

Every stage and the evaluator read their data through this module. 
"""

import os
import json
import functools
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent

DATA_ROOT = REPO_ROOT / "data"

CATEGORIES = ["Video_Games", "Office_Products", "Industrial_and_Scientific"]

SPLITS = ("train", "validation", "test")
KINDS = ("seqrec", "catalog", "reasoning", "cot", "rl")

# The K teacher-CoT samples consumed by Stage-2 best-of-N (the paper uses K=5).
COT_SAMPLES = ("sample_1", "sample_2", "sample_3", "sample_4", "sample_5")


# --------------------------------------------------------------------------- #
# low-level loading (cached)                                                   #
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=None)
def _read_dir(directory):
    """Read every ``*.parquet`` shard in ``directory`` as one DataFrame."""
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(
            f"missing data directory: {path}\n"
            f"Download the data archive and unpack it so that this path exists "
            f"(see the Data section of README.md)."
        )
    shards = sorted(path.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"no *.parquet shards in {path}; the directory exists but is empty"
        )
    if len(shards) == 1:
        return pd.read_parquet(shards[0])
    return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)


@functools.lru_cache(maxsize=None)
def _catalog(category):
    return _read_dir(DATA_ROOT / category / "catalog")


# --------------------------------------------------------------------------- #
# ./data path parsing                                                          #
# --------------------------------------------------------------------------- #
def parse_locator(path):
    """Resolve a ``./data`` path into ``(category, kind, split)``.

    Only the components after ``data`` matter, so a directory and a file inside
    it resolve identically::

        ./data/Video_Games/seqrec/test        -> ("Video_Games", "seqrec", "test")
        ./data/Video_Games/seqrec/test/x.csv  -> ("Video_Games", "seqrec", "test")
        ./data/Video_Games/catalog/           -> ("Video_Games", "catalog", "train")
        ./data/Video_Games/cot/sample_3/      -> ("Video_Games", "cot", "sample_3")
        ./data/general_reasoning/             -> (None, "general_reasoning", "train")

    A ``seqrec`` path without a split component defaults to ``train``.
    """
    parts = [p for p in str(path).replace("\\", "/").split("/") if p and p != "."]
    if "data" in parts:
        parts = parts[parts.index("data") + 1:]
    if not parts:
        raise ValueError(
            f"{path!r} is not a ./data path; expected e.g. "
            "./data/<Category>/seqrec/test/"
        )

    if parts[0] == "general_reasoning":
        return None, "general_reasoning", "train"

    category = parts[0]
    kind = parts[1] if len(parts) > 1 else "seqrec"
    if kind not in KINDS:
        raise ValueError(
            f"unknown data kind {kind!r} in {path!r}; expected one of {KINDS}"
        )

    split = "train"
    if kind == "seqrec" and len(parts) > 2 and parts[2] in SPLITS:
        split = parts[2]
    elif kind == "cot" and len(parts) > 2:
        split = parts[2]
    return category, kind, split


def resolve(path):
    """Map a ``./data`` path onto the real directory under ``DATA_ROOT``."""
    category, kind, split = parse_locator(path)
    if kind == "general_reasoning":
        return DATA_ROOT / "general_reasoning"
    directory = DATA_ROOT / category / kind
    if kind in ("seqrec", "cot"):
        directory = directory / split
    return directory


def infer_category(path):
    """Extract the category name from a ``./data`` path."""
    return parse_locator(path)[0]


def _stringify_list_columns(df):
    """Reproduce ``pd.read_csv`` semantics: list/array cells -> ``str([...])``.

    The parquet stores real Python lists for ``history_item_*`` columns, but the
    dataset classes call ``eval(...)`` on those cells, so list-typed columns are
    converted back to their string representation.
    """
    df = df.copy()
    for col in df.columns:
        sample = next((v for v in df[col].head(50) if v is not None), None)
        if isinstance(sample, (list, np.ndarray)):
            df[col] = df[col].map(
                lambda x: str(list(x)) if isinstance(x, (list, np.ndarray)) else x
            )
    return df


# --------------------------------------------------------------------------- #
# explicit APIs                                                                #
# --------------------------------------------------------------------------- #
def load_seqrec(category, split="train"):
    """Load a category's sequential-recommendation split."""
    if split not in SPLITS:
        raise ValueError(f"Unsupported seqrec split: {split}")
    return _stringify_list_columns(_read_dir(DATA_ROOT / category / "seqrec" / split))


def load_sequence_narratives(category):
    """Load the category-level reasoning narratives used by sequence tasks."""
    return _stringify_list_columns(_read_dir(DATA_ROOT / category / "reasoning"))


def load_cot_sample(category, sample):
    """Load one teacher-CoT sample, e.g. ``load_cot_sample("Video_Games", "sample_3")``."""
    return _stringify_list_columns(_read_dir(DATA_ROOT / category / "cot" / sample))


def load_item_features(category):
    """Return catalog item metadata keyed by item ID."""
    df = _catalog(category)
    features = {}
    for row in df.itertuples(index=False):
        features[str(row.item_id)] = {
            "title": row.title,
            "description": row.description,
            "brand": getattr(row, "brand", None),
            "categories": "",
        }
    return features


def load_sid_indices(category):
    """Return each catalog item's ordered SID-token sequence."""
    df = _catalog(category)
    return {
        str(row.item_id): list(row.sid_tokens)
        for row in df.itertuples(index=False)
    }


def load_sid_tokens(category):
    """Return the sorted, unique SID tokens that extend the model vocabulary."""
    return sorted({
        token
        for sid_tokens in load_sid_indices(category).values()
        for token in sid_tokens
    })


def load_item_narratives(category):
    """Return item-level SID/text narratives keyed by item ID."""
    df = _catalog(category)
    narratives = {}
    for row in df.itertuples(index=False):
        narrative = row.sid_interleaved_narrative
        if narrative is not None and not (
            isinstance(narrative, float) and np.isnan(narrative)
        ):
            narratives[str(row.item_id)] = {
                "sid_interleaved_narrative": narrative
            }
    return narratives


def load_general_reasoning():
    """Return decoded role/content messages for general-reasoning SFT."""
    df = _read_dir(DATA_ROOT / "general_reasoning")
    output = []
    for messages in df["messages"].tolist():
        # The column may contain nested JSON strings; decode to the actual list.
        for _ in range(3):
            if isinstance(messages, str):
                messages = json.loads(messages)
            else:
                break
        output.append(messages)
    return output


# --------------------------------------------------------------------------- #
# ./data path adapters                                                         #
# --------------------------------------------------------------------------- #
def load_df(path):
    """Load a ``seqrec`` or ``reasoning`` directory as a DataFrame.

    If ``path`` is a real file (e.g. the per-GPU chunk CSVs that
    ``evaluation/split.py`` materializes), it is read directly; otherwise it is
    treated as a ``./data`` directory.
    """
    if os.path.isfile(str(path)):
        return pd.read_csv(path)
    category, kind, split = parse_locator(path)
    if kind == "reasoning":
        return load_sequence_narratives(category)
    return load_seqrec(category, split)


def load_item_feat(item_file):
    """``./data/<Category>/catalog/`` -> ``{item_id: {title, description, ..}}``."""
    return load_item_features(infer_category(item_file))


def load_indices(index_file):
    """``./data/<Category>/catalog/`` -> ``{item_id: [sid_tokens]}``."""
    return load_sid_indices(infer_category(index_file))


def load_enhanced(json_file):
    """``./data/<Category>/catalog/`` -> ``{item_id: {sid_interleaved_narrative}}``.

    That narrative is the only field ``SidTextInterleaveItemDataset`` consumes.
    """
    return load_item_narratives(infer_category(json_file))


def load_cot(path):
    """``./data/<Category>/cot/sample_N/`` -> one teacher-CoT candidate frame."""
    category, kind, sample = parse_locator(path)
    if kind != "cot":
        raise ValueError(
            f"{path!r} is not a CoT path; expected ./data/<Category>/cot/sample_N/"
        )
    return load_cot_sample(category, sample)


def load_general(path=None):
    """``./data/general_reasoning/`` -> list of ``messages`` objects.

    Each element is a list of role/content dicts.
    """
    return load_general_reasoning()


def load_info_lines(info_file):
    """``./data/<Category>/catalog/`` -> ``semantic_id \\t title \\t item_id`` lines.

    Ordered by ``item_id`` (== the 0-based index the evaluator relies on).
    """
    df = _catalog(infer_category(info_file)).sort_values("item_id")
    lines = []
    for r in df.itertuples(index=False):
        lines.append(f"{r.sid}\t{r.title}\t{r.item_id}\n")
    return lines
