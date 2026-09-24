"""Utility to merge multi-GPU FSDP actor checkpoints into a single-card Hugging Face model.

Stage 3 saves the actor as FSDP shards, which vLLM cannot load. This gathers
them into the standard HF layout that ``evaluation/run_eval.sh`` expects.

Normally invoked through ``stage3_rl/merge_fsdp_ckpt.sh``, which sets the
PYTHONPATH this needs. To call it directly, do so from the repo root so the
vendored top-level ``verl`` package resolves:

    PYTHONPATH=. python stage3_rl/merge_fsdp_checkpoint.py \
        --checkpoint ./checkpoint/Video_Games/stage3/global_step_2600

Writes to the sibling directory ``global_step_2600/actor_merged``. Pass that
path as ``MODEL`` to run_eval.sh, or copy its contents into
``checkpoint/<CATEGORY>/stage3/final/`` to match the released-checkpoint layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge FSDP actor checkpoint into single-card HF format.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to global_step_xx directory or its actor subfolder produced by VERL training.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to write merged weights. "
             "Defaults to a sibling of the actor directory named <actor>_merged.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass through to transformers AutoConfig when loading tokenizer/config files.",
    )
    parser.add_argument(
        "--use-cpu-init",
        action="store_true",
        help="Initialise the transformers model on CPU before loading weights (helps for large models).",
    )
    return parser.parse_args()


def _resolve_actor_dir(checkpoint: Path) -> Path:
    """Return the actor directory whether the input is global_step or actor itself."""
    if checkpoint.name == "actor" and checkpoint.is_dir():
        return checkpoint
    actor_dir = checkpoint / "actor"
    if actor_dir.is_dir():
        return actor_dir
    raise FileNotFoundError(f"Could not locate actor directory under {checkpoint}")


def main() -> None:
    args = _parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    actor_dir = _resolve_actor_dir(checkpoint_path)

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        # save to "{original_dir}_merged"
        output_dir = actor_dir.parent / f"{actor_dir.name}_merged"
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_dir = actor_dir / "huggingface"
    if not hf_dir.is_dir():
        raise FileNotFoundError(f"Expected Hugging Face config files under {hf_dir}")

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        target_dir=str(output_dir),
        hf_upload_path=None,
        private=False,
        test_hf_dir=None,
        tie_word_embedding=False,
        trust_remote_code=args.trust_remote_code,
        is_value_model=False,
        local_dir=str(actor_dir),
        hf_model_config_path=str(hf_dir),
        use_cpu_initialization=args.use_cpu_init,
    )

    merger = FSDPModelMerger(config)
    merger.merge_and_save()
    merger.cleanup()

    print(f"Merged checkpoint saved to {output_dir}")


if __name__ == "__main__":
    main()
