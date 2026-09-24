"""
Stage 2 ("cold-start reasoning activation") starts from the Stage-1 checkpoint and
SFTs on the integrated reasoning narratives: reason over the history inside a
``<think>...</think>`` block, then emit the target Semantic ID. Launch it with the
``deepspeed`` launcher (see ``sft_reasoning_activation.sh``).
"""

import os
import math
import time
import random
import argparse

import numpy as np
import torch
import deepspeed
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
from torch.utils.data.distributed import DistributedSampler
from data_Qwen3 import (
    ReasoningActivationDataset,
    SidHistory2SidSFTDataset,
    TitleSidTranslationDataset,
)

try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:  # pragma: no cover - wandb is optional
    _WANDB_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Small distributed helpers (kept intentionally un-encapsulated)              #
# --------------------------------------------------------------------------- #
def get_rank() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def is_rank_0() -> bool:
    return get_rank() == 0


def print_rank_0(msg):
    if is_rank_0():
        print(msg, flush=True)


def to_device(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def torch_dtype_from(dtype: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(dtype, torch.bfloat16)


def get_train_ds_config(offload, dtype, stage, micro_batch_size, grad_accum, world_size):
    """Build a self-contained DeepSpeed runtime config (no external helpers)."""
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {"device": device},
        "offload_optimizer": {"device": device},
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False,
    }
    if dtype == "fp16":
        precision = {"fp16": {"enabled": True, "loss_scale_window": 100}}
    elif dtype == "bf16":
        precision = {"bf16": {"enabled": True}}
    else:
        precision = {}

    config = {
        "train_batch_size": micro_batch_size * grad_accum * world_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "steps_per_print": 50,
        "zero_optimization": zero_opt_dict,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }
    config.update(precision)
    return config


def get_optimizer_grouped_parameters(model, weight_decay):
    """Only trainable params; no weight-decay on biases / norms."""
    no_decay = ["bias", "layer_norm.weight", "layernorm.weight", "ln_f.weight", "norm.weight"]
    groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(nd in n.lower() for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n.lower() for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return [g for g in groups if g["params"]]


@torch.no_grad()
def evaluation(model, eval_dataloader, device):
    """Mean cross-entropy loss / perplexity across an eval set (all-reduced)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in eval_dataloader:
        batch = to_device(batch, device)
        outputs = model(**batch, use_cache=False)
        total_loss = total_loss + outputs.loss.float()
        n_batches += 1

    model.train()
    if n_batches == 0:
        return float("inf"), float("inf")

    mean_loss = total_loss / n_batches
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(mean_loss, op=torch.distributed.ReduceOp.SUM)
        mean_loss = mean_loss / torch.distributed.get_world_size()
    mean_loss = mean_loss.item()
    try:
        ppl = math.exp(mean_loss)
    except OverflowError:
        ppl = float("inf")
    return ppl, mean_loss


def save_hf_checkpoint(model, tokenizer, save_dir):
    """Persist a plain Hugging Face checkpoint (rank-0 only).

    We train with ZeRO stage <= 2, where every rank holds a full copy of the
    parameters, so a plain ``save_pretrained`` on rank 0 is enough.
    """
    model_to_save = model.module if hasattr(model, "module") else model
    if is_rank_0():
        os.makedirs(save_dir, exist_ok=True)
        model_to_save.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


class TokenExtender:
    def __init__(self, data_path, dataset, index_file=".index.json"):
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file
        self.indices = None
        self.new_tokens = None

    def _load_data(self):
        import data_io
        self.indices = data_io.load_indices(os.path.join(self.data_path, self.dataset + self.index_file))

    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens
        if self.indices is None:
            self._load_data()
        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        return self.new_tokens


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _decode_tokens(tokens, tokenizer_ref):
    if not isinstance(tokens, (list, tuple)):
        return ""
    valid_ids = [tid for tid in tokens if isinstance(tid, int) and tid >= 0]
    if not valid_ids:
        return ""
    return tokenizer_ref.decode(valid_ids, skip_special_tokens=False)


def _preview_dataset(dataset, name, tokenizer_ref, max_samples=3):
    print(f"[Preview] {name}: displaying up to {max_samples} samples")
    preview_count = min(max_samples, len(dataset))
    for idx in range(preview_count):
        sample = dataset[idx]
        input_text = ""
        label_text = ""
        if isinstance(sample, dict):
            if "input_ids" in sample:
                input_text = _decode_tokens(sample["input_ids"], tokenizer_ref)
            if "labels" in sample:
                # Filter label padding tokens (e.g., -100) before decoding for readability
                label_ids = [tid for tid in sample["labels"] if isinstance(tid, int) and tid >= 0]
                label_text = _decode_tokens(label_ids, tokenizer_ref)
        print(f"Sample {idx + 1}:")
        if input_text:
            print(f"  Input : {input_text}")
        if label_text:
            print(f"  Label : {label_text}")
            print(f"  Length: {len(label_ids)} tokens")
        print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage-2 reasoning-activation SFT with an explicit DeepSpeed training loop."
    )

    # model / data
    parser.add_argument("--base_model", type=str,
                        default="./checkpoint/Video_Games/stage1/final_checkpoint",
                        help="Stage-1 checkpoint to initialise from.")
    parser.add_argument("--output_dir", type=str,
                        default="./checkpoint/Video_Games/stage2_scorer")
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    # --category is the single source of truth for the reasoning, eval, and
    # catalog data. No per-file path flags are exposed.
    parser.add_argument("--category", type=str, default="Video_Games")

    # training hyperparams
    parser.add_argument("--micro_batch_size", type=int, default=8, help="Per-GPU micro-batch size.")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--cutoff_len", type=int, default=1024)
    parser.add_argument("--num_warmup_steps", type=int, default=10)
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
                        choices=["linear", "cosine", "cosine_with_restarts",
                                 "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--early_stopping_patience", type=int, default=-1,
                        help="Stop if eval loss does not improve for N epochs. <=0 disables it "
                             "(default: run all epochs and save every one for recsys-metric selection).")
    parser.add_argument("--logging_steps", type=int, default=1)

    parser.add_argument("--gradient_checkpointing", action="store_true")

    # deepspeed / precision
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)

    # logging
    parser.add_argument("--wandb_project", type=str, default="EvoRec_Stage2")
    parser.add_argument("--wandb_run_name", type=str,
                        default="Video_Games_stage2_reasoning_activation_Qwen3-1.7B")
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "none"])

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


CATEGORY_DICT = {
    "Industrial_and_Scientific": "industrial and scientific items",
    "Office_Products": "office products",
    "Toys_and_Games": "toys and games",
    "Sports": "sports and outdoors",
    "Books": "books",
    "Video_Games": "video games",
}


def derive_data_paths(args):
    """Populate the remaining training data paths from ``--category``.

    ``data_io.py`` resolves these against ``DATA_ROOT``, so the run works from
    any working directory. Stage-2 evaluation uses explicit category/split APIs
    instead.
    """
    cat = args.category
    args.reasoning_train_file = f"./data/{cat}/reasoning/"
    args.sid_index_path = f"./data/{cat}/catalog/"
    args.item_meta_path = f"./data/{cat}/catalog/"


def build_datasets(args, tokenizer, category):
    """One reasoning-activation training set plus the three eval sets.

    Train  = ``<cat>_reasoning`` narratives (reason over history -> emit SID).
    Eval   = sid_pred (next-item SID) + title2sid / sid2title translations, the
             same three loss probes Stage-1 tracks.
    """
    train_data = ReasoningActivationDataset(
        reasoning_train_file=args.reasoning_train_file, item_file=args.item_meta_path,
        index_file=args.sid_index_path, tokenizer=tokenizer, max_len=args.cutoff_len,
        sample=args.sample, seed=args.seed, category=category)

    if is_rank_0():
        _preview_dataset(train_data, "ReasoningActivationDataset", tokenizer)

    val_sid = SidHistory2SidSFTDataset(
        data_category=args.category, split="validation", tokenizer=tokenizer,
        max_len=args.cutoff_len,
        sample=args.sample, seed=args.seed, category=category, test=False, mask_assistant=True)
    val_t2s = TitleSidTranslationDataset(
        data_category=args.category, tokenizer=tokenizer,
        max_len=args.cutoff_len, sample=args.sample, seed=args.seed, category=category,
        task_type='title2sid', test=False, mask_assistant=True)
    val_s2t = TitleSidTranslationDataset(
        data_category=args.category, tokenizer=tokenizer,
        max_len=args.cutoff_len, sample=args.sample, seed=args.seed, category=category,
        task_type='sid2title', test=False, mask_assistant=True)

    return train_data, val_sid, val_t2s, val_s2t


def main():
    args = parse_args()
    derive_data_paths(args)
    set_seed(args.seed)

    # --- distributed / device set-up (explicit, no wrapper) ---
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()
    args.global_rank = get_rank()
    world_size = get_world_size()

    # Gradient accumulation is hardcoded to 1 (mirrors Stage-1).
    # global batch = micro_batch * grad_accum * world_size = micro_batch * world_size.
    grad_accum = 1
    ds_config = get_train_ds_config(
        offload=args.offload,
        dtype=args.dtype,
        stage=args.zero_stage,
        micro_batch_size=args.micro_batch_size,
        grad_accum=grad_accum,
        world_size=world_size,
    )

    use_wandb = args.report_to == "wandb" and _WANDB_AVAILABLE and is_rank_0()
    if use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        wandb.login()
        init_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "config": vars(args),
        }
        if args.wandb_run_id:
            init_kwargs.update({"id": args.wandb_run_id, "resume": "allow"})
        wandb.init(**init_kwargs)

    category = CATEGORY_DICT.get(args.category, "items")
    print_rank_0(f"[Config] category={args.category} -> '{category}' | "
                 f"world_size={world_size} | micro_bs={args.micro_batch_size} | grad_accum={grad_accum}")

    # --- model + tokenizer (initialised from the Stage-1 checkpoint) ---
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch_dtype_from(args.dtype))
    print_rank_0(f"Loaded Stage-1 weights from {args.base_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    print_rank_0(f"Tokenizer length: {len(tokenizer)}")

    # --- ensure the SID tokens exist. The Stage-1 checkpoint already contains
    #     them (they were added + trained in Stage-1), so this is normally a
    #     no-op; kept for safety in case a raw base model is passed. ---
    if args.sid_index_path:
        print_rank_0(f"Loading index from {args.sid_index_path}")
        token_extender = TokenExtender(
            data_path=os.path.dirname(args.sid_index_path),
            dataset=os.path.basename(args.sid_index_path).split('.')[0],
        )
        new_tokens = token_extender.get_new_tokens()
        if new_tokens:
            existing_vocab = set(tokenizer.get_vocab().keys())
            tokens_to_add = [tok for tok in new_tokens if tok not in existing_vocab]
            if tokens_to_add:
                print_rank_0(f"Adding {len(tokens_to_add)} new tokens to tokenizer")
                tokenizer.add_tokens(tokens_to_add)
                model.resize_token_embeddings(len(tokenizer))
            else:
                print_rank_0("All SID tokens already present (expected for a Stage-1 checkpoint).")

    # Full-parameter fine-tuning: attention blocks, FFNs, and embeddings are all trainable.
    print_rank_0("Full fine-tuning enabled: attention blocks, FFNs, and embeddings remain trainable.")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percent = (trainable_params / total_params) * 100 if total_params > 0 else 0.0
    print_rank_0(f"Trainable parameters: {trainable_params} / {total_params} ({percent:.4f}%)")

    # --- data pipeline (plain torch Dataset / DataLoader) ---
    train_data, val_data, val_t2s, val_s2t = build_datasets(args, tokenizer, category)
    print_rank_0("LOAD DATA FINISHED")

    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

    if world_size > 1:
        train_sampler = DistributedSampler(train_data, shuffle=True, seed=args.seed)
    else:
        train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, collate_fn=collator,
        batch_size=args.micro_batch_size, pin_memory=True)

    def make_eval_loader(ds):
        sampler = DistributedSampler(ds, shuffle=False) if world_size > 1 else SequentialSampler(ds)
        return DataLoader(ds, sampler=sampler, collate_fn=collator,
                          batch_size=args.micro_batch_size, pin_memory=True)

    val_dataloader = make_eval_loader(val_data)
    t2s_dataloader = make_eval_loader(val_t2s)
    s2t_dataloader = make_eval_loader(val_s2t)

    # --- optimizer + lr scheduler (built by hand) ---
    # AdamW with betas=(0.9, 0.999); linear schedule + short warmup (mirrors the
    # original HF Trainer defaults used in Stage-2).
    optim_groups = get_optimizer_grouped_parameters(model, args.weight_decay)
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optim_groups, lr=args.learning_rate, betas=(0.9, 0.999))

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # --- wrap everything in the DeepSpeed engine ---
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
    )

    print_rank_0("***** Running Stage-2 reasoning activation *****")
    print_rank_0(f"  Num train examples          = {len(train_data)}")
    print_rank_0(f"  Num micro-batches per epoch = {len(train_dataloader)}")
    print_rank_0(f"  Optimizer steps per epoch   = {num_update_steps_per_epoch}")
    print_rank_0(f"  Total optimizer steps       = {max_train_steps}")

    # --- pre-training evaluation ---
    ppl, eval_loss = evaluation(model, val_dataloader, device)
    print_rank_0(f"[Eval @ epoch 0] sid_pred loss = {eval_loss:.4f} | ppl = {ppl:.4f}")

    best_eval_loss = float("inf")
    best_epoch = 0
    patience = 0
    global_step = 0
    final_dir = os.path.join(args.output_dir, "final_checkpoint")

    # =====================  the training loop  ===================== #
    for epoch in range(args.num_epochs):
        model.train()
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        print_rank_0(f"===== Epoch {epoch + 1}/{args.num_epochs} =====")
        for step, batch in enumerate(train_dataloader):
            start = time.time()
            batch = to_device(batch, device)

            # ---- forward / backward / step (fully in the open) ----
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            model.backward(loss)
            model.step()

            if model.is_gradient_accumulation_boundary():
                global_step += 1

            if is_rank_0() and (step % args.logging_steps == 0):
                cur_loss = loss.item()
                try:
                    cur_ppl = math.exp(min(20.0, cur_loss))
                except OverflowError:
                    cur_ppl = float("inf")
                try:
                    cur_lr = model.get_lr()[0]
                except Exception:
                    cur_lr = optimizer.param_groups[0]["lr"]
                dt_ms = (time.time() - start) * 1000
                print_rank_0(
                    f"epoch {epoch} | micro-step {step}/{len(train_dataloader)} | "
                    f"opt-step {global_step} | loss {cur_loss:.4f} | ppl {cur_ppl:.3f} | "
                    f"lr {cur_lr:.2e} | {dt_ms:.0f} ms"
                )
                if use_wandb:
                    wandb.log({
                        "train/loss": cur_loss,
                        "train/ppl": cur_ppl,
                        "train/lr": cur_lr,
                        "epoch": epoch,
                        "global_step": global_step,
                    })

        # ---- end-of-epoch evaluation on all three sets ----
        ppl, eval_loss = evaluation(model, val_dataloader, device)
        _, t2s_loss = evaluation(model, t2s_dataloader, device)
        _, s2t_loss = evaluation(model, s2t_dataloader, device)
        print_rank_0(
            f"[Eval @ epoch {epoch + 1}] "
            f"sid_pred loss={eval_loss:.4f} ppl={ppl:.3f} | "
            f"title2sid loss={t2s_loss:.4f} | sid2title loss={s2t_loss:.4f}"
        )
        if use_wandb:
            wandb.log({
                "eval/loss": eval_loss,
                "eval/ppl": ppl,
                "eval_title2sid/loss": t2s_loss,
                "eval_sid2title/loss": s2t_loss,
                "epoch": epoch + 1,
                "global_step": global_step,
            })

        # ---- save EVERY epoch's checkpoint ----
        epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
        print_rank_0(f"Saving epoch {epoch + 1} checkpoint -> {epoch_dir}")
        save_hf_checkpoint(model, tokenizer, epoch_dir)

        # ---- also keep the loss-best as a convenience pointer + optional early stopping ----
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_epoch = epoch + 1
            patience = 0
            print_rank_0(f"New best eval loss {best_eval_loss:.4f} (epoch {best_epoch}) -> {final_dir}")
            save_hf_checkpoint(model, tokenizer, final_dir)
        else:
            patience += 1
            if args.early_stopping_patience > 0:
                print_rank_0(f"No eval-loss improvement ({patience}/{args.early_stopping_patience}).")
                if patience >= args.early_stopping_patience:
                    print_rank_0("Early stopping triggered.")
                    break

    # Guarantee a final checkpoint even if the loss never improved after epoch 1.
    if not os.path.isdir(final_dir):
        save_hf_checkpoint(model, tokenizer, final_dir)

    print_rank_0(
        f"Training finished. Best eval loss = {best_eval_loss:.4f} (epoch {best_epoch}). "
        f"Per-epoch checkpoints in {args.output_dir}/epoch_*; loss-best copy at {final_dir}. "
        f"Select the final checkpoint by recsys metrics (NDCG@10 / HR@10) over the epoch_* dirs."
    )
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
