#!/usr/bin/env python3
# H100 LoRA trainer — optimized throughput + clean quality metrics
# Now with cosine LR decay options (linear / cosine / cosine_restart)
# + HF load_from_disk dataset support
# + Early stopping & best-checkpoint save/load
# + Debug/visibility: unbuffered stdout, micro heartbeats, loader safe mode
# + Dataset alignment: pad with model pad_id, bool attention_mask
# + Rich trainer_state.json (HF-style log_history, best checkpoint, counters)

import os
import sys
import math
import time
import json
import glob
import argparse
import re
import shutil
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

# ---- make stdout line-buffered so logs show up immediately (e.g., SLURM) ----
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import torch
from torch.utils.data import Dataset, DataLoader, Subset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    # --- FIX: Updated to use the new sdpa_kernel API ---
    from torch.nn.attention import sdpa_kernel
    # To explicitly prefer FlashAttention, use backend='flash'.
    # This replaces the old enable_flash=True.
    # PyTorch will automatically fall back to mem_efficient or math if flash is not available.
    with sdpa_kernel(backend='flash'):
        pass # The context manager sets the preference for the block.
except Exception as e:
    # Added a print statement in case enabling sdpa_kernel fails
    print(f"[WARN] Failed to enable sdpa_kernel (FlashAttention/Mem-efficient attention): {e}", flush=True)
    pass

# --- AMP compatibility shim ---
import contextlib

def make_autocast(bf16: bool, fp16: bool):
    if not (bf16 or fp16):
        return contextlib.nullcontext()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    try:
        return torch.amp.autocast("cuda", dtype=amp_dtype)
    except Exception:
        return torch.cuda.amp.autocast(dtype=amp_dtype)

def make_scaler(fp16: bool):
    # BF16 does not typically use a scaler, FP16 requires one
    try:
        return torch.amp.GradScaler("cuda", enabled=fp16)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=fp16)

# ---- Env defaults tuned for H100s (safe to keep) ----
os.environ.setdefault("DISABLE_CACHING_ALLOCATOR_WARMUP", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---- HF bits ----
from transformers import (
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    AutoTokenizer
)
from transformers.trainer_pt_utils import get_parameter_names

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

# HF datasets (optional, only needed if you pass a save_to_disk dir)
_DATASETS_AVAILABLE = False
try:
    from datasets import load_from_disk
    _DATASETS_AVAILABLE = True
except Exception:
    _DATASETS_AVAILABLE = False

# --- Qwen3 Specific Defaults ---
DEFAULT_MODEL = "Qwen/Qwen3-8B-Base"
DEFAULT_TOKENIZED_DIR = "/arf/scratch/teknogrp6/qwen3/data/tokenized"
DEFAULT_OUT_DIR = "/arf/scratch/teknogrp6/qwen3/checkpoints"

# -----------------------------
# Data
# -----------------------------

class PretokenizedDataset(Dataset):
    """
    Accepts any of:
      - Hugging Face `datasets` saved via save_to_disk (Arrow dir with dataset_info.json)
      - One/more .pt files: each contains {"samples": list[dict]} or directly list[dict]
      - One/more .jsonl files with {"input_ids":[...], "labels":[...], "attention_mask":[...]}
    """
    def __init__(self, data_dir: str):
        self._use_hf = False
        self._hf_ds = None
        self.samples: List[Dict[str, Any]] = []
        pt_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        jsonl_files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
        info_json = os.path.join(data_dir, "dataset_info.json")

        if os.path.exists(info_json):
            if not _DATASETS_AVAILABLE:
                raise ImportError("Install `datasets` or provide .pt/.jsonl shards.")
            self._hf_ds = load_from_disk(data_dir)
            self._use_hf = True
            print(f"[DATA] Loaded {len(self._hf_ds)} samples from HF 'save_to_disk' at {data_dir}.", flush=True)
        elif pt_files:
            for p in pt_files:
                chunk = torch.load(p, map_location="cpu")
                if isinstance(chunk, dict) and "samples" in chunk:
                    chunk = chunk["samples"]
                assert isinstance(chunk, list), f"PT file must contain list[dict], got {type(chunk)} from {p}"
                self.samples.extend(chunk)
            print(f"[DATA] Loaded {len(self.samples)} samples from {len(pt_files)} .pt shard(s).", flush=True)
        elif jsonl_files:
            import json as _json
            for j in jsonl_files:
                with open(j, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = _json.loads(line)
                        self.samples.append(obj)
            print(f"[DATA] Loaded {len(self.samples)} samples from {len(jsonl_files)} .jsonl file(s).", flush=True)
        else:
            raise FileNotFoundError(
                f"No HF dataset (dataset_info.json) or .pt/.jsonl shards found in {data_dir}."
            )

    def __len__(self):
        return len(self._hf_ds) if self._use_hf else len(self.samples)

    def __getitem__(self, idx):
        ex = self._hf_ds[idx] if self._use_hf else self.samples[idx]
        return {
            "input_ids": ex["input_ids"],
            "labels": ex.get("labels", ex["input_ids"]),
            "attention_mask": ex.get("attention_mask", [1]*len(ex["input_ids"])),
        }

def make_collate(max_seq_len: Optional[int] = None, pad_id: int = 0):
    """
    Pads input_ids with pad_id, attention_mask to False beyond sequence,
    labels to -100 for ignored tokens. Returns attention_mask as torch.bool.
    """
    def collate_batch(batch: List[Dict[str, Any]]):
        cap = max_seq_len if max_seq_len and max_seq_len > 0 else 10**9
        max_len = min(max(len(x["input_ids"]) for x in batch), cap)

        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            ids = x["input_ids"][:max_len]
            am  = x["attention_mask"][:max_len]
            lab = x["labels"][:max_len]
            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids += [pad_id]*pad_len
                am  += [0]*pad_len
                lab += [-100]*pad_len  # ignore padded tokens in loss
            input_ids.append(torch.tensor(ids, dtype=torch.long))
            attention_mask.append(torch.tensor(am, dtype=torch.bool))  # bool mask: memory-friendly
            labels.append(torch.tensor(lab, dtype=torch.long))

        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "labels": torch.stack(labels, dim=0),
        }
    return collate_batch

# -----------------------------
# Utils
# -----------------------------

def find_latest_checkpoint(out_dir: str) -> Optional[str]:
    cpts = glob.glob(os.path.join(out_dir, "checkpoint-*"))
    if not cpts:
        return None
    def step_of(path: str) -> int:
        import re
        m = re.search(r"checkpoint-(\d+)", path)
        return int(m.group(1)) if m else -1
    cpts.sort(key=step_of)
    return cpts[-1]

@dataclass
class EMA:
    beta: float = 0.98
    value: Optional[float] = None
    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.beta * self.value + (1 - self.beta) * x
        return self.value

def grad_global_norm(model: torch.nn.Module) -> Optional[float]:
    norms = []
    for p in model.parameters():
        if p.grad is not None:
            norms.append(p.grad.detach().data.norm(2))
    if not norms:
        return None
    return torch.norm(torch.stack(norms), 2).item()

# ---- checkpoint pruning helpers ----
def _list_step_ckpts(out_dir: str):
    cpts = [d for d in glob.glob(os.path.join(out_dir, "checkpoint-*")) if os.path.isdir(d)]
    def step(path: str) -> int:
        m = re.search(r"checkpoint-(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else -1
    return sorted(cpts, key=step)

def prune_checkpoints(out_dir: str, keep: int):
    # Only touches checkpoint-* dirs; does NOT touch 'best'
    if keep is None or keep <= 0:
        return
    cpts = _list_step_ckpts(out_dir)
    if len(cpts) <= keep:
        return
    to_remove = cpts[:-keep]  # keep the most recent `keep`
    for d in to_remove:
        try:
            shutil.rmtree(d)
            print(f"[PRUNE] Deleted old checkpoint {d}", flush=True)
        except Exception as e:
            print(f"[PRUNE] Failed to delete {d}: {e}", flush=True)

# -----------------------------
# Train
# -----------------------------

def main():
    p = argparse.ArgumentParser(
        description="Qwen3 H100 LoRA trainer (with cosine LR options + early stop + best-save)" # Updated description
    )
    # Files / model
    p.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL)
    p.add_argument("--tokenized_dir", type=str, default=DEFAULT_TOKENIZED_DIR)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

    # Core training
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=0, help="If > 0, overrides num_train_epochs.")
    p.add_argument("--batch_size", type=int, default=1, help="Global batch size (samples/step after accumulation).")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--micro_batch_size", type=int, default=1, help="Per-step batch size before accumulation.")
    p.add_argument("--auto_micro_batch", action="store_true", help="Try to auto-fit micro-batch to GPU memory.")
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--log_steps", type=int, default=10)

    # Optimizer
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.95)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=0.0, help=">0 to enable grad clipping.")

    # LR schedule
    p.add_argument("--lr_schedule", type=str, default="cosine",
                   choices=["linear", "cosine", "cosine_restart"],
                   help="Learning rate schedule type.")
    p.add_argument("--warmup_ratio", type=float, default=0.03,
                   help="Warmup fraction of total training updates.")
    p.add_argument("--num_cycles", type=float, default=0.5,
                   help="For cosine: number/fraction of cycles. For restarts: number of hard restarts.")

    # LoRA
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj")

    # FP/AMP
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 (recommended on H100/H200).")
    p.add_argument("--fp16", action="store_true", help="Use float16 AMP.")

    # Eval / early stopping / best
    p.add_argument("--eval_ratio", type=float, default=0.0, help="Holdout ratio from tokenized dataset (0=disable).")
    p.add_argument("--eval_every", type=int, default=1000, help="Run evaluation every N global steps.")
    p.add_argument("--early_stop_patience", type=int, default=0, help="Stop if no eval improvement for N evals (0=off).")
    p.add_argument("--min_delta", type=float, default=0.0, help="Required improvement in eval loss to reset patience.")
    p.add_argument("--save_best", action="store_true", help="Save best eval-loss checkpoint to out_dir/best.")
    p.add_argument("--load_best_at_end", action="store_true", help="Load best checkpoint into memory after training.")
    p.add_argument("--resume_from_best", action="store_true", help="If best exists, start from it (ignores latest).")

    # Misc / Debug
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ema_beta", type=float, default=0.98, help="EMA smoothing for train_loss_ema.")
    p.add_argument("--resume", action="store_true", help="Auto-resume from latest checkpoint in out_dir if present.")
    p.add_argument("--max_seq_len", type=int, default=4096, help="Clip sequences to this length in collate.")
    p.add_argument("--loader_safe", action="store_true",
                   help="Use loader settings that avoid worker/pin_memory stalls (num_workers=0, no pin_memory).")
    p.add_argument("--micro_heartbeat", type=int, default=0,
                   help="Print a [MICRO] heartbeat every N micro-steps (0=disable).")

    # NEW: limit how many step-checkpoints to keep
    p.add_argument(
        "--max_keep_ckpts", type=int, default=20,
        help="Keep at most N recent step checkpoints (excludes 'best'). 0=keep all."
    )

    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    dataset = PretokenizedDataset(args.tokenized_dir)

    # Split train/eval if requested (index-based)
    eval_loader = None
    eval_ds = None
    if args.eval_ratio and args.eval_ratio > 0.0:
        n_total = len(dataset)
        n_eval = max(1, int(n_total * args.eval_ratio))
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(n_total, generator=g).tolist()
        eval_idx = perm[:n_eval]
        train_idx = perm[n_eval:]
        train_ds = Subset(dataset, train_idx)
        eval_ds = Subset(dataset, eval_idx)
    else:
        train_ds = dataset

    # ------------------------------------------------------------------
    # Model (base) + PEFT attach / resume
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    print(f"[INFO] Loading {args.model_name_or_path} on {device} (dtype={dtype})", flush=True)

    # Decide resume directory
    resume_ckpt = None
    if args.resume_from_best:
        candidate = os.path.join(args.out_dir, "best")
        if os.path.isdir(candidate):
            resume_ckpt = candidate
    elif args.resume:
        resume_ckpt = find_latest_checkpoint(args.out_dir)

    # Determine if checkpoint contains PEFT adapters
    resuming_lora = bool(resume_ckpt and os.path.exists(os.path.join(resume_ckpt, "adapter_config.json")))

    # Always load base model from the original model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=dtype, low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True # Qwen models often need this
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Attach or load LoRA
    if args.use_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("peft is not installed. pip install peft")
        if resuming_lora:
            model = PeftModel.from_pretrained(model, resume_ckpt, is_trainable=True)
            print(f"[PEFT] Loaded adapters from {resume_ckpt}", flush=True)
        else:
            target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
            lora_cfg = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            model = get_peft_model(model, lora_cfg)
            print(f"[PEFT] Enabled LoRA on modules: {target_modules} (r={args.lora_r}, alpha={args.lora_alpha})", flush=True)

    model.train()

    # --- Clamp max_seq_len to model context ---
    if getattr(model.config, "max_position_embeddings", None) is not None and \
       args.max_seq_len > int(model.config.max_position_embeddings):
        args.max_seq_len = int(model.config.max_position_embeddings)
        print(f"[INFO] Clamped --max_seq_len to model context: {args.max_seq_len}", flush=True)

    # --- Determine pad_id from model (fallback to EOS) & build collate ---
    pad_id = getattr(model.config, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(model.config, "eos_token_id", 0)
        if pad_id == 0:
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                if pad_id is None: pad_id = 0
                print(f"[INFO] Determined pad_id from tokenizer: {pad_id}", flush=True)
            except Exception as e:
                print(f"[WARN] Could not load tokenizer to determine pad_id: {e}. Using fallback {pad_id}", flush=True)

    print(f"[INFO] Using pad_id: {pad_id} for data collation.", flush=True)
    collate = make_collate(args.max_seq_len, pad_id)


    # --- Auto-fit micro-batch on GPU (worst-case sequence) ---
    def probe_micro_batch(model, tried, device, max_seq_len, bf16, fp16, vocab_id=1):
        for cand in tried:
            try:
                ids = torch.full((cand, max_seq_len), pad_id, dtype=torch.long, device=device)
                am = torch.zeros_like(ids)
                labels = torch.full_like(ids, -100)
                ids[:, :16] = vocab_id
                am[:, :16] = 1

                model.zero_grad(set_to_none=True)
                with make_autocast(bf16, fp16):
                    out = model(input_ids=ids, attention_mask=am, labels=labels)
                    loss = out.loss
                loss.backward()
                model.zero_grad(set_to_none=True)
                del ids, am, labels, out, loss
                torch.cuda.synchronize(); torch.cuda.empty_cache()
                print(f"[AUTO] Fit micro-batch={cand} (worst-case length {max_seq_len})", flush=True)
                return cand
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[AUTO] OOM at micro-batch={cand}, trying smaller…", flush=True)
                    model.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                raise
        print("[AUTO] Could not fit any micro-batch > 0, defaulting to 1", flush=True)
        return 1

    tried = [32, 24, 20, 16, 12, 8, 4, 2, 1]
    if args.auto_micro_batch:
        micro_bs = probe_micro_batch(model, tried, device, args.max_seq_len, args.bf16, args.fp16, vocab_id=pad_id+1)
    else:
        micro_bs = args.micro_batch_size

    # --- Derive accumulation to satisfy global batch size ---
    if args.batch_size < micro_bs:
        print(f"[WARN] batch_size({args.batch_size}) < micro_batch_size({micro_bs}). Using micro_batch_size as effective.", flush=True)
        args.gradient_accumulation_steps = 1
        effective_bs = micro_bs
    else:
        args.gradient_accumulation_steps = max(1, args.batch_size // micro_bs)
        effective_bs = micro_bs * args.gradient_accumulation_steps
    print(f"[INFO] micro_batch_size={micro_bs}, grad_accum={args.gradient_accumulation_steps}, effective_bs={effective_bs}", flush=True)

    # --- Build the real loaders (now that micro_bs is final) ---
    if args.loader_safe:
        nw, pm, pw = 0, False, False
    else:
        nw, pm, pw = 4, True, True

    train_loader = DataLoader(
        train_ds, batch_size=micro_bs, shuffle=True,
        num_workers=nw, pin_memory=pm, persistent_workers=pw,
        collate_fn=collate
    )
    if eval_ds is not None:
        eval_loader = DataLoader(
            eval_ds, batch_size=max(1, micro_bs), shuffle=False,
            num_workers=nw, pin_memory=pm, persistent_workers=pw,
            collate_fn=collate
        )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ("bias" in n) or ("norm" in n.lower()):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Build AdamW with fused=True when available; fall back if unsupported
    adamw_kwargs = dict(
        params=optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )
    try:
        optimizer = torch.optim.AdamW(**adamw_kwargs, fused=True)
    except TypeError:
        optimizer = torch.optim.AdamW(**adamw_kwargs)

    # ------------------------------------------------------------------
    # Scheduler (after loaders exist)
    # ------------------------------------------------------------------
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_steps and args.max_steps > 0:
        total_updates = args.max_steps
    else:
        total_updates = updates_per_epoch * args.num_train_epochs

    warmup_steps = int(total_updates * args.warmup_ratio)
    if args.lr_schedule == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates
        )
    elif args.lr_schedule == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates,
            num_cycles=args.num_cycles
        )
    else:
        lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates,
            num_cycles=int(max(1, round(args.num_cycles)))
        )

    # ------------------------------------------------------------------
    # AMP
    # ------------------------------------------------------------------
    use_amp = args.fp16 or args.bf16
    scaler = make_scaler(args.fp16)

    # ------------------------------------------------------------------
    # Rich trainer state (init) + Resume (optimizer/scheduler/state only)
    # ------------------------------------------------------------------
    train_samples_per_epoch = len(train_ds)
    eval_samples = len(eval_ds) if eval_ds is not None else 0
    trainer_state_path = os.path.join(args.out_dir, "trainer_state.json")

    log_history: List[Dict[str, Any]] = []
    samples_seen = 0
    tokens_seen = 0
    best_eval = float("inf")
    best_global_step: Optional[int] = None
    best_model_checkpoint: Optional[str] = None

    global_step = 0
    start_epoch = 0

    if args.resume_from_best:
        candidate = os.path.join(args.out_dir, "best")
        if os.path.isdir(candidate):
            resume_ckpt = candidate
    elif args.resume:
        resume_ckpt = find_latest_checkpoint(args.out_dir)

    # Determine if checkpoint contains PEFT adapters
    resuming_lora = bool(resume_ckpt and os.path.exists(os.path.join(resume_ckpt, "adapter_config.json")))

    if resume_ckpt:
        print(f"[RESUME] Auto-resuming from {resume_ckpt}", flush=True)

        # Optimizer
        opt_p = os.path.join(resume_ckpt, "optimizer.pt")
        if os.path.exists(opt_p):
            try:
                optimizer.load_state_dict(torch.load(opt_p, map_location="cpu"))
            except Exception as e:
                print(f"[RESUME] Optimizer load skipped ({e})", flush=True)

        # Scheduler
        sch_p = os.path.join(resume_ckpt, "scheduler.pt")
        if os.path.exists(sch_p):
            try:
                lr_scheduler.load_state_dict(torch.load(sch_p, map_location="cpu"))
            except Exception as e:
                print(f"[RESUME] Scheduler load skipped ({e})", flush=True)

        # Trainer state (back-compat with HF-style files)
        st_p = os.path.join(resume_ckpt, "trainer_state.json")
        if os.path.exists(st_p):
            try:
                with open(st_p, "r") as f:
                    st = json.load(f)

                # Common fields
                global_step = int(st.get("global_step", 0))

                loaded_epoch = st.get("epoch", 0)
                epoch_index = st.get("epoch_index", None)
                if epoch_index is not None:
                    start_epoch = int(epoch_index)
                else:
                    start_epoch = int(loaded_epoch) if isinstance(loaded_epoch, (int, float)) else 0

                # Rich fields
                log_history = st.get("log_history", []) or []
                samples_seen = int(st.get("samples_seen", st.get("num_train_samples_seen", 0)))
                tokens_seen = int(st.get("num_input_tokens_seen", st.get("tokens_seen", 0)))

                if samples_seen == 0 and isinstance(loaded_epoch, float):
                    samples_seen = int(round(loaded_epoch * train_samples_per_epoch))

                best_eval = float(st.get("best_metric", st.get("best_eval", float("inf"))) or float("inf"))
                bgs = st.get("best_global_step", None)
                best_global_step = int(bgs) if bgs is not None else None
                best_model_checkpoint = st.get("best_model_checkpoint", None)

            except Exception as e:
                print(f"[RESUME] State load skipped ({e})", flush=True)

        # For non-PEFT checkpoints that saved full weights (no adapters)
        if not resuming_lora:
            sd_path = os.path.join(resume_ckpt, "pytorch_model.bin")
            if os.path.exists(sd_path):
                try:
                    model.load_state_dict(torch.load(sd_path, map_location="cpu"), strict=False)
                    print("[RESUME] Loaded model state_dict from checkpoint.", flush=True)
                except Exception as e:
                    print(f"[RESUME] Model state_dict load skipped ({e})", flush=True)

    # ------------------------------------------------------------------
    # Eval helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_eval() -> float:
        if eval_loader is None:
            return float("inf")
        model.eval()
        losses = 0.0
        count = 0
        for batch in eval_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with make_autocast(args.bf16, args.fp16):
                out = model(**batch)
                loss = out.loss
            B = batch["input_ids"].shape[0]
            losses += loss.item() * B
            count += B
        model.train()
        return losses / max(1, count)

    # ---- Rich state dump ----
    def dump_trainer_state(path: str, epoch_idx: int):
        epoch_cont = samples_seen / max(1, train_samples_per_epoch)
        try:
            cur_lr = lr_scheduler.get_last_lr()[0]
        except Exception:
            cur_lr = optimizer.param_groups[0]["lr"]

        state = {
            "best_global_step": best_global_step,
            "best_metric": None if not math.isfinite(best_eval) else float(best_eval),
            "best_model_checkpoint": best_model_checkpoint,
            "epoch": float(epoch_cont),
            "epoch_index": int(epoch_idx),
            "eval_steps": int(args.eval_every),
            "global_step": int(global_step),
            "is_hyper_param_search": False,
            "is_local_process_zero": True,
            "is_world_process_zero": True,
            "log_history": log_history,
            "logging_steps": int(args.log_steps),
            "max_steps": int(total_updates),
            "num_input_tokens_seen": int(tokens_seen),
            "num_train_epochs": int(args.num_train_epochs),
            "save_steps": int(args.save_steps),
            "train_batch_size": int(micro_bs),
            "grad_accum": int(args.gradient_accumulation_steps),
            "effective_batch_size": int(micro_bs * args.gradient_accumulation_steps),
            "updates_per_epoch": int(updates_per_epoch),
            "train_dataset_size": int(train_samples_per_epoch),
            "eval_dataset_size": int(eval_samples),
            "eval_ratio": float(args.eval_ratio),
            "lr": float(cur_lr),
            "lr_schedule": args.lr_schedule,
            "warmup_ratio": float(args.warmup_ratio),
            "num_cycles": float(args.num_cycles),
            "weight_decay": float(args.weight_decay),
            "adam_beta1": float(args.adam_beta1),
            "adam_beta2": float(args.adam_beta2),
            "adam_eps": float(args.adam_eps),
            "max_grad_norm": float(args.max_grad_norm),
            "bf16": bool(args.bf16),
            "fp16": bool(args.fp16),
            "use_lora": bool(args.use_lora),
            "lora_r": int(args.lora_r) if args.use_lora else None,
            "lora_alpha": int(args.lora_alpha) if args.use_lora else None,
            "lora_dropout": float(args.lora_dropout) if args.use_lora else None,
            "lora_target_modules": args.lora_target_modules if args.use_lora else None,
            "model_name_or_path": args.model_name_or_path,
            "resume": bool(args.resume),
            "resume_from_best": bool(args.resume_from_best),
            "samples_seen": int(samples_seen),
            "tokens_seen": int(tokens_seen),
        }
        with open(path, "w") as f:
            json.dump(state, f)

    # ---- Checkpoint helpers (write rich state in ckpt + root) ----
    def save_checkpoint(step: int, epoch_idx: int):
        ckpt_dir = os.path.join(args.out_dir, f"checkpoint-{step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        try:
            model.save_pretrained(ckpt_dir)
        except Exception:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
        torch.save(lr_scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
        dump_trainer_state(os.path.join(ckpt_dir, "trainer_state.json"), epoch_idx=epoch_idx)
        dump_trainer_state(trainer_state_path, epoch_idx=epoch_idx)
        print(f"[OK] Saved to {ckpt_dir}", flush=True)
        prune_checkpoints(args.out_dir, args.max_keep_ckpts)

    def save_best_checkpoint(tag: str = "best", step: int = 0, epoch_idx: int = 0):
        nonlocal best_model_checkpoint, best_global_step
        ckpt_dir = os.path.join(args.out_dir, tag)
        os.makedirs(ckpt_dir, exist_ok=True)
        try:
            model.save_pretrained(ckpt_dir)
        except Exception:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
        torch.save(lr_scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
        best_model_checkpoint = ckpt_dir
        best_global_step = step
        dump_trainer_state(os.path.join(ckpt_dir, "trainer_state.json"), epoch_idx=epoch_idx)
        dump_trainer_state(trainer_state_path, epoch_idx=epoch_idx)
        print(f"[OK] Saved BEST to {ckpt_dir}", flush=True)

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------
    ema = EMA(beta=args.ema_beta)
    t_start = time.time()
    last_log_t = time.time()
    tok_window = 0
    samp_window = 0

    bad_counts = 0
    optimizer.zero_grad(set_to_none=True)

    print("[INFO] Starting training loop…", flush=True)
    first_batch_t0 = time.time()

    for epoch in range(start_epoch, args.num_train_epochs):
        epoch_start_step = global_step
        for step, batch in enumerate(train_loader):
            if step == 0 and global_step == 0:
                print(f"[DEBUG] First batch fetched after {time.time() - first_batch_t0:.1f}s", flush=True)
            if args.micro_heartbeat and (step % args.micro_heartbeat == 0):
                print(f"[MICRO] step={step} acc={(step % args.gradient_accumulation_steps)+1}/{args.gradient_accumulation_steps}", flush=True)

            # Move to device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with make_autocast(args.bf16, args.fp16):
                out = model(**batch)
                loss = out.loss / args.gradient_accumulation_steps

            # Backprop
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            cur_tokens = int(batch["attention_mask"].sum().item())
            cur_bs = batch["input_ids"].size(0)
            tokens_seen += cur_tokens
            tok_window += cur_tokens
            samp_window += cur_bs
            samples_seen += cur_bs
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm and args.max_grad_norm > 0.0:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if args.fp16:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # Log
                smoothed = ema.update(loss.item() * args.gradient_accumulation_steps)
                if global_step % args.log_steps == 0:
                    elapsed = max(1e-6, time.time() - last_log_t)
                    tok_per_s = tok_window / elapsed
                    samp_per_s = samp_window / elapsed
                    cur_lr = lr_scheduler.get_last_lr()[0]
                    epoch_cont = samples_seen / max(1, train_samples_per_epoch)

                    entry = {
                        "epoch": float(epoch_cont),
                        "learning_rate": float(cur_lr),
                        "loss": float(smoothed),
                        "step": int(global_step),
                    }
                    log_history.append(entry)

                    dump_trainer_state(trainer_state_path, epoch_idx=epoch)

                    print(f"[STEP {global_step:6d}] train_loss_ema={smoothed:.4f} | "
                          f"lr={cur_lr:.6e} | tok/s={tok_per_s:.0f} | samp/s={samp_per_s:.2f} | "
                          f"epoch={epoch_cont:.3f}", flush=True)
                    last_log_t = time.time()
                    tok_window = 0
                    samp_window = 0

                # Save
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(global_step, epoch_idx=epoch)

                # Eval + Early stop + Best
                if eval_loader is not None and (global_step % args.eval_every == 0):
                    eval_t0 = time.time()
                    cur_eval = run_eval()
                    eval_rt = time.time() - eval_t0
                    improved = (best_eval - cur_eval) > args.min_delta
                    print(f"[EVAL step {global_step}] eval_loss={cur_eval:.4f} (best={best_eval:.4f}) {'IMPROVED' if improved else ''}", flush=True)

                    log_history.append({
                        "epoch": float(samples_seen / max(1, train_samples_per_epoch)),
                        "eval_loss": float(cur_eval),
                        "eval_runtime": float(eval_rt),
                        "step": int(global_step),
                    })
                    dump_trainer_state(trainer_state_path, epoch_idx=epoch)

                    if improved:
                        best_eval = cur_eval
                        bad_counts = 0
                        if args.save_best:
                            save_best_checkpoint("best", step=global_step, epoch_idx=epoch)
                    else:
                        bad_counts += 1
                        if args.early_stop_patience > 0 and bad_counts >= args.early_stop_patience:
                            print(f"[EARLY-STOP] No improvement in {bad_counts} evals. Stopping at step {global_step}.", flush=True)
                            save_checkpoint(global_step, epoch_idx=epoch)
                            if args.save_best and args.load_best_at_end:
                                best_dir = os.path.join(args.out_dir, "best")
                                if os.path.isdir(best_dir):
                                    print("[LOAD] Loaded BEST into memory (due to early stop).", flush=True)
                                    if os.path.exists(os.path.join(best_dir, "adapter_model.safetensors")):
                                        model = PeftModel.from_pretrained(model, best_dir, is_trainable=True)
                                    elif os.path.exists(os.path.join(best_dir, "pytorch_model.bin")):
                                        model.load_state_dict(torch.load(os.path.join(best_dir, "pytorch_model.bin"), map_location="cpu"), strict=False)
                                else:
                                    print("[WARN] Best checkpoint directory not found for loading at end.", flush=True)
                            print("[DONE] Training complete (early stop).", flush=True)
                            return

                # Stop if max_steps
                if args.max_steps and global_step >= args.max_steps:
                    print(f"[DONE] Reached max_steps={args.max_steps}. Saving final checkpoint.", flush=True)
                    save_checkpoint(global_step, epoch_idx=epoch)
                    if args.save_best and args.load_best_at_end and os.path.isdir(os.path.join(args.out_dir, "best")):
                        print("[LOAD] Loaded BEST at end of training (max_steps reached).", flush=True)
                        best_dir = os.path.join(args.out_dir, "best")
                        if os.path.exists(os.path.join(best_dir, "adapter_model.safetensors")):
                            model = PeftModel.from_pretrained(model, best_dir, is_trainable=True)
                        elif os.path.exists(os.path.join(best_dir, "pytorch_model.bin")):
                            model.load_state_dict(torch.load(os.path.join(best_dir, "pytorch_model.bin"), map_location="cpu"), strict=False)
                    return

        # End of epoch save
        print(f"[INFO] End of epoch {epoch}. Saving checkpoint.", flush=True)
        save_checkpoint(global_step, epoch_idx=epoch)

    print("[DONE] Training complete.", flush=True)
    if args.save_best and args.load_best_at_end and os.path.isdir(os.path.join(args.out_dir, "best")):
        print("[LOAD] Loaded BEST at end of training (final).", flush=True)
        best_dir = os.path.join(args.out_dir, "best")
        if os.path.exists(os.path.join(best_dir, "adapter_model.safetensors")):
            model = PeftModel.from_pretrained(model, best_dir, is_trainable=True)
        elif os.path.exists(os.path.join(best_dir, "pytorch_model.bin")):
            model.load_state_dict(torch.load(os.path.join(best_dir, "pytorch_model.bin"), map_location="cpu"), strict=False)


if __name__ == "__main__":
    main()