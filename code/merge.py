#!/usr/bin/env python3
"""
LoRA -> Base merge (H200 / CUDA-first, bf16)
- Merges entirely on GPU if available (bf16 on Hopper/H200).
- Saves in .safetensors (optionally sharded).
- Adds adapter sanity checks + CLI.
"""

import os, sys, json, argparse, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

HF_TOKEN = os.environ.get("HF_TOKEN")

def auth_kwargs():
    # Newer Transformers prefers 'token'; older uses 'use_auth_token'
    return {"token": HF_TOKEN} if HF_TOKEN else {}

def parse_dtype(s: str):
    s = s.lower()
    if s in ("bf16", "bfloat16"): return torch.bfloat16
    if s in ("fp16", "float16", "half"): return torch.float16
    if s in ("fp32", "float32", "float"): return torch.float32
    if s in ("auto",): return "auto"
    raise ValueError(f"Unknown dtype: {s}")

def load_base(model_id: str, device: str, dtype):
    kw = dict(
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        device_map={"": device},
        torch_dtype=dtype,
    )
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kw, **auth_kwargs())
    except TypeError:
        # old API fallback
        kw.pop("torch_dtype", None)  # older versions dislike non-tensor dtype sometimes
        return AutoModelForCausalLM.from_pretrained(model_id, **kw, use_auth_token=(HF_TOKEN or True))

def load_tok(model_id: str):
    kw = dict(use_fast=True, trust_remote_code=False)
    try:
        tok = AutoTokenizer.from_pretrained(model_id, **kw, **auth_kwargs())
    except TypeError:
        tok = AutoTokenizer.from_pretrained(model_id, **kw, use_auth_token=(HF_TOKEN or True))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def check_lora_dir(lora_dir: Path, expected_base: str | None):
    cfg = lora_dir / "adapter_config.json"
    weights_any = list(lora_dir.glob("adapter_model*.safetensors"))
    if not cfg.exists():
        raise FileNotFoundError(f"Missing {cfg}")
    if not weights_any:
        raise FileNotFoundError(f"No adapter_model*.safetensors in {lora_dir}")
    try:
        data = json.loads(cfg.read_text())
        # Many PEFT configs include this, but not all—only warn if mismatch detectable.
        base_in_cfg = data.get("base_model_name_or_path")
        if expected_base and base_in_cfg and (expected_base != base_in_cfg):
            print(f"[WARN] Adapter built on '{base_in_cfg}', current base is '{expected_base}'.")
    except Exception as e:
        print(f"[WARN] Could not parse adapter_config.json: {e}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--lora_dir", default="/arf/scratch/teknogrp6/checkpoints/best")
    p.add_argument("--out_dir", default="/arf/scratch/teknogrp6/merged_model")
    p.add_argument("--dtype", default="bf16", help="bf16|fp16|fp32|auto")
    p.add_argument("--device", default="cuda:0", help="cuda:0|cpu")
    p.add_argument("--max_shard_size", default="40GB")
    args = p.parse_args()

    # Device sanity for H200
    if args.device.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA requested but not available."
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f">> CUDA device 0: {torch.cuda.get_device_name(0)}")

    dtype = parse_dtype(args.dtype)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Early checks on LoRA folder
    check_lora_dir(Path(args.lora_dir), expected_base=args.base)

    print(f">> Loading BASE on {args.device} (dtype={args.dtype}): {args.base}")
    base = load_base(args.base, args.device, dtype)

    print(">> Loading tokenizer")
    tok = load_tok(args.base)

    print(f">> Loading LoRA from: {args.lora_dir}")
    lora = PeftModel.from_pretrained(base, args.lora_dir)  # attaches in-place on same device

    print(">> Merging (merge_and_unload) on-device...")
    with torch.inference_mode():
        merged = lora.merge_and_unload()

    # Preserve dtype in config for downstream loaders
    if getattr(merged.config, "torch_dtype", None) is None and dtype != "auto":
        merged.config.torch_dtype = dtype

    # (Optional) you can move to CPU before saving if you want to free GPU VRAM first:
    # merged = merged.to("cpu"); torch.cuda.empty_cache()

    print(f">> Saving merged model to: {out_dir} (safetensors, shard={args.max_shard_size})")
    merged.save_pretrained(out_dir, safe_serialization=True, max_shard_size=args.max_shard_size)
    tok.save_pretrained(out_dir)

    print("\n[DONE] Merge complete.")
    print(f"- Output: {out_dir.resolve()}")
    print("- Files: model*.safetensors (+ index if sharded), config.json, tokenizer files")

if __name__ == "__main__":
    main()