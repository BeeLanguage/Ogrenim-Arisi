#!/usr/bin/env python3
# merge_lora_to_ckpt.py
import argparse, os, torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base model HF id or local path (must match LoRA)")
    ap.add_argument("--lora", required=True, help="LoRA adapter folder (contains adapter_model.safetensors)")
    ap.add_argument("--out",  default="merged_model.ckpt", help="Output .ckpt path")
    ap.add_argument("--dtype", choices=["fp16","bf16","fp32"], default="fp16")
    ap.add_argument("--cpu", action="store_true", help="Merge on CPU (slower, but minimal VRAM)")
    args = ap.parse_args()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    print("• Loading base model…")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu" if args.cpu else "auto",
        trust_remote_code=True,   # harmless if not needed; helps custom archs
    )
    print("• Applying LoRA…")
    merged = PeftModel.from_pretrained(base, args.lora)
    merged = merged.merge_and_unload()   # <- bakes LoRA into the base weights

    print("• Extracting state_dict…")
    state = merged.state_dict()

    print(f"• Saving Lightning-style checkpoint → {args.out}")
    torch.save({"state_dict": state}, args.out)

    # (nice to have) also save a standard HF folder next to it
    hf_out = os.path.splitext(args.out)[0] + "_hf"
    merged.save_pretrained(hf_out)
    print(f"✔ wrote {args.out} and HF folder {hf_out}/")

if __name__ == "__main__":
    main()