#!/usr/bin/env python3
# Pre-tokenize an Alpaca-style SFT dataset once and save to disk (HF Datasets Arrow)
# Changes: no pre-padding, truncation_side='left', prompt-only masking.

import os, argparse
from typing import List, Dict
from datasets import load_dataset, load_from_disk, Dataset
from transformers import AutoTokenizer

DEFAULT_MODEL   = "meta-llama/Llama-3.1-8B"
DEFAULT_RAW_DATA = "/arf/scratch/teknogrp6/llama31/data/alpaca.jsonl"
DEFAULT_OUT_DIR  = "/arf/scratch/teknogrp6/llama31/data/tokenized"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def alpaca_map_fn(tok, max_len: int):
    head_i = "### Instruction:\n"
    head_in = "\n\n### Input:\n"
    head_r = "\n\n### Response:\n"
    def _fn(batch):
        n = len(next(iter(batch.values()))) if len(batch) else 0
        instrs  = batch.get("instruction") or [""] * n
        inputs  = batch.get("input")       or [""] * n
        outputs = batch.get("output")      or [""] * n

        prompts, texts = [], []
        eos = tok.eos_token or ""
        for ins, inp, out in zip(instrs, inputs, outputs):
            p = f"{head_i}{ins}{(head_in + inp) if inp else ''}{head_r}"
            prompts.append(p)
            texts.append(p + (out or "") + eos)

        enc   = tok(texts,   max_length=max_len, truncation=True, padding=False)
        enc_p = tok(prompts, max_length=max_len, truncation=True, padding=False)

        input_ids_list, attn_list, labels_list = [], [], []
        for ids_full, ids_p in zip(enc["input_ids"], enc_p["input_ids"]):
            plen = min(len(ids_full), len(ids_p))
            labels = ids_full.copy()
            for i in range(plen):
                labels[i] = -100
            input_ids_list.append(ids_full)
            attn_list.append([1] * len(ids_full))
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attn_list,
            "labels": labels_list,
        }
    return _fn

def main():
    ap = argparse.ArgumentParser("Pretokenize Alpaca-style dataset (variable length, left truncation)")
    ap.add_argument("--model_name", default=DEFAULT_MODEL)
    ap.add_argument("--dataset_path", default=DEFAULT_RAW_DATA,
                    help="Dir (load_from_disk) | file (.json/.jsonl) | HF dataset id")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                    help="Directory to save tokenized dataset")
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--num_proc", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    # Keep pad token = eos for runtime padding, but we don't pre-pad here.
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    # Preserve answers when sequences are long
    tok.truncation_side = "left"

    dp = args.dataset_path
    if os.path.isdir(dp):
        raw = load_from_disk(dp)
    elif os.path.isfile(dp):
        ext = os.path.splitext(dp)[1].lower()
        if ext in [".jsonl", ".json"]:
            raw = load_dataset("json", data_files=dp, split="train")
        else:
            raise FileNotFoundError(f"Unsupported file extension: {dp}")
    else:
        raw = load_dataset(dp, split="train")

    mapped: Dataset = raw.map(
        alpaca_map_fn(tok, args.max_seq_length),
        batched=True,
        batch_size=2048,
        remove_columns=raw.column_names,
        num_proc=args.num_proc,
        desc="Tokenizing (variable-length, no pre-padding)",
    )

    cols = ["input_ids", "attention_mask", "labels"]
    drop_cols = [c for c in mapped.column_names if c not in cols]
    if drop_cols:
        mapped = mapped.remove_columns(drop_cols)

    os.makedirs(args.out_dir, exist_ok=True)
    mapped.save_to_disk(args.out_dir)
    print(f"[OK] Saved tokenized dataset to: {args.out_dir}")

if __name__ == "__main__":
    main()
