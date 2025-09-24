import os, json, argparse, torch, glob, re
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel

# ---------- Environment ----------
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("NCCL_IB_HCA", "mlx5")
os.environ.setdefault("NCCL_NVLS_ENABLE", "1")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

# ---------- Small utils ----------
def detect_flash_attn():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return None

def set_perf(tf32=True):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def format_alpaca_prompt(instruction: str, input_text: Optional[str]) -> str:
    head_i = "### Instruction:\n"
    head_in = "\n\n### Input:\n"
    head_r = "\n\n### Response:\n"
    if not input_text or str(input_text).strip() == "":
        return f"{head_i}{instruction}{head_r}"
    return f"{head_i}{instruction}{head_in}{input_text}{head_r}"

def get_eos_ids(tok) -> List[int]:
    ids = []
    if tok.eos_token_id is not None:
        ids.append(tok.eos_token_id)
    for t in ("<|eot_id|>", "<|eot|>", "<|end_of_text|>", "</s>"):
        try:
            tid = tok.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid >= 0:
                ids.append(tid)
        except Exception:
            pass
    ids = sorted({i for i in ids if isinstance(i, int) and i >= 0})
    return ids or [tok.eos_token_id]

# ---------- LoRA checkpoint resolver ----------
_STEP_RE = re.compile(r"checkpoint[-_](\d+)$")
def _extract_step_from_path(p: str) -> Optional[int]:
    m = _STEP_RE.search(os.path.basename(os.path.normpath(p)))
    return int(m.group(1)) if m else None

def resolve_lora_checkpoint(lora_path: Optional[str]) -> Tuple[Optional[str], Optional[int], str]:
    """
    Returns (resolved_path, step, strategy_msg)
    Strategy:
      1) None -> (None, None, "no_lora")
      2) If lora_path has adapter files -> use it (maybe a single checkpoint dir)
      3) Else find subdirs 'checkpoint-*' that contain adapters
         Prefer highest numeric step; fallback to latest mtime.
    """
    if not lora_path:
        return None, None, "no_lora"

    acfg = os.path.join(lora_path, "adapter_config.json")
    amodel = os.path.join(lora_path, "adapter_model.safetensors")
    if os.path.isfile(acfg) or os.path.isfile(amodel):
        return lora_path, _extract_step_from_path(lora_path), "given_path_used"

    candidates = []
    for d in glob.glob(os.path.join(lora_path, "checkpoint-*")):
        if os.path.isfile(os.path.join(d, "adapter_config.json")) or \
           os.path.isfile(os.path.join(d, "adapter_model.safetensors")):
            candidates.append((d, _extract_step_from_path(d), os.path.getmtime(d)))

    if not candidates:
        return lora_path, None, "no_checkpoints_found_used_root"

    with_step = [c for c in candidates if c[1] is not None]
    if with_step:
        chosen = max(with_step, key=lambda x: x[1])  # highest step
        return chosen[0], chosen[1], "auto_pick_highest_step"
    chosen = max(candidates, key=lambda x: x[2])  # latest mtime
    return chosen[0], None, "auto_pick_latest_mtime"

# ---------- Tokenize with room so '### Response:' survives ----------
def tokenize_with_room(tokenizer, prompts: List[str], max_new_tokens: int):
    model_max = getattr(tokenizer, "model_max_length", 4096)
    if model_max is None or model_max > 10_000_000:  # HF sometimes sets a huge sentinel
        model_max = 4096
    room = max(8, max_new_tokens + 8)
    max_inp = max(16, model_max - room)
    return tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_inp)

def clean_response(txt: str) -> str:
    cut_markers = ["### Instruction", "### Input", "### Response", "<|eot_id|>", "<|end_of_text|>", "</s>"]
    cut_pos = [txt.find(m) for m in cut_markers if m in txt]
    cut_pos = [p for p in cut_pos if p > 0]
    if cut_pos:
        txt = txt[:min(cut_pos)]
    return txt.strip()

# ---------- Generation kwargs builder (gates sampling-only knobs) ----------
def build_gen_kwargs(*,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    num_beams: int,
    repetition_penalty: float,
    max_new_tokens: int,
    min_new_tokens: int,
    eos_token_id,
    pad_token_id
):
    # Safe temp: neutralize if not sampling or temp<=0
    eff_temp = temperature if (do_sample and (temperature is not None) and (temperature > 0)) else 1.0
    kw = dict(
        max_new_tokens=max_new_tokens,
        min_new_tokens=max(0, min(min_new_tokens, max_new_tokens)),
        repetition_penalty=repetition_penalty,
        num_beams=num_beams,
        do_sample=do_sample,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        temperature=eff_temp,
    )
    if do_sample:
        if top_p is not None and 0 < top_p <= 1:
            kw["top_p"] = top_p
        if top_k is not None and top_k > 0:
            kw["top_k"] = top_k
    return kw

# ---------- Model load ----------
def load_model_and_tokenizer(model_path: str,
                             lora_path: Optional[str],
                             device: str,
                             dtype,
                             trust_remote_code=True):
    attn_impl = detect_flash_attn()
    kwargs = dict(torch_dtype=dtype, trust_remote_code=trust_remote_code, low_cpu_mem_usage=True)
    if device == "cuda":
        kwargs["device_map"] = {"": 0}
        if attn_impl is not None:
            kwargs["attn_implementation"] = attn_impl

    print(f"[INFO] Loading base model: {model_path} (device={device}, dtype={dtype}, attn={attn_impl or 'sdpa'})")
    try:
        base = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        base = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=trust_remote_code)

    # Left padding for decoder-only + ensure pad token exists
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    try:
        base.config.pad_token_id = tok.pad_token_id
    except Exception:
        pass

    resolved_lora, lora_step, strategy = resolve_lora_checkpoint(lora_path)
    if resolved_lora:
        print(f"[INFO] LoRA root         : {lora_path}")
        print(f"[INFO] LoRA resolved     : {resolved_lora}")
        print(f"[INFO] LoRA pick strategy: {strategy}")
        if lora_step is not None:
            print(f"[INFO] LoRA step         : {lora_step}")
        print(f"[INFO] Loading LoRA adapter…")
        base = PeftModel.from_pretrained(base, resolved_lora, is_trainable=False)

    base.eval()
    base.config.use_cache = True
    return base, tok, resolved_lora, lora_step, strategy

# ---------- Generation ----------
def generate_batch(model,
                   tokenizer,
                   prompts: List[str],
                   max_new_tokens: int = 512,
                   min_new_tokens: int = 0,
                   temperature: float = 0.7,
                   top_p: float = 0.9,
                   top_k: int = 50,
                   repetition_penalty: float = 1.1,
                   do_sample: bool = True,
                   num_beams: int = 1,
                   eos_token_id=None,
                   device: str = "cuda"):
    enc = tokenize_with_room(tokenizer, prompts, max_new_tokens)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    eos_ids = eos_token_id or get_eos_ids(tokenizer)
    gen_kwargs = build_gen_kwargs(
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

    # With left padding, continuation starts at the common input width S
    decoded = []
    S = input_ids.size(1)
    for i in range(out.size(0)):
        cont = out[i, S:]
        txt = tokenizer.decode(cont, skip_special_tokens=True).strip()
        decoded.append(txt)

    # Retry empties greedily
    for i, txt in enumerate(decoded):
        if txt:
            continue
        single_ids = input_ids[i:i+1]
        single_attn = attention_mask[i:i+1]
        with torch.no_grad():
            out2 = model.generate(
                input_ids=single_ids, attention_mask=single_attn,
                **build_gen_kwargs(
                    do_sample=False,  # greedy retry
                    temperature=1.0, top_p=1.0, top_k=0,
                    num_beams=1, repetition_penalty=1.0,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=max(0, min_new_tokens // 2),
                    eos_token_id=eos_ids,
                    pad_token_id=tokenizer.pad_token_id,
                )
            )
        S2 = single_ids.size(1)
        decoded[i] = tokenizer.decode(out2[0, S2:], skip_special_tokens=True).strip()

    return [clean_response(t) for t in decoded]

def strip_to_response(text: str) -> str:
    tag = "### Response:\n"
    idx = text.rfind(tag)
    return text[idx + len(tag):].strip() if idx >= 0 else text.strip()

# ---------- I/O helpers ----------
def load_records(jsonl_path: str) -> List[Dict]:
    """
    Reads either Alpaca JSONL or prompt JSONL and returns
    a list of dicts with keys: instruction, input, prompt
    (BOM-safe and with line-numbered error messages)
    """
    out = []
    # utf-8-sig auto-strips BOM if present
    with open(jsonl_path, "r", encoding="utf-8-sig") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                candidate = line.lstrip("\ufeff").rstrip(",")
                try:
                    rec = json.loads(candidate)
                except Exception:
                    snippet = (line[:160] + "…") if len(line) > 160 else line
                    raise ValueError(
                        f"Invalid JSON on line {lineno}: {e.msg} (col {e.colno}). Offending text: {snippet}"
                    ) from e

            if "instruction" in rec:
                instr = rec.get("instruction", "")
                inp = rec.get("input", "")
                out.append({"instruction": instr, "input": inp,
                            "prompt": format_alpaca_prompt(instr, inp)})
            elif "prompt" in rec:
                p = str(rec["prompt"])
                if p.lstrip().startswith("### Instruction:"):
                    out.append({"instruction": "", "input": "", "prompt": p})
                else:
                    out.append({"instruction": p, "input": "", "prompt": format_alpaca_prompt(p, None)})
            # else: silently ignore unknown rows
    return out

def write_outputs(path: Optional[str], results: List[Dict], meta: Dict):
    if not path:
        return
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        with open(path, "w", encoding="utf-8") as wf:
            for rec in results:
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[OK] Wrote {len(results)} generations to {path} (JSONL)")
    elif ext == ".json":
        payload = {"meta": meta, "results": results}
        with open(path, "w", encoding="utf-8") as wf:
            json.dump(payload, wf, ensure_ascii=False, indent=2)
        print(f"[OK] Wrote {len(results)} generations to {path} (pretty JSON)")
    else:
        raise ValueError(f"--out must end with .json or .jsonl (got {path})")

# ---------- Main ----------
def main():
    from argparse import BooleanOptionalAction
    ap = argparse.ArgumentParser(
        "LLM inference (merged or base+LoRA)",
        allow_abbrev=False  # avoid '--out' ambiguity
    )
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lora_path", default=None)
    ap.add_argument("--prompt", default=None, help="Single prompt string. If omitted, use --input_jsonl.")
    ap.add_argument("--input_jsonl", default=None, help="Alpaca JSONL or plain prompt JSONL")
    # Unified output path; keep old --out_jsonl for compatibility.
    ap.add_argument("--out", default=None, help="Output file (.json or .jsonl)")
    ap.add_argument("--out_jsonl", default=None, help="[Deprecated] If set, write JSONL here.")
    ap.add_argument("--out_key", choices=["output","response"], default="output", help="Answer field name")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--min_new_tokens", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    # Accept both dashed and underscored forms
    ap.add_argument("--do_sample", "--do-sample", action=BooleanOptionalAction, default=True,
                    help="Enable/disable sampling (also supports --no-do_sample / --no-do-sample)")
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--trust_remote_code", "--trust-remote-code",
                    action=BooleanOptionalAction, default=True,
                    help="Allow custom model code (also supports --no-trust_remote_code / --no-trust-remote-code)")
    args = ap.parse_args()

    # unify deprecated flag
    if args.out_jsonl and args.out:
        raise ValueError("Provide either --out or --out_jsonl, not both.")
    if args.out_jsonl and not args.out:
        args.out = args.out_jsonl  # keep working for old calls

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    set_perf(tf32=True)

    cuda_ok = torch.cuda.is_available()
    if args.device == "cuda" and not cuda_ok:
        raise RuntimeError("CUDA requested but not available.")
    device = ("cuda" if cuda_ok else "cpu") if args.device == "auto" else args.device

    if args.dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    model, tok, resolved_lora, lora_step, lora_strategy = load_model_and_tokenizer(
        model_path=args.model_path,
        lora_path=args.lora_path,
        device=device,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )

    if args.prompt and args.input_jsonl:
        raise ValueError("Provide either --prompt OR --input_jsonl, not both.")

    eos_ids = get_eos_ids(tok)

    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_path": args.model_path,
        "lora_root": args.lora_path,
        "lora_resolved": resolved_lora,
        "lora_step": lora_step,
        "lora_pick_strategy": lora_strategy,
        "device": device,
        "dtype": str(dtype),
        "gen_params": {
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "do_sample": args.do_sample,
            "num_beams": args.num_beams,
        },
        "template": "alpaca_v1 (### Instruction / ### Input / ### Response)",
        "out_key": args.out_key,
    }

    # Single prompt
    if args.prompt:
        prompt = format_alpaca_prompt(args.prompt, None)
        streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        enc = tokenize_with_room(tok, [prompt], args.max_new_tokens)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        gen_kwargs = build_gen_kwargs(
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            eos_token_id=eos_ids,
            pad_token_id=tok.pad_token_id,
        )

        with torch.no_grad():
            print("=== Generation ===")
            _ = model.generate(input_ids=input_ids, attention_mask=attn, streamer=streamer, **gen_kwargs)
            print("\n==================")
        if args.out:
            with torch.no_grad():
                out_ids = model.generate(input_ids=input_ids, attention_mask=attn, **gen_kwargs)
            S = input_ids.size(1)
            txt = tok.decode(out_ids[0, S:], skip_special_tokens=True).strip()
            resp = clean_response(txt)
            record = {"instruction": args.prompt, "input": "", args.out_key: resp}
            write_outputs(args.out, [record], meta)
        return

    # Batch
    if args.input_jsonl:
        if not os.path.isfile(args.input_jsonl):
            raise FileNotFoundError(args.input_jsonl)

        records = load_records(args.input_jsonl)
        if not records:
            raise ValueError("No valid rows found in JSONL. Expect 'instruction'/'input' or 'prompt' keys.")

        results = []
        bs = max(1, args.batch_size)
        for i in range(0, len(records), bs):
            batch = records[i:i + bs]
            prompts = [r["prompt"] for r in batch]
            outs = generate_batch(
                model, tok, prompts,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                eos_token_id=eos_ids,
                device=device,
            )
            for r, full_text in zip(batch, outs):
                response = clean_response(full_text)
                rec_out = {
                    "instruction": r.get("instruction",""),
                    "input": r.get("input",""),
                    meta["out_key"]: response
                }
                results.append(rec_out)
                # Compact console log
                print(f"\n---\nInstruction: {r.get('instruction','')[:140]}")
                if r.get("input",""):
                    print(f"Input      : {r.get('input','')[:140]}")
                print(f"{meta['out_key'].capitalize()}:\n{response[:1000]}\n")

        write_outputs(args.out, results, meta)
        return

    raise ValueError("Provide either --prompt or --input_jsonl.")

if __name__ == "__main__":
    main()