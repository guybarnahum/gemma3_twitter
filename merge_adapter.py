#!/usr/bin/env python3
import argparse, os, sys, json, shutil, torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

# Keep runs stable across stacks
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
try:
    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = True
except Exception:
    pass

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-3-4b-it")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)

    # Parity with your infer scripts
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--attn", choices=["sdpa", "eager", "flash2"], default="sdpa")
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    ap.add_argument("--max_shard_size", default="2GB")
    return ap.parse_args()

def _hf_kwargs(token: str):
    try:
        return {"token": token} if token else {}
    except TypeError:
        return {"use_auth_token": token} if token else {}

def _select_device(opt: str) -> str:
    if opt == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if opt == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def _select_dtype(opt: str, device: str):
    if opt == "fp32": return torch.float32
    if opt == "fp16": return torch.float16
    if opt == "bf16": return torch.bfloat16
    # auto
    if device == "cuda":
        return torch.bfloat16
    return torch.float32

def _model_ctx_len(model):
    for k in ("max_position_embeddings", "max_seq_len", "max_sequence_length"):
        v = getattr(model.config, k, None)
        if isinstance(v, int) and v > 0:
            return v
    return 8192

def _check_base_match(adapter_dir: str, base_name: str):
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    try:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        want = cfg.get("base_model_name_or_path")
        if want and want != base_name:
            print(f"[merge] WARNING: adapter trained on {want!r}, but --base is {base_name!r}.",
                  file=sys.stderr)
    except Exception:
        pass

def _load_base(base_id: str, dtype, attn_impl: str, device: str, hf_token: str):
    def _actually_load(attn: str):
        m = Gemma3ForConditionalGeneration.from_pretrained(
            base_id,
            torch_dtype=dtype,
            attn_implementation=attn,
            low_cpu_mem_usage=True,
            **_hf_kwargs(hf_token),
        ).to(device)
        m.config._attn_implementation = attn
        return m
    try:
        return _actually_load(attn_impl), attn_impl
    except Exception:
        if attn_impl != "eager":
            print(f"[merge] '{attn_impl}' failed; retrying with 'eager'…", file=sys.stderr)
            return _actually_load("eager"), "eager"
        raise

def main():
    args = parse()
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN") or ""
    device = _select_device(args.device)
    dtype = _select_dtype(args.dtype, device)

    print(f"[merge] base={args.base}", file=sys.stderr)
    print(f"[merge] adapter={args.adapter}", file=sys.stderr)
    print(f"[merge] out={args.out}", file=sys.stderr)
    print(f"[merge] device={device}, dtype={dtype}, attn={args.attn}", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    _check_base_match(args.adapter, args.base)

    # Prefer tokenizer from adapter (to keep added tokens/chat template), else from base
    tok_src = args.adapter if any(
        os.path.exists(os.path.join(args.adapter, f))
        for f in ("tokenizer.json", "tokenizer.model", "added_tokens.json")
    ) else args.base
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True, **_hf_kwargs(hf_token))

    base, attn_used = _load_base(args.base, dtype, args.attn, device, hf_token)
    print(f"[merge] attn(used)={attn_used}", file=sys.stderr)

    # Attach LoRA on the same device (no device_map='auto' here)
    from peft import PeftModel
    lora = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    lora = lora.to(device)

    # Merge and unload adapters → plain HF model
    print("[merge] merging adapters into base…", file=sys.stderr)
    merged = lora.merge_and_unload()  # returns a plain Gemma3ForConditionalGeneration
    merged = merged.to(device)

    # Save model (safetensors) + tokenizer
    print("[merge] saving model…", file=sys.stderr)
    merged.save_pretrained(
        args.out,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tok.save_pretrained(args.out)

    # If the adapter shipped a chat template, keep it
    ct = os.path.join(args.adapter, "chat_template.jinja")
    if os.path.exists(ct):
        try:
            shutil.copy(ct, os.path.join(args.out, "chat_template.jinja"))
            print("[merge] copied chat_template.jinja from adapter.", file=sys.stderr)
        except Exception as e:
            print(f"[merge] could not copy chat_template.jinja: {e}", file=sys.stderr)

    # (Optional) Save generation config if present
    try:
        merged.generation_config.save_pretrained(args.out)
    except Exception:
        pass

    # Heads-up about dtype
    ctx = _model_ctx_len(merged)
    print(f"Saved merged model to: {args.out} (attn={attn_used}, dtype={dtype}, ctx≈{ctx})")

if __name__ == "__main__":
    main()
