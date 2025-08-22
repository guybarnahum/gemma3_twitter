#!/usr/bin/env python3
import argparse, os, sys, threading, time, torch

# --- Make runs robust against TorchDynamo graph breaks ---
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # hard-disable torch.compile/dynamo by default
try:
    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = True  # if anything slips through, fall back to eager
except Exception:
    pass

from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    TextIteratorStreamer,
)

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-3-4b-it")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
    ap.add_argument("--load_in_4bit", action="store_true", help="(GPU only) use bitsandbytes 4-bit")
    ap.add_argument("--attn", choices=["sdpa","eager","flash2"], default="sdpa",
                    help="Attention implementation to use")
    return ap.parse_args()

def _hf_kwargs(token: str):
    try:
        return {"token": token} if token else {}
    except TypeError:
        return {"use_auth_token": token} if token else {}

def _mask(tok: str) -> str:
    return tok[:6] + "…" if tok and len(tok) > 8 else ("(none)" if not tok else tok)

def _select_device(devopt: str):
    if devopt == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if devopt == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def _load_base(base_name, dtype, attn_impl, hf_token, device, load_in_4bit):
    # Try requested attention impl; on failure, transparently retry with eager.
    def _actually_load(attn):
        if load_in_4bit and device == "cuda":
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            m = Gemma3ForConditionalGeneration.from_pretrained(
                base_name, quantization_config=bnb, device_map="auto",
                attn_implementation=attn, **_hf_kwargs(hf_token)
            )
        else:
            m = Gemma3ForConditionalGeneration.from_pretrained(
                base_name, torch_dtype=dtype, attn_implementation=attn, **_hf_kwargs(hf_token)
            ).to(device)
        # Make sure config reflects the chosen impl (used in forward)
        m.config._attn_implementation = attn
        return m

    try:
        return _actually_load(attn_impl), attn_impl
    except Exception as e:
        if attn_impl != "eager":
            print(f"[infer] '{attn_impl}' attention failed ({e.__class__.__name__}). Retrying with 'eager'…",
                  file=sys.stderr)
            return _actually_load("eager"), "eager"
        raise

def main():
    args = parse()
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN") or ""

    device = _select_device(args.device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[infer] base={args.base}", file=sys.stderr)
    print(f"[infer] adapter={args.adapter}", file=sys.stderr)
    print(f"[infer] device={device}, torch={torch.__version__}, "
          f"cuda_available={torch.cuda.is_available()}, cuda_version={getattr(torch.version, 'cuda', None)}",
          file=sys.stderr)
    print(f"[infer] hf_token={_mask(hf_token)}", file=sys.stderr)
    print(f"[infer] attn(requested)={args.attn}, 4bit={bool(args.load_in_4bit)}", file=sys.stderr)
    if device == "cpu":
        print("[infer] WARNING: running on CPU — first token can take a while on 4B.", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, **_hf_kwargs(hf_token))
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    base, attn_used = _load_base(args.base, dtype, args.attn, hf_token, device, args.load_in_4bit)
    print(f"[infer] attn(used)={attn_used}", file=sys.stderr)

    from peft import PeftModel
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).to(device).eval()

    messages = [{"role": "user", "content": args.prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tok.pad_token_id,
        streamer=streamer,
        use_cache=True,
    )

    print("[infer] starting generation…", file=sys.stderr)
    t0 = time.time()

    def _generate():
        try:
            with torch.inference_mode():
                model.generate(**gen_kwargs)
        except Exception as e:
            print(f"\n[infer] generation error: {e}", file=sys.stderr)

    th = threading.Thread(target=_generate, daemon=True)
    th.start()

    first = True
    try:
        for piece in streamer:
            if first:
                print(f"[infer] first token after {time.time()-t0:.2f}s", file=sys.stderr)
                first = False
            print(piece, end="", flush=True)
    finally:
        th.join()
        if sys.stdout and getattr(sys.stdout, "isatty", lambda: False)():
            print()

if __name__ == "__main__":
    main()
