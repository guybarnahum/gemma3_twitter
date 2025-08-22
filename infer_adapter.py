#!/usr/bin/env python3
import argparse, os, sys, threading, time, torch
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
    ap.add_argument("--device", choices=["auto","cuda","cpu"], default="auto",
                    help="Force device (default: auto)")
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="(GPU only) load base in 4-bit with bitsandbytes")
    return ap.parse_args()

def _hf_kwargs(token: str):
    try:
        return {"token": token} if token else {}
    except TypeError:
        return {"use_auth_token": token} if token else {}

def _mask(tok: str) -> str:
    return tok[:6] + "…" if tok and len(tok) > 8 else ("(none)" if not tok else tok)

def main():
    args = parse()

    # Headless-friendly HF auth
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN") or ""

    # Device select
    if args.device == "cuda":
        use_cuda = torch.cuda.is_available()
    elif args.device == "cpu":
        use_cuda = False
    else:
        use_cuda = torch.cuda.is_available()

    device = "cuda" if use_cuda else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Diagnostics up front
    print(f"[infer] base={args.base}", file=sys.stderr)
    print(f"[infer] adapter={args.adapter}", file=sys.stderr)
    print(f"[infer] device={device}, torch={torch.__version__}, "
          f"cuda_available={torch.cuda.is_available()}, cuda_version={getattr(torch.version, 'cuda', None)}",
          file=sys.stderr)
    print(f"[infer] hf_token={_mask(hf_token)}", file=sys.stderr)

    if device == "cpu":
        # Helpful heads-up so it doesn't feel stuck
        print("[infer] WARNING: running on CPU — first token may take a while for 4B. "
              "Use a GPU for speed.", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(
        args.base, use_fast=True, **_hf_kwargs(hf_token)
    )
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    # Load base model
    if args.load_in_4bit and device == "cuda":
        # Optional fast path if you installed bitsandbytes
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        base = Gemma3ForConditionalGeneration.from_pretrained(
            args.base,
            quantization_config=bnb,
            device_map="auto",
            attn_implementation="sdpa",
            **_hf_kwargs(hf_token),
        )
    else:
        base = Gemma3ForConditionalGeneration.from_pretrained(
            args.base,
            torch_dtype=dtype,
            attn_implementation="sdpa",  # OK on CPU too, but remove if you prefer
            **_hf_kwargs(hf_token),
        ).to(device)

    # Attach LoRA
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model = model.to(device)
    model.eval()

    # Build input
    messages = [{"role": "user", "content": args.prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(device)

    # Streamer
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

    # Async generate
    def _generate():
        try:
            model.generate(**gen_kwargs)
        except Exception as e:
            print(f"\n[infer] generation error: {e}", file=sys.stderr)

    print("[infer] starting generation…", file=sys.stderr)
    t0 = time.time()
    thread = threading.Thread(target=_generate, daemon=True)
    thread.start()

    # Print streamed text
    first_piece = True
    try:
        for piece in streamer:
            if first_piece:
                dt = time.time() - t0
                print(f"[infer] first token after {dt:.2f}s", file=sys.stderr)
                first_piece = False
            print(piece, end="", flush=True)
    except KeyboardInterrupt:
        print("\n[infer] interrupted.", file=sys.stderr)
    finally:
        thread.join()
        # tidy newline
        if sys.stdout and getattr(sys.stdout, "isatty", lambda: False)():
            print()

if __name__ == "__main__":
    main()
