#!/usr/bin/env python3
import argparse, os, sys, threading, torch
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
    return ap.parse_args()

def _hf_kwargs(token: str):
    # Works across transformers versions
    try:
        return {"token": token} if token else {}
    except TypeError:
        return {"use_auth_token": token} if token else {}

def main():
    args = parse()

    # Pull HF token from env if present (headless friendly)
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN") or ""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(
        args.base, use_fast=True, **_hf_kwargs(hf_token)
    )
    # Ensure we have a pad token to avoid warnings
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    # IMPORTANT: avoid device_map="auto" to prevent meta/offload path issues with PEFT
    base = Gemma3ForConditionalGeneration.from_pretrained(
        args.base,
        torch_dtype=dtype,
        attn_implementation="sdpa",  # faster on recent PyTorch/CUDA; harmless on CPU
        **_hf_kwargs(hf_token),
    ).to(device)

    from peft import PeftModel
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model = model.to(device)
    model.eval()

    # Chat template → prompt ids
    messages = [{"role": "user", "content": args.prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(device)

    # Streamer for live tokens
    streamer = TextIteratorStreamer(
        tok,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gen_kwargs = dict(
        **ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tok.pad_token_id,
        streamer=streamer,
    )

    # Run generation in a background thread; print pieces as they arrive
    def _generate():
        try:
            model.generate(**gen_kwargs)
        except KeyboardInterrupt:
            pass

    thread = threading.Thread(target=_generate, daemon=True)
    thread.start()

    try:
        for piece in streamer:
            # Print as tokens stream in; flush for immediate display
            print(piece, end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        thread.join()
        # Ensure trailing newline for clean shell prompt
        if sys.stdout and getattr(sys.stdout, "isatty", lambda: False)():
            print()

if __name__ == "__main__":
    main()
