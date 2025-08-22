#!/usr/bin/env python3
import argparse, os, torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

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

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, **_hf_kwargs(hf_token))

    # IMPORTANT: no device_map="auto" (avoids meta/offload), just load then .to(device)
    base = Gemma3ForConditionalGeneration.from_pretrained(
        args.base,
        torch_dtype=dtype,
        attn_implementation="sdpa",   # optional, faster on modern PyTorch
        **_hf_kwargs(hf_token)
    ).to(device)

    from peft import PeftModel
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model = model.to(device)
    model.eval()

    messages = [{"role":"user","content": args.prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(device)

    out = model.generate(
        **ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=True, temperature=args.temperature, top_p=args.top_p,
        pad_token_id=tok.eos_token_id,
    )
    print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
