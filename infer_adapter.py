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
    # Transformers >=4.41: 'token'; older: 'use_auth_token'
    if not token:
        return {}
    try:
        return {"token": token}
    except TypeError:
        return {"use_auth_token": token}

def main():
    args = parse()
    hf_token = (
        os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
        or ""
    )

    dtype = torch.bfloat16 if torch.cuda.is_available() else None

    tok = AutoTokenizer.from_pretrained(
        args.base, use_fast=True, **_hf_kwargs(hf_token)
    )
    base = Gemma3ForConditionalGeneration.from_pretrained(
        args.base, device_map="auto", torch_dtype=dtype, **_hf_kwargs(hf_token)
    )

    from peft import PeftModel
    model = PeftModel.from_pretrained(base, args.adapter)

    messages = [{"role":"user","content": args.prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)

    out = model.generate(
        **ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tok.eos_token_id,
    )
    print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
