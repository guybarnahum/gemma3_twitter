#!/usr/bin/env python3
import argparse, torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to merged model folder")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    return ap.parse_args()

def main():
    args = parse()
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else None
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, device_map="auto", torch_dtype=dtype
    )

    messages = [{"role":"user","content": args.prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)

    out = model.generate(
        **ids, max_new_tokens=args.max_new_tokens,
        do_sample=True, temperature=args.temperature, top_p=args.top_p
    )
    print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()

