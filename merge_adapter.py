#!/usr/bin/env python3
import argparse, torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from peft import PeftModel

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-3-4b-it")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args()

def main():
    args = parse()
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else None
    base = Gemma3ForConditionalGeneration.from_pretrained(
        args.base, device_map="auto", torch_dtype=dtype
    )
    lora = PeftModel.from_pretrained(base, args.adapter)
    merged = lora.merge_and_unload()
    merged.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved merged model to: {args.out}")

if __name__ == "__main__":
    main()

