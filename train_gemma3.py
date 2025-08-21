#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA SFT (Gemma and compatible chat LMs) with TRL 0.9.6-style API.

- Loads JSONL with {"messages":[...]} or {"text":"..."} records.
- Applies model chat template when `messages` exist; saves to a `text` column.
- Uses SFTConfig (not raw TrainingArguments) and passes it to SFTTrainer as `args=`.
- Works on CPU/Mac (no bf16 on MPS) and CUDA if available.
"""

import os, math, json, argparse
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig


# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="LoRA SFT (Gemma et al.) with TRL 0.9.6 API")

    # Data
    ap.add_argument("--data", required=True, help="Train JSONL")
    ap.add_argument("--eval", default="", help="Eval JSONL (optional)")
    ap.add_argument("--cutoff_len", type=int, default=2048, help="Max sequence length (tokens)")
    ap.add_argument("--no_packing", action="store_true", help="Disable sample packing")

    # Model / output
    ap.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "google/gemma-3-270m-it"))
    ap.add_argument("--output_dir", default=os.environ.get("ADAPTER_DIR", "out/gemma3-twitter-lora"))
    ap.add_argument("--hf_token", default=os.environ.get("HUGGINGFACE_HUB_TOKEN", ""))

    # Train loop
    ap.add_argument("--epochs", type=float, default=float(os.environ.get("EPOCHS", "1")))
    ap.add_argument("--batch_size", type=int, default=1, help="Per-device train batch")
    ap.add_argument("--eval_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--logging_steps", type=int, default=25)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--report_to", default="", help='Comma-separated reporters (e.g., "wandb"); empty disables')

    # Precision / device
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    ap.add_argument("--fp16", action="store_true", help="Use fp16 on CUDA")
    ap.add_argument("--attn_impl", choices=["eager", "sdpa"], default=None,
                    help="Attention implementation. For Gemma3, 'eager' is recommended.")

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                    help='Comma-separated module names for LoRA (Gemma default provided)')

    return ap.parse_args()


# --------------- Data helpers ---------------
def jsonl_to_dataset(path: str):
    return load_dataset("json", data_files=path, split="train")


def apply_chat_template_or_fallback(example: Dict[str, Any], tokenizer: AutoTokenizer) -> str:
    msgs = example.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            pass
        # Fallback stitch
        parts = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)
    return example.get("text", "")


def ensure_text_column(ds, tokenizer):
    def _map(ex):
        return {"text": apply_chat_template_or_fallback(ex, tokenizer)}
    return ds.map(_map, desc="Applying chat template → text")


# ---------------- Main ----------------
def main():
    args = parse_args()
    set_seed(args.seed)

    # Resolve reporters
    report_to = [s.strip() for s in args.report_to.split(",") if s.strip()] if args.report_to else []

    # Choose device & dtype
    use_cuda = torch.cuda.is_available() and (args.device in ["auto", "cuda"])
    use_mps = (getattr(torch.backends, "mps", None) is not None) and torch.backends.mps.is_available() and (args.device in ["auto", "mps"])
    force_cpu = (args.device == "cpu") or (not use_cuda and not use_mps)

    if force_cpu:
        device_map = None
        torch_dtype = torch.float32
        print("[device] Using CPU (torch.float32)")
    elif use_cuda:
        device_map = "auto"
        torch_dtype = torch.float16 if args.fp16 else torch.float32
        print(f"[device] Using CUDA (dtype={torch_dtype})")
    else:  # MPS
        device_map = "auto"   # Transformers will place on mps
        torch_dtype = torch.float32  # bf16/half not supported reliably on MPS
        print("[device] Using MPS (torch.float32)")

    # Tokenizer
    token = args.hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, token=token)
    # Right padding avoids half-precision overflow issues in trainers
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # after you compute `token`
    attn_impl = args.attn_impl
    try:
        cfg = AutoConfig.from_pretrained(args.model_name, token=token, trust_remote_code=True)
        if attn_impl is None and getattr(cfg, "model_type", "") in ("gemma3", "gemma3_text"):
            attn_impl = "eager"
    except Exception:
        pass

    # Datasets → text
    train_ds = ensure_text_column(jsonl_to_dataset(args.data), tokenizer)
    eval_ds = None
    if args.eval and os.path.exists(args.eval):
        eval_ds = ensure_text_column(jsonl_to_dataset(args.eval), tokenizer)

    # Model
    _ = AutoConfig.from_pretrained(args.model_name, token=token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        token=token,
        attn_implementation=attn_impl
    )

    # LoRA
    target_modules = [t.strip() for t in args.target_modules.split(",") if t.strip()]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # SFTConfig (TRL 0.9.x expects these here; NOT in SFTTrainer kwargs)
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=(args.eval_steps if eval_ds is not None else None),
        eval_strategy=("steps" if eval_ds is not None else "no"),
        report_to=report_to if report_to else None,
        remove_unused_columns=True,          # we use the 'text' column only
        dataset_text_field="text",
        max_seq_length=args.cutoff_len,
        packing=(not args.no_packing),
        fp16=(use_cuda and args.fp16),
        bf16=False,                          # keep False for CPU/MPS safety
        gradient_checkpointing=False,        # off by default; turn on if needed
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,          # <- pass SFTConfig here (correct API)
        tokenizer=tokenizer,      # accepted in TRL 0.9.x; deprecation is for future versions
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )

    # Train
    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Evaluate
    metrics = {}
    if eval_ds is not None:
        eval_metrics = trainer.evaluate()
        metrics.update(eval_metrics)
        if "eval_loss" in eval_metrics and eval_metrics["eval_loss"] is not None:
            try:
                metrics["eval_ppl"] = math.exp(eval_metrics["eval_loss"])
            except OverflowError:
                metrics["eval_ppl"] = float("inf")

    # Log a tiny summary
    print("\n=== Training summary ===")
    for k in sorted(metrics.keys()):
        print(f"{k}: {metrics[k]}")
    print(f"\nAdapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

