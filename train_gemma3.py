#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma (and compatible chat LMs) SFT with LoRA + visible progress bars.

- Expects JSONL rows with either:
    {"messages":[{"role":...,"content":...}, ...]}
  or a plain:
    {"text":"..."}  (fallback)
- Applies the model's chat template when messages are present.
- Trains a LoRA adapter (QLoRA-capable) with TRL's SFTTrainer.
- Shows tqdm progress during train/eval, logs losses, and reports perplexity.
- Supports gated HF models via --hf_token or HUGGINGFACE_HUB_TOKEN.

Examples
--------
python train_gemma3.py \
  --data dataset/train.jsonl \
  --eval dataset/train_eval.jsonl \
  --output_dir out/gemma3-twitter-lora \
  --model_name google/gemma-3-4b-it \
  --epochs 1 --batch_size 2 --lr 2e-4 \
  --hf_token "$HUGGINGFACE_HUB_TOKEN"
"""

import os
import math
import json
import argparse
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
    logging as hf_logging,
)
from trl import SFTTrainer
from peft import LoraConfig


# ---------------- CLI ----------------

def parse_args():
    ap = argparse.ArgumentParser(description="LoRA SFT (Gemma et al.) with progress bars")

    # Data
    ap.add_argument("--data", required=True, help="Train JSONL file")
    ap.add_argument("--eval", default="", help="Eval JSONL file (optional)")
    ap.add_argument("--cutoff_len", type=int, default=2048, help="Max sequence length (tokens)")
    ap.add_argument("--packing", action="store_true", help="Enable sample packing (defaults to ON unless explicitly disabled)")
    ap.add_argument("--no_packing", action="store_true", help="Disable packing")

    # Model / output
    ap.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "google/gemma-3-4b-it"))
    ap.add_argument("--output_dir", default=os.environ.get("ADAPTER_DIR", "out/gemma3-twitter-lora"))
    ap.add_argument("--hf_token", default=os.environ.get("HUGGINGFACE_HUB_TOKEN", ""),
                    help="Hugging Face token for gated models")

    # Train loop
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "1")))
    ap.add_argument("--batch_size", type=int, default=2, help="Per-device train batch size")
    ap.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--resume", action="store_true", help="Resume from last checkpoint if present")
    ap.add_argument("--lr_scheduler_type", default="cosine",
                    choices=["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"])

    # Precision / memory
    ap.add_argument("--bf16", action="store_true", help="Use bfloat16 where available")
    ap.add_argument("--fp16", action="store_true", help="Use float16 where available")
    ap.add_argument("--load_in_4bit", action="store_true", help="Use QLoRA 4-bit loading (Linux+CUDA w/ bitsandbytes)")
    ap.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", default="all-linear",
                    help='e.g. "q_proj,v_proj" or "all-linear" to let TRL choose')

    # UX
    ap.add_argument("--disable_tqdm", action="store_true", help="Disable progress bars")
    ap.add_argument("--report_to", default="", help='Comma-separated reporters (e.g., "wandb"); empty disables')

    return ap.parse_args()


# --------------- Data helpers ---------------

def jsonl_to_dataset(path: str):
    """Load JSONL into an HF dataset split."""
    return load_dataset("json", data_files=path, split="train")


def apply_chat_template_or_fallback(example: Dict[str, Any], tokenizer: AutoTokenizer) -> str:
    """
    If the example has `messages`, format with the model's chat template.
    Otherwise, fall back to `text` or empty string.
    """
    msgs = example.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
        try:
            return tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass  # fall through to stitched fallback
        # crude fallback if template not available
        parts = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)
    return example.get("text", "")


def ensure_text_column(ds, tokenizer):
    """Map dataset to a 'text' column using chat template when applicable."""
    def _map(ex):
        return {"text": apply_chat_template_or_fallback(ex, tokenizer)}
    return ds.map(_map, desc="Applying chat template → text")


# ---------------- Main ----------------

def main():
    args = parse_args()

    # Respect tqdm setting
    os.environ["HF_DISABLE_PROGRESS_BARS"] = "1" if args.disable_tqdm else "0"
    hf_logging.set_verbosity_info()
    set_seed(args.seed)

    # Tokenizer
    token = args.hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Datasets
    train_ds = jsonl_to_dataset(args.data)
    train_ds = ensure_text_column(train_ds, tokenizer)
    eval_ds = None
    if args.eval and os.path.exists(args.eval):
        eval_ds = jsonl_to_dataset(args.eval)
        eval_ds = ensure_text_column(eval_ds, tokenizer)

    # Precision & quantization
    torch_dtype = None
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16

    quantization_config = None
    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype or torch.bfloat16,
            )
        except Exception:
            print("[warn] bitsandbytes not available; proceeding without 4-bit quantization")
            quantization_config = None

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        quantization_config=quantization_config,
        token=token,
    )

    # LoRA config
    target_modules = None if args.target_modules == "all-linear" else [
        x.strip() for x in args.target_modules.split(",") if x.strip()
    ]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # Packing toggle (default ON unless --no_packing passed)
    packing_flag = False if args.no_packing else True
    if args.packing:
        packing_flag = True

    # Reporters
    report_to = [s.strip() for s in args.report_to.split(",") if s.strip()] if args.report_to else []

    # TrainingArguments
    evaluation_strategy = "steps" if eval_ds is not None else "no"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_ds is not None else None,
        evaluation_strategy=evaluation_strategy,
        do_eval=eval_ds is not None,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to=report_to,  # [] disables W&B, etc.
        load_best_model_at_end=False,
        save_total_limit=3,
        max_grad_norm=1.0,
        logging_first_step=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        disable_tqdm=bool(args.disable_tqdm),
    )

    # SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=args.cutoff_len,
        packing=packing_flag,
    )

    # Resume
    resume_ckpt = None
    if args.resume and os.path.isdir(args.output_dir):
        cands = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if cands:
            latest = sorted(cands, key=lambda s: int(s.split("-")[-1]))[-1]
            resume_ckpt = os.path.join(args.output_dir, latest)
            print(f"[train] Resuming from {resume_ckpt}")

    # Train (progress via tqdm)
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(args.output_dir)   # saves LoRA adapter
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

    # Train metrics
    if train_result.metrics and "train_loss" in train_result.metrics:
        try:
            metrics["train_loss"] = train_result.metrics["train_loss"]
            metrics["train_ppl"] = math.exp(train_result.metrics["train_loss"])
        except Exception:
            pass

    # Persist & print
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("\n=== Training summary ===")
    for k in sorted(metrics.keys()):
        print(f"{k}: {metrics[k]}")
    print(f"\nAdapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

