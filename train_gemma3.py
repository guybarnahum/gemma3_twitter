#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA SFT trainer for Gemma (and similar chat LMs), with:
- BF16/FP16 control
- Optional 4-bit (bitsandbytes) QLoRA
- Gradient checkpointing
- Gemma-3 eager attention by default
- Chat-template application (messages -> text)

Compatible with:
  transformers==4.46.x, trl==0.9.6, accelerate==0.33.x, peft==0.13.x, torch 2.3.x

Example:
  python train_gemma3.py \
    --data dataset/train.jsonl \
    --eval dataset/train_eval.jsonl \
    --model_name google/gemma-3-270m-it \
    --epochs 1 --batch_size 2 --grad_accum 8 \
    --cutoff_len 1024 --bf16 --load_in_4bit --gradient_checkpointing
"""

import os
import math
import argparse
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# ----------------------- CLI -----------------------

def parse_args():
    ap = argparse.ArgumentParser(description="LoRA SFT trainer (Gemma et al.)")

    # Data
    ap.add_argument("--data", required=True, help="Train JSONL")
    ap.add_argument("--eval", default="", help="Eval JSONL (optional)")
    ap.add_argument("--cutoff_len", type=int, default=1024, help="Max sequence length (tokens)")
    ap.add_argument("--no_packing", action="store_true", help="Disable sample packing")

    # Model / repo
    ap.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "google/gemma-3-270m-it"))
    ap.add_argument("--output_dir", default=os.environ.get("ADAPTER_DIR", "out/gemma3-twitter-lora"))
    ap.add_argument("--hf_token", default=os.environ.get("HUGGINGFACE_HUB_TOKEN", ""))

    # Train loop
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "1")))
    ap.add_argument("--batch_size", type=int, default=2, help="Per-device train batch size")
    ap.add_argument("--eval_batch_size", type=int, default=2, help="Per-device eval batch size")
    ap.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--report_to", default="", help='Comma-separated reporters (e.g., "wandb"); empty disables')
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    ap.add_argument("--fp16", action="store_true", help="Use float16")
    ap.add_argument("--bf16", action="store_true", help="Use bfloat16")
    ap.add_argument("--attn_impl", choices=["eager", "sdpa"], default="eager")

    # New toggles
    ap.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit QLoRA (bitsandbytes)")
    ap.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    return ap.parse_args()


# -------------------- Data helpers --------------------

def jsonl_to_dataset(path: str):
    return load_dataset("json", data_files=path, split="train")


def apply_chat_template_or_fallback(example: Dict[str, Any], tokenizer: AutoTokenizer) -> str:
    msgs = example.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
        try:
            return tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass
        # fallback: stitch role: content
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


# ------------------------ Main ------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Device / dtype
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # Dtype selection
    use_bf16 = bool(args.bf16)
    use_fp16 = bool(args.fp16)
    if device == "cuda":
        bf16_ok = torch.cuda.is_bf16_supported()
        if use_bf16 and not bf16_ok:
            print("[warn] bf16 requested but not supported on this GPU; falling back to fp16.")
            use_bf16 = False
            use_fp16 = True
    else:
        # bf16/fp16 training not meaningful off-GPU
        if use_bf16 or use_fp16:
            print("[warn] Half-precision requested but non-CUDA device in use; forcing float32.")
        use_bf16 = use_fp16 = False

    print(f"[device] Using {device.upper()} ({'bf16' if use_bf16 else 'fp16' if use_fp16 else 'float32'})")

    # Tokenizer
    token = args.hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Datasets -> mapped to 'text'
    train_ds = ensure_text_column(jsonl_to_dataset(args.data), tokenizer)
    eval_ds = None
    if args.eval and os.path.exists(args.eval):
        eval_ds = ensure_text_column(jsonl_to_dataset(args.eval), tokenizer)

    # Quantization config (optional QLoRA)
    quantization_config = None
    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            print("[qlora] Using 4-bit NF4 with compute dtype:", compute_dtype)
        except Exception as e:
            print("[warn] bitsandbytes not available; proceeding without 4-bit. Error:", e)
            quantization_config = None

    # Model config & load
    config = AutoConfig.from_pretrained(
        args.model_name,
        token=token,
        trust_remote_code=True,
    )

    torch_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    device_map = "auto" if device in ("cuda", "mps") else None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        token=token,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_impl,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        quantization_config=quantization_config,
    )

    if args.gradient_checkpointing:
        # Disable cache for gradient checkpointing
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    # LoRA
    tmods = [t.strip() for t in args.target_modules.split(",") if t.strip()]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=tmods,
    )

    # Reporters
    report_to = [s.strip() for s in args.report_to.split(",") if s.strip()] if args.report_to else []

    # SFT configuration (inherits TrainingArguments in TRL 0.9.6)
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=report_to,
        seed=args.seed,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        # SFT-specific
        dataset_text_field="text",
        max_seq_length=args.cutoff_len,
        packing=not args.no_packing,
        # Eval: if we pass eval_dataset, trainer.evaluate() will run afterwards
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        processing_class=tokenizer,        # (preferred over deprecated 'tokenizer' kw)
        peft_config=peft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    # Train
    train_result = trainer.train()
    trainer.save_model(args.output_dir)   # saves adapter
    tokenizer.save_pretrained(args.output_dir)

    # Evaluate (optional)
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

    # Persist metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("\n=== Training summary ===")
    for k in sorted(metrics.keys()):
        print(f"{k}: {metrics[k]}")
    print(f"\nAdapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
