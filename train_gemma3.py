#!/usr/bin/env python3
import os, glob, json, argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers import Gemma3ForConditionalGeneration
from trl import SFTTrainer

def latest_checkpoint_dir(outdir: str):
    paths = sorted(glob.glob(os.path.join(outdir, "checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-",1)[-1].isdigit() else -1)
    return paths[-1] if paths else None

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("MODEL_NAME", "google/gemma-3-4b-it"))
    ap.add_argument("--data", default="dataset/train.jsonl")
    ap.add_argument("--eval", default="dataset/train_eval.jsonl")
    ap.add_argument("--out",  default="out/gemma3-twitter-lora")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--resume", action="store_true", help="Force resume if a checkpoint exists")
    ap.add_argument("--lr-resume-scale", type=float, default=0.3, help="Multiply LR by this when resuming")
    ap.add_argument("--no_bnb", action="store_true", help="Disable 4-bit quantization (bitsandbytes)")
    ap.add_argument("--eval_steps", type=int, default=400)
    ap.add_argument("--save_steps", type=int, default=400)
    ap.add_argument("--early_stop_patience", type=int, default=3)
    return ap.parse_args()

def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Load datasets
    train = load_dataset("json", data_files=args.data, split="train")
    eval_ = load_dataset("json", data_files=args.eval, split="train") if os.path.exists(args.eval) else None

    # Map messages -> plain text using Gemma's chat template
    def to_text(ex):
        return {"text": tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )}
    train = train.map(to_text, remove_columns=train.column_names)
    if eval_:
        eval_ = eval_.map(to_text, remove_columns=eval_.column_names)

    # Optional 4-bit QLoRA for memory efficiency (Linux-only bnb)
    bnb_cfg = None
    if not args.no_bnb:
        try:
            bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        except Exception:
            bnb_cfg = None  # gracefully fall back if bnb not installed

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=bnb_cfg,
        device_map="auto"
    )

    # Find checkpoint if present
    ckpt = latest_checkpoint_dir(args.out)
    do_resume = bool(ckpt) and (args.resume or True)  # default: auto-resume if any checkpoint exists
    lr = args.lr * (args.lr_resume_scale if do_resume else 1.0)

    train_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        logging_steps=50,
        save_steps=args.save_steps,
        evaluation_strategy="steps" if eval_ else "no",
        eval_steps=args.eval_steps if eval_ else None,
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05 if not do_resume else 0.0,  # tiny/no warmup on resume
        report_to="none",
        load_best_model_at_end=True if eval_ else False,
        metric_for_best_model="loss",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train,
        eval_dataset=eval_,
        dataset_text_field="text",
        packing=False,  # tweets are short; no need to pack
        args=train_args,
        peft_config={
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        },
        callbacks=([EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience)] if eval_ else [])
    )

    resume_arg = ckpt if do_resume else None
    if resume_arg:
        print(f"Resuming from {resume_arg} with LR={lr:.2e}")
    else:
        print(f"Fresh run. LR={lr:.2e}")

    trainer.train(resume_from_checkpoint=resume_arg)
    trainer.save_model()
    tokenizer.save_pretrained(args.out)

if __name__ == "__main__":
    main()

