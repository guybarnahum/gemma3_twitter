#!/usr/bin/env python3
import argparse, os, sys, threading, time, torch

# Keep things stable across stacks (avoid Dynamo graph issues on Gemma3 attn)
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
try:
    import torch._dynamo as dynamo
    dynamo.config.suppress_errors = True
except Exception:
    pass

from transformers import (
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    TextIteratorStreamer,
)

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to merged model folder")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system", default=None, help="Optional system instruction")

    # Default None → auto-fill up to context window (minus prompt)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--min_new_tokens", type=int, default=0)

    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)

    ap.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
    ap.add_argument("--load_in_4bit", action="store_true", help="(GPU only) bitsandbytes 4-bit")
    ap.add_argument("--attn", choices=["sdpa","eager","flash2"], default="sdpa")
    ap.add_argument("--no_eos", action="store_true", help="Ignore EOS/EOT; stop only at token limits")
    return ap.parse_args()

def _select_device(opt: str):
    if opt == "cuda": return "cuda" if torch.cuda.is_available() else "cpu"
    if opt == "cpu":  return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def _model_ctx_len(model, tok):
    for k in ("max_position_embeddings", "max_seq_len", "max_sequence_length"):
        v = getattr(model.config, k, None)
        if isinstance(v, int) and v > 0:
            return v
    return 8192  # sensible fallback

def _load_base(local_path, dtype, attn_impl, device, load_in_4bit):
    def _actually_load(attn):
        if load_in_4bit and device == "cuda":
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            m = Gemma3ForConditionalGeneration.from_pretrained(
                local_path,
                quantization_config=bnb,
                device_map="auto",          # let HF shard across GPUs if present
                attn_implementation=attn,
            )
        else:
            m = Gemma3ForConditionalGeneration.from_pretrained(
                local_path,
                torch_dtype=dtype,
                attn_implementation=attn,
            ).to(device)
        m.config._attn_implementation = attn
        return m

    try:
        return _actually_load(attn_impl), attn_impl
    except Exception:
        if attn_impl != "eager":
            print(f"[infer] '{attn_impl}' failed; retrying with 'eager'…", file=sys.stderr)
            return _actually_load("eager"), "eager"
        raise

def main():
    args = parse()

    device = _select_device(args.device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[infer] model={args.model}", file=sys.stderr)
    print(f"[infer] device={device}, torch={torch.__version__}, "
          f"cuda_available={torch.cuda.is_available()}, cuda_version={getattr(torch.version, 'cuda', None)}",
          file=sys.stderr)
    print(f"[infer] attn={args.attn} | 4bit={bool(args.load_in_4bit)}", file=sys.stderr)
    if device == "cpu":
        print("[infer] WARNING: running on CPU — first token can be slow on 4B.", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    model, attn_used = _load_base(args.model, dtype, args.attn, device, args.load_in_4bit)
    print(f"[infer] attn(used)={attn_used}", file=sys.stderr)

    # Optional: ignore EOS/EOT (also set in gen kwargs later)
    if args.no_eos:
        model.generation_config.eos_token_id = None

    # Build chat messages
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(device)

    # Length controls
    input_len = ids["input_ids"].shape[-1]
    if args.max_new_tokens is None:
        ctx = _model_ctx_len(model, tok)
        auto_max_new = max(32, int(ctx - input_len - 8))
        max_new_tokens = auto_max_new
        max_note = "auto"
    else:
        max_new_tokens = args.max_new_tokens
        max_note = "user"

    logits_processor = None
    min_length = None
    if args.min_new_tokens > 0:
        try:
            from transformers.generation.logits_process import (
                LogitsProcessorList, MinNewTokensLengthLogitsProcessor,
            )
            eos_for_min = None if args.no_eos else tok.eos_token_id
            logits_processor = LogitsProcessorList([
                MinNewTokensLengthLogitsProcessor(
                    prompt_length=input_len,
                    min_new_tokens=args.min_new_tokens,
                    eos_token_id=eos_for_min,
                )
            ])
            print(f"[infer] min_new_tokens via logits_processor={args.min_new_tokens}", file=sys.stderr)
        except Exception:
            # Fallback for older Transformers
            min_length = input_len + args.min_new_tokens
            if max_new_tokens is not None:
                min_length = min(min_length, input_len + max_new_tokens)
            print(f"[infer] MinNewTokens missing → using min_length={min_length}", file=sys.stderr)

    # Block common EOT-like tokens to avoid early stops
    bad_words_ids = []
    for t in ("<end_of_turn>", "<eot>", "<|eot_id|>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0:
            bad_words_ids.append([tid])

    # Streamer
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tok.pad_token_id,
        streamer=streamer,
        use_cache=True,
        repetition_penalty=args.repetition_penalty,
    )
    if logits_processor is not None:
        gen_kwargs["logits_processor"] = logits_processor
    if min_length is not None:
        gen_kwargs["min_length"] = min_length
    if args.no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    if args.no_eos:
        gen_kwargs["eos_token_id"] = None
    if bad_words_ids:
        gen_kwargs["bad_words_ids"] = bad_words_ids

    print(f"[infer] limits: max_new={max_new_tokens} ({max_note}), min_new={args.min_new_tokens}, no_eos={bool(args.no_eos)}",
          file=sys.stderr)
    print("[infer] starting generation…", file=sys.stderr)
    t0 = time.time()

    def _generate():
        try:
            with torch.inference_mode():
                model.generate(**gen_kwargs)
        except Exception as e:
            print(f"\n[infer] generation error: {e}", file=sys.stderr)

    th = threading.Thread(target=_generate, daemon=True)
    th.start()

    first = True
    try:
        for piece in streamer:
            if first:
                print(f"[infer] first token after {time.time()-t0:.2f}s", file=sys.stderr)
                first = False
            print(piece, end="", flush=True)
    finally:
        th.join()
        if sys.stdout and getattr(sys.stdout, "isatty", lambda: False)():
            print()

if __name__ == "__main__":
    main()
