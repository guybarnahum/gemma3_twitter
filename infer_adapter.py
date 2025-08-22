#!/usr/bin/env python3
import argparse, os, sys, threading, time, torch

# Disable Dynamo jit weirdness by default (safer on Gemma3 attn)
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
    ap.add_argument("--base", default="google/gemma-3-4b-it")
    ap.add_argument("--adapter", required=True, help='Path to LoRA, or "none" to use base only')
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system", default=None, help="Optional system message to steer style/length")

    # IMPORTANT: default None → "no limit" behavior (auto up to context window)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--min_new_tokens", type=int, default=0)

    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
    ap.add_argument("--load_in_4bit", action="store_true", help="(GPU only) use bitsandbytes 4-bit")
    ap.add_argument("--attn", choices=["sdpa","eager","flash2"], default="sdpa")
    ap.add_argument("--no_eos", action="store_true", help="Ignore EOS/EOT; run until min/max/context")
    return ap.parse_args()

def _hf_kwargs(token: str):
    try:
        return {"token": token} if token else {}
    except TypeError:
        return {"use_auth_token": token} if token else {}

def _mask(tok: str) -> str:
    return tok[:6] + "…" if tok and len(tok) > 8 else ("(none)" if not tok else tok)

def _select_device(opt: str):
    if opt == "cuda": return "cuda" if torch.cuda.is_available() else "cpu"
    if opt == "cpu":  return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def _model_ctx_len(model, tok):
    # Prefer model config; tokenizer.model_max_length is often a huge sentinel
    for k in ("max_position_embeddings", "max_seq_len", "max_sequence_length"):
        v = getattr(model.config, k, None)
        if isinstance(v, int) and v > 0:
            return v
    # Fallback to something sensible if config lacks it
    return 8192

def _load_base(name, dtype, attn_impl, hf_token, device, load_in_4bit):
    def _actually_load(attn):
        if load_in_4bit and device == "cuda":
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            m = Gemma3ForConditionalGeneration.from_pretrained(
                name, quantization_config=bnb, device_map="auto",
                attn_implementation=attn, **_hf_kwargs(hf_token)
            )
        else:
            m = Gemma3ForConditionalGeneration.from_pretrained(
                name, torch_dtype=dtype, attn_implementation=attn, **_hf_kwargs(hf_token)
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
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN") or ""

    device = _select_device(args.device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[infer] base={args.base}", file=sys.stderr)
    print(f"[infer] adapter={args.adapter}", file=sys.stderr)
    print(f"[infer] device={device}, torch={torch.__version__}, "
          f"cuda_available={torch.cuda.is_available()}, cuda_version={getattr(torch.version, 'cuda', None)}",
          file=sys.stderr)
    print(f"[infer] hf_token={_mask(hf_token)} | attn={args.attn} | 4bit={bool(args.load_in_4bit)}",
          file=sys.stderr)
    if device == "cpu":
        print("[infer] WARNING: running on CPU — first token can be slow on 4B.", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, **_hf_kwargs(hf_token))
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    base, attn_used = _load_base(args.base, dtype, args.attn, hf_token, device, args.load_in_4bit)
    print(f"[infer] attn(used)={attn_used}", file=sys.stderr)

    # Optional: ignore EOS/EOT so generation only stops at token limits
    if args.no_eos:
        base.generation_config.eos_token_id = None

    # Attach LoRA unless explicitly disabled
    if args.adapter.lower() != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).to(device).eval()
    else:
        print("[infer] adapter=none → using BASE MODEL ONLY", file=sys.stderr)
        model = base.to(device).eval()

    # Messages with optional system instruction
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(device)

    # ----- Length controls -----
    input_len = ids["input_ids"].shape[-1]
    # If max_new_tokens omitted → auto up to context window minus prompt (with margin)
    if args.max_new_tokens is None:
        ctx = _model_ctx_len(model, tok)
        auto_max_new = max(32, int(ctx - input_len - 8))
        max_new_tokens = auto_max_new
        max_note = "auto"
    else:
        max_new_tokens = args.max_new_tokens
        max_note = "user"

    # Try to enforce a minimum length via logits processor if available; else emulate via min_length
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
            # Fallback: total min length = input_len + min_new_tokens (clamped to max if necessary)
            min_length = input_len + args.min_new_tokens
            if max_new_tokens is not None:
                min_length = min(min_length, input_len + max_new_tokens)
            print(f"[infer] MinNewTokens processor missing → using min_length={min_length}", file=sys.stderr)

    # Optionally block common end-of-turn tokens to avoid early stops from chat/tweet priors
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
        gen_kwargs["eos_token_id"] = None  # ensure no EOS stopping
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
