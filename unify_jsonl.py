#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unify multiple JSONL datasets into a single, normalized Gemma/Chat SFT JSONL.

- Normalizes rows to a minimal schema so HF `datasets` doesn't choke:
    default keep keys: ["messages", "created_at"]
- Maps roles to a consistent set: {"system", "user", "assistant"}
    ("model" -> "assistant"; unknown -> "user")
- Optionally coerces {"text": "..."} into messages with a fallback prompt.
- Deduplicates by last assistant message (or whole text).
- Optional shuffle for training.

Examples
--------
python unify_jsonl.py \
  --in dataset/tweets_base.jsonl --in dataset/docs.jsonl \
  --out dataset/train.jsonl \
  --shuffle --seed 13 \
  --keep messages --keep created_at \
  --coerce-messages --prompt-fallback "Write a paragraph in my signature style."

"""

import argparse, json, pathlib, sys, hashlib, html, re, random
from typing import List, Dict, Any, Iterable, Tuple, Optional

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

WS_RE = re.compile(r"\s+")

def norm_text(t: str) -> str:
    t = html.unescape(t or "")
    t = WS_RE.sub(" ", t).strip()
    return t

def hash_for_dedup(messages: Optional[List[Dict[str, str]]], text: str) -> str:
    if messages:
        # use last assistant turn if present
        for m in reversed(messages):
            if m.get("role","").lower() in ("assistant", "model"):
                return hashlib.sha256(norm_text(m.get("content","")).lower().encode("utf-8")).hexdigest()
        # else hash the whole stitched dialog
        stitched = " ".join(norm_text(m.get("content","")) for m in messages)
        return hashlib.sha256(stitched.lower().encode("utf-8")).hexdigest()
    else:
        return hashlib.sha256(norm_text(text).lower().encode("utf-8")).hexdigest()

def map_role(role: str) -> str:
    r = (role or "").strip().lower()
    if r in ("assistant", "model"): return "assistant"
    if r in ("user", "system"): return r
    return "user"

def coerce_messages_from_text(text: str, prompt: str, assistant_role: str) -> List[Dict[str,str]]:
    text = norm_text(text)
    if not text: return []
    return [
        {"role":"user", "content": prompt},
        {"role": assistant_role, "content": text}
    ]

def normalize_row(row: Dict[str, Any],
                  keep_keys: List[str],
                  coerce: bool,
                  prompt_fallback: str,
                  assistant_role: str) -> Optional[Dict[str, Any]]:
    """
    Returns normalized row or None to drop.
    """
    msgs = row.get("messages")
    text = row.get("text")

    norm: Dict[str, Any] = {}

    if isinstance(msgs, list):
        cleaned: List[Dict[str,str]] = []
        for m in msgs:
            if not isinstance(m, dict): continue
            content = m.get("content")
            if not isinstance(content, str): continue
            content = norm_text(content)
            if not content: continue
            role = map_role(m.get("role","user"))
            cleaned.append({"role": role, "content": content})
        # need at least one user + one assistant for SFT
        has_user = any(m["role"]=="user" for m in cleaned)
        has_asst = any(m["role"]=="assistant" for m in cleaned)
        if not (has_user and has_asst):
            # try to coerce if there's a single segment left
            if coerce and not has_asst and cleaned:
                # last becomes assistant, synthesize a generic user prompt
                last = cleaned[-1]["content"]
                cleaned = [{"role":"user","content": prompt_fallback},
                           {"role":"assistant","content": last}]
                has_user = has_asst = True
        if not (has_user and has_asst):
            return None
        norm["messages"] = cleaned
    elif isinstance(text, str) and text.strip():
        if not coerce:
            return None
        norm["messages"] = coerce_messages_from_text(text, prompt_fallback, assistant_role)
        if not norm["messages"]:
            return None
    else:
        return None

    # keep only whitelisted keys that exist in source
    for k in keep_keys:
        if k == "messages":
            norm["messages"] = norm.get("messages", [])
        elif k in row and k not in ("messages","text"):
            norm[k] = row[k]
    # ensure created_at is string if kept
    if "created_at" in norm and norm["created_at"] is not None:
        norm["created_at"] = str(norm["created_at"])

    return norm

def iter_jsonl(paths: List[pathlib.Path]) -> Iterable[Dict[str, Any]]:
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

def main():
    ap = argparse.ArgumentParser(description="Unify and normalize multiple JSONL datasets.")
    ap.add_argument("--in", dest="inputs", action="append", required=True,
                    help="Input JSONL (repeatable)")
    ap.add_argument("--out", required=True, help="Output JSONL")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--normalize", action="store_true", default=True,
                    help="(default) normalize schema/roles; use --no-normalize to disable",
                    )
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    ap.add_argument("--keep", action="append", default=None,
                    help="Keys to preserve (repeatable). Default: messages,created_at")
    ap.add_argument("--coerce-messages", action="store_true",
                    help="Convert {'text': ...} rows into chat messages using a fallback prompt")
    ap.add_argument("--prompt-fallback", default="Write a paragraph in my signature style.",
                    help="User prompt used with --coerce-messages")
    ap.add_argument("--assistant-role", default="assistant",
                    help="Role label for target when coercing from text")
    args = ap.parse_args()

    in_paths = [pathlib.Path(x) for x in args.inputs]
    for p in in_paths:
        if not p.exists():
            sys.exit(f"[unify] missing input: {p}")

    keep_keys = args.keep if args.keep else ["messages","created_at"]

    # Read all rows
    all_rows: List[Dict[str,Any]] = []
    total_in = 0
    iterator = iter_jsonl(in_paths)

    def _progress(iterable, total=None, desc=""):
        if args.no_progress or tqdm is None:
            return iterable
        return tqdm(iterable, total=total, desc=desc, unit="rows")

    for row in _progress(iterator, desc="Reading"):
        total_in += 1
        all_rows.append(row)

    kept: List[Dict[str,Any]] = []
    dropped_schema = 0
    seen_hashes = set()

    # Normalize / dedup
    it = _progress(all_rows, desc="Normalizing")
    for row in it:
        if args.normalize:
            norm = normalize_row(
                row,
                keep_keys=keep_keys,
                coerce=args.coerce_messages,
                prompt_fallback=args.prompt_fallback,
                assistant_role=args.assistant_role,
            )
            if norm is None:
                dropped_schema += 1
                continue
            row = norm
        # dedup
        msgs = row.get("messages")
        text = row.get("text","")
        h = hash_for_dedup(msgs, text)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        kept.append(row)

    # Shuffle
    if args.shuffle:
        rnd = random.Random(args.seed)
        rnd.shuffle(kept)

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[unify] inputs={len(in_paths)} rows_in={total_in} kept={len(kept)} dropped_schema={dropped_schema} dups={len(seen_hashes)-len(kept)}")
    print(f"[unify] wrote: {outp}")

if __name__ == "__main__":
    main()

