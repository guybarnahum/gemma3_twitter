#!/usr/bin/env python3
# save as twitter_to_gemma_jsonl.py
import json, re, argparse, gzip, glob, os, html
from datetime import datetime
from pathlib import Path

JS_PREFIXES = (
    "window.YTD.tweets.part",        # classic export
    "window.YTD.tweet.part",         # variation
    "window.YTD.account.part",       # other files
)

def _load_tweet_objects(path):
    """
    Supports:
      - big single JS file: data/tweets.js
      - sharded parts: data/tweets/*.js
      - raw JSON lines or array
    Returns a list of {'tweet': {...}} or bare tweet dicts.
    """
    p = Path(path)
    files = []
    if p.is_dir():
        # common layouts
        maybe = list(p.glob("data/tweets.js")) + list(p.glob("data/tweets/*.js")) + list(p.glob("tweets/*.js")) + list(p.glob("*.json")) + list(p.glob("*.jsonl"))
        files = [str(x) for x in maybe]
    else:
        files = [str(p)]

    out = []
    for f in files:
        text = open(f, "r", encoding="utf-8").read()
        # strip JS assignment wrapper if present
        if any(text.startswith(pref) for pref in JS_PREFIXES):
            # find first '[' or '{' and parse the trailing JSON
            i = min([i for i in [text.find("["), text.find("{")] if i != -1])
            payload = text[i:]
            data = json.loads(payload)
            # classic export: list of {"tweet": {...}}
            if isinstance(data, list):
                out.extend(data)
        else:
            # raw JSON or JSONL
            text = text.strip()
            if text.startswith("["):
                out.extend(json.loads(text))
            else:
                # JSONL
                for line in text.splitlines():
                    line=line.strip()
                    if not line: continue
                    out.append(json.loads(line))
    return out

def _is_retweet(t):
    txt = t.get("full_text") or t.get("text") or ""
    return bool(t.get("retweeted_status_id") or txt.startswith("RT @"))

def _is_quote(t):
    # X archives vary; some include "is_quote_status" or quoted status id fields.
    return bool(t.get("is_quote_status") or t.get("quoted_status_id") or t.get("quoted_status"))

def _extract_text(t):
    txt = t.get("full_text") or t.get("text") or ""
    # unescape & normalize whitespace
    txt = html.unescape(txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def _extract_topics(text, entities):
    """Simple heuristic: hashtags + cleaned keywords."""
    topics = set()
    # hashtags
    for h in (entities or {}).get("hashtags", []):
        tag = h.get("text") or h.get("tag") or ""
        if tag: topics.add(f"#{tag}")
    # naive keywords from remaining text (drop links, mentions)
    t = re.sub(r"https?://\S+", "", text)
    t = re.sub(r"[@#]\w+", "", t)
    words = re.findall(r"[A-Za-z][A-Za-z0-9’'-]{2,}", t)
    # keep a few frequent-ish content words
    for w in words[:6]:
        topics.add(w.lower())
    return sorted(list(topics))[:6]

def _clean_tweet_text_for_label(txt):
    # keep URLs & emojis if part of your style, but you can strip tracking params:
    txt = re.sub(r"(https?://\S+?)(\?|&)(utm_[^=\s]+=\S+)", r"\1", txt)
    return txt.strip()

def convert(archive_path, out_path, min_len=10, include_quotes=True):
    rows = _load_tweet_objects(archive_path)
    examples = []
    for row in rows:
        t = row.get("tweet", row)  # some are {"tweet": {...}}
        if not isinstance(t, dict): continue

        if _is_retweet(t):
            continue  # skip RTs (not your voice)
        if _is_quote(t) and not include_quotes:
            continue

        txt = _extract_text(t)
        if len(txt) < min_len:
            continue

        entities = t.get("entities") or {}
        topics = _extract_topics(txt, entities)
        if not topics:
            # fallback: generic topic
            topics = ["general"]

        # Build SFT message pair
        user_prompt = (
            "Write a concise tweet in my signature style about: "
            + ", ".join(topics)
            + ". Keep it under 280 characters. Use my tone and phrasing."
        )
        label = _clean_tweet_text_for_label(txt)

        # Optional: keep thread id/date for dedup & eval splits
        created_at = t.get("created_at") or t.get("created_at_datetime")
        tweet_id = t.get("id") or t.get("id_str")

        ex = {
            "tweet_id": tweet_id,
            "created_at": created_at,
            "messages": [
                {"role":"user", "content": user_prompt},
                {"role":"model","content": label}
            ]
        }
        examples.append(ex)

    # de-duplicate near-identical tweets
    seen = set()
    uniq = []
    for ex in examples:
        key = re.sub(r"\W+", "", ex["messages"][1]["content"].lower())
        if key in seen: 
            continue
        seen.add(key)
        uniq.append(ex)

    # split by time (last 5–10% → eval)
    uniq.sort(key=lambda x: x.get("created_at") or "")
    n = len(uniq)
    eval_cut = max(1, int(n * 0.1))

    train = uniq[:-eval_cut] if n > 1 else uniq
    eval_ = uniq[-eval_cut:] if n > 1 else uniq

    def dump_jsonl(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(str(out), train)
    dump_jsonl(str(out.with_name(out.stem + "_eval.jsonl")), eval_)
    print(f"wrote {len(train)} train, {len(eval_)} eval -> {out} / {out.with_name(out.stem + '_eval.jsonl')}")
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("archive_path", help="Path to the unzipped Twitter archive folder or a specific file")
    ap.add_argument("--out", default="train.jsonl")
    ap.add_argument("--min_len", type=int, default=10)
    ap.add_argument("--no_quotes", action="store_true")
    args = ap.parse_args()
    convert(args.archive_path, args.out, min_len=args.min_len, include_quotes=(not args.no_quotes))

