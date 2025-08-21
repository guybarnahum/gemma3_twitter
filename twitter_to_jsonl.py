#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a Twitter/X archive (unzipped folder or .js/.json/.jsonl file)
into Gemma-style SFT JSONL with optional progress bars.

Features:
- Skips retweets; optionally skip quote tweets.
- Optional include of replies.
- Optional dialog mode: if a reply's parent tweet exists in the archive,
  build a short context (parent -> your reply).
- De-duplicates near-identical tweets.
- Optional eval split (time-based) or --no-eval to defer until after unify.
- Progress bars (tqdm) for normalize, filter+dedup, build, and write.

Examples:
  python twitter_to_jsonl.py ~/Downloads/twitter-archive --out dataset/tweets_base.jsonl --no-eval
  python twitter_to_jsonl.py data/tweets.js --out dataset/tweets_base.jsonl --eval-pct 0.05
"""

import argparse, json, pathlib, re, html, hashlib, sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

# ---------- tqdm wrapper ----------

def get_progress_iter(disable: bool):
    """
    Returns a function pbar(iterable, total, desc) that yields either a tqdm
    iterator (if available and not disabled) or the plain iterable.
    """
    if disable:
        def _plain(iterable, total=None, desc=None):  # pylint: disable=unused-argument
            return iterable
        return _plain
    try:
        import tqdm  # type: ignore
        def _tqdm(iterable, total=None, desc=None):
            return tqdm.tqdm(iterable, total=total, desc=desc, unit="it", dynamic_ncols=True)
        return _tqdm
    except Exception:
        # tqdm not installed → no progress
        def _plain(iterable, total=None, desc=None):  # pylint: disable=unused-argument
            return iterable
        return _plain

# ---------- helpers ----------

TW_TIME_FMTS = (
    "%a %b %d %H:%M:%S %z %Y",   # Mon Jan 01 00:00:00 +0000 2024
    "%Y-%m-%d %H:%M:%S%z",       # 2024-01-01 00:00:00+00:00
    "%Y-%m-%d %H:%M:%S",         # 2024-01-01 00:00:00
    "%Y-%m-%dT%H:%M:%S%z",       # 2024-01-01T00:00:00+00:00
    "%Y-%m-%dT%H:%M:%S",         # 2024-01-01T00:00:00
    "%Y-%m-%d",                  # 2024-01-01
)

def parse_ts(s: Optional[str]) -> Optional[datetime]:
    if not s: return None
    s = s.strip().replace("Z", "+00:00")
    for fmt in TW_TIME_FMTS:
        try:
            dt = datetime.strptime(s, fmt)
            if not dt.tzinfo:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            pass
    try:
        dt = datetime.fromisoformat(s)
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None

TAG_RE = re.compile(r"<[^>]+>")
A_TEXT_RE = re.compile(r">([^<]+)<")   # text inside an <a> tag

URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")

def clean_text(t: str) -> str:
    if not t: return ""
    t = html.unescape(t)
    t = t.replace("\u200f", "").replace("\u200e", "")  # RTL marks that can confuse rendering
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = WS_RE.sub(" ", t).strip()
    return t

def norm_for_dedup(t: str) -> str:
    t = html.unescape(t)
    t = URL_RE.sub("", t)                   # strip URLs
    t = re.sub(r"[#@]\w+", "", t)           # strip simple hashtags/mentions
    t = WS_RE.sub(" ", t).strip().lower()
    return t

def parse_source_app(src_html: Optional[str]) -> str:
    if not src_html: return ""
    # typical: <a href="https://mobile.twitter.com" rel="nofollow">Twitter Web App</a>
    m = A_TEXT_RE.search(src_html)
    return clean_text(m.group(1)) if m else clean_text(TAG_RE.sub("", src_html))

# ---------- load ----------

def load_js(path: pathlib.Path) -> List[Dict[str, Any]]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    # Strip "window.YTD.tweets.part0 = " wrapper if present
    i = txt.find("[")
    j = txt.rfind("]")
    payload = txt[i:j+1] if (i != -1 and j != -1 and j > i) else txt
    data = json.loads(payload)
    out = []
    for it in data:
        out.append(it["tweet"] if "tweet" in it else it)
    return out

def load_json(path: pathlib.Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    return [x["tweet"] if "tweet" in x else x for x in (data if isinstance(data, list) else [data])]

def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            out.append(obj["tweet"] if "tweet" in obj else obj)
    return out

def load_archive(path: pathlib.Path) -> List[Dict[str, Any]]:
    if path.is_file():
        ext = path.suffix.lower()
        if ext == ".js":
            return load_js(path)
        elif ext == ".json":
            return load_json(path)
        elif ext in (".jsonl", ".ndjson"):
            return load_jsonl(path)
        else:
            raise SystemExit(f"Unsupported file type: {path}")
    # directory: look for data/tweets.js or data/tweets/*.js
    candidates = []
    p1 = path / "data" / "tweets.js"
    if p1.exists():
        candidates.append(p1)
    for p in (path / "data" / "tweets").glob("*.js"):
        candidates.append(p)
    if not candidates:
        raise SystemExit(f"No tweets.js found under: {path}")
    tweets = []
    for p in sorted(candidates):
        tweets.extend(load_js(p))
    return tweets

# ---------- unify tweet fields ----------

def unify_tweet(t: Dict[str, Any]) -> Dict[str, Any]:
    id_str = t.get("id_str") or str(t.get("id") or "")
    text = t.get("full_text") or t.get("text") or ""
    text = clean_text(text)
    created_at = t.get("created_at")
    lang = t.get("lang") or ""
    src_app = parse_source_app(t.get("source"))
    in_reply_to = t.get("in_reply_to_status_id_str") or (str(t.get("in_reply_to_status_id")) if t.get("in_reply_to_status_id") else "")
    is_quote = bool(t.get("is_quote_status") or t.get("quoted_status_id") or t.get("quoted_status_id_str"))
    is_retweet = bool(t.get("retweeted") or t.get("retweeted_status_id") or t.get("retweeted_status") or text.startswith("RT @"))
    parent_text = ""
    if t.get("quoted_status") and isinstance(t["quoted_status"], dict):
        q = t["quoted_status"]
        parent_text = clean_text(q.get("full_text") or q.get("text") or "")
    return {
        "id_str": id_str,
        "text": text,
        "created_at": created_at,
        "ts": parse_ts(created_at),
        "lang": lang,
        "source_app": src_app,
        "in_reply_to_id": in_reply_to,
        "is_quote": is_quote,
        "is_retweet": is_retweet,
        "parent_text": parent_text,
    }

# ---------- SFT builders ----------

def make_style_example(tweet: Dict[str, Any], prompt: str, role_assistant: str) -> Dict[str, Any]:
    return {
        "tweet_id": tweet["id_str"],
        "created_at": tweet["created_at"],
        "lang": tweet["lang"],
        "source_app": tweet["source_app"],
        "messages": [
            {"role": "user", "content": prompt},
            {"role": role_assistant, "content": tweet["text"]},
        ],
    }

def make_dialog_example(tweet: Dict[str, Any], by_id: Dict[str, Dict[str, Any]], max_context: int, role_assistant: str) -> Optional[Dict[str, Any]]:
    chain: List[Dict[str, Any]] = []
    cur = tweet
    steps = 0
    while cur and steps < max_context:
        pid = cur.get("in_reply_to_id")
        if not pid: break
        parent = by_id.get(pid)
        if not parent: break
        chain.append(parent)
        cur = parent
        steps += 1
    chain.reverse()

    messages = []
    for p in chain:
        if p["text"]:
            messages.append({"role": "user", "content": p["text"]})
    messages.append({"role": role_assistant, "content": tweet["text"]})
    if len(messages) < 2:
        return None
    return {
        "tweet_id": tweet["id_str"],
        "created_at": tweet["created_at"],
        "lang": tweet["lang"],
        "source_app": tweet["source_app"],
        "messages": messages,
    }

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Twitter/X archive → Gemma SFT JSONL")
    ap.add_argument("archive", help="Path to unzipped archive folder or a .js/.json/.jsonl file")
    ap.add_argument("--out", required=True, help="Output train JSONL path")
    ap.add_argument("--no-eval", action="store_true", help="Do not write an eval split")
    ap.add_argument("--eval-pct", type=float, default=0.10, help="Eval fraction (ignored with --no-eval)")
    ap.add_argument("--eval-out", default=None, help="Eval output path (default: <OUT> with _eval.jsonl)")
    ap.add_argument("--include-replies", action="store_true", help="Include replies (in_reply_to*)")
    ap.add_argument("--no-quotes", action="store_true", help="Exclude quote tweets")
    ap.add_argument("--exclude-sources", default="", help="Comma/space-separated app names to skip (e.g., 'MyBotApp,AnotherApp')")
    ap.add_argument("--dialog", action="store_true", help="If parent exists in archive, build a short dialog (parent → your reply)")
    ap.add_argument("--max-dialog-context", type=int, default=1, help="Max parent hops to include in dialog mode")
    ap.add_argument("--prompt", default="Write a concise tweet in my signature style. Keep it under 280 characters.",
                    help="User prompt for style SFT")
    ap.add_argument("--role-assistant", default="model", choices=["model","assistant"], help="Role label for target replies")
    ap.add_argument("--min-chars", type=int, default=5, help="Drop tweets shorter than this many chars after cleaning")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    args = ap.parse_args()

    pbar = get_progress_iter(disable=args.no_progress)

    src = pathlib.Path(args.archive)
    raw = load_archive(src)

    # normalize / unify
    unified: List[Dict[str, Any]] = []
    for t in pbar(raw, total=len(raw), desc="normalize"):
        unified.append(unify_tweet(t))

    # index by id for dialog mode
    by_id = {t["id_str"]: t for t in unified if t["id_str"]}

    # parse excludes
    excl = set()
    if args.exclude_sources.strip():
        for tok in re.split(r"[,\s]+", args.exclude_sources.strip()):
            if tok:
                excl.add(tok.strip())

    # filter + dedup
    kept: List[Dict[str, Any]] = []
    seen_hashes = set()
    for tw in pbar(unified, total=len(unified), desc="filter+dedup"):
        if tw["is_retweet"]:
            continue
        if args.no_quotes and tw["is_quote"]:
            continue
        if (not args.include_replies) and tw["in_reply_to_id"]:
            continue
        if tw["source_app"] and tw["source_app"] in excl:
            continue
        if not tw["text"] or len(clean_text(tw["text"])) < args.min_chars:
            continue

        key = hashlib.sha256(norm_for_dedup(tw["text"]).encode("utf-8")).hexdigest()
        if key in seen_hashes:
            continue
        seen_hashes.add(key)
        kept.append(tw)

    # order by time (oldest→newest), unknown timestamps first
    kept.sort(key=lambda x: (x["ts"] is None, x["ts"] or datetime(1970,1,1,tzinfo=timezone.utc)))

    # build SFT rows
    rows: List[Dict[str, Any]] = []
    for tw in pbar(kept, total=len(kept), desc="build examples"):
        if args.dialog:
            ex = make_dialog_example(tw, by_id, args.max_dialog_context, args.role_assistant)
            if ex:
                rows.append(ex)
                continue
        rows.append(make_style_example(tw, args.prompt, args.role_assistant))

    if not rows:
        print("No tweets to write (after filters).")
        return

    # split
    eval_pct = 0.0 if args.no_eval else max(0.0, min(1.0, args.eval_pct))
    n_eval = int(round(len(rows) * eval_pct))
    train_rows, eval_rows = rows, []
    if n_eval > 0:
        eval_rows = rows[-n_eval:]
        train_rows = rows[:-n_eval]

    # write
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with outp.open("w", encoding="utf-8") as f:
        for r in pbar(train_rows, total=len(train_rows), desc="write train"):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    msg = f"wrote {len(train_rows)} train"
    if eval_rows:
        eval_path = pathlib.Path(args.eval_out) if args.eval_out else outp.with_name(outp.stem + "_eval.jsonl")
        with eval_path.open("w", encoding="utf-8") as f:
            for r in pbar(eval_rows, total=len(eval_rows), desc="write eval"):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        msg += f", {len(eval_rows)} eval -> {str(outp)} / {str(eval_path)}"
    else:
        msg += f" -> {str(outp)} (no eval)"
    print(msg)

if __name__ == "__main__":
    main()

