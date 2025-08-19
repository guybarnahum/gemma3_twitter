#!/usr/bin/env python3
"""
Incrementally pull new tweets since last sync, convert to Gemma SFT JSONL, and append.

Usage:
  python incremental_sync.py --username your_handle \
      --out dataset/train.jsonl \
      --state state/last_sync.json \
      --bearer $TWITTER_BEARER_TOKEN \
      --exclude-sources "MyBotApp" \
      --generated-registry generated_tweets.jsonl \
      --no-quotes \
      --include-replies

Notes:
- Keep your eval split frozen. This script only appends to train.jsonl.
- Requires: requests (pip install requests)
"""

import argparse, os, json, re, html, sys, pathlib, datetime, time
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher

import requests

API = "https://api.twitter.com/2"

def iso_now() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", required=True, help="Twitter/X handle without @")
    ap.add_argument("--bearer", default=os.getenv("TWITTER_BEARER_TOKEN"), help="Twitter API v2 Bearer token")
    ap.add_argument("--out", default="dataset/train.jsonl", help="Output JSONL to append to")
    ap.add_argument("--state", default="state/last_sync.json", help="State file storing last sync cursor")
    ap.add_argument("--generated-registry", default=None, help="JSONL file of model-generated tweets (ids or texts)")
    ap.add_argument("--exclude-sources", default="", help="Comma/space-separated app names to skip if tweet.source matches")
    ap.add_argument("--min-len", type=int, default=10, help="Minimum tweet text length")
    ap.add_argument("--no-quotes", action="store_true", help="Exclude quote tweets")
    ap.add_argument("--include-replies", action="store_true", help="Include replies (good for dialog SFT)")
    ap.add_argument("--dry-run", action="store_true", help="Fetch/convert but do not write")
    return ap.parse_args()

# ---------- Helpers: loading cursors, existing dataset, registry ----------

def load_state(state_path: str) -> Dict[str, Any]:
    p = pathlib.Path(state_path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}

def save_state(state_path: str, state: Dict[str, Any]):
    p = pathlib.Path(state_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")

def normalize_text(t: str) -> str:
    t = html.unescape(t or "")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"(https?://\S+?)(\?|&)(utm_[^=\s]+=\S+)", r"\1", t)  # strip tracking
    return t

def scan_dataset_for_existing(out_path: str):
    """Return (max_created_at, existing_ids, existing_norm_hash)."""
    max_dt = None
    ids = set()
    texts = set()  # normalized
    p = pathlib.Path(out_path)
    if not p.exists():
        return (None, ids, texts)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            tid = obj.get("tweet_id")
            if tid: ids.add(str(tid))
            created = obj.get("created_at")
            if created:
                try:
                    dt = datetime.datetime.fromisoformat(created.replace("Z","+00:00"))
                    if (max_dt is None) or (dt > max_dt):
                        max_dt = dt
                except Exception:
                    pass
            # also build text signature for crude dedup
            try:
                msgs = obj["messages"]
                if msgs and isinstance(msgs, list) and len(msgs) >= 2:
                    label = msgs[-1].get("content","")
                    if label:
                        texts.add(normalize_text(label).lower())
            except Exception:
                pass
    return (max_dt, ids, texts)

def load_generated_registry(registry_path: Optional[str]):
    """Load model-generated tweets (ids or texts) to filter out."""
    ids, texts = set(), set()
    if not registry_path:
        return ids, texts
    p = pathlib.Path(registry_path)
    if not p.exists():
        return ids, texts
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                # maybe raw text line
                obj = {"text": line}
            tid = obj.get("tweet_id") or obj.get("id") or obj.get("id_str")
            if tid: ids.add(str(tid))
            txt = obj.get("text") or obj.get("tweet") or ""
            if txt:
                texts.add(normalize_text(txt).lower())
    return ids, texts

def looks_like_generated(text: str, gen_texts: set, threshold=0.98) -> bool:
    """Near-dup match vs registry with a conservative ratio."""
    norm = normalize_text(text).lower()
    if norm in gen_texts:
        return True
    # cheap pass: exact words length filter
    for cand in gen_texts:
        if abs(len(cand) - len(norm)) > 40:
            continue
        if SequenceMatcher(None, cand, norm).ratio() >= threshold:
            return True
    return False

# ---------- Twitter API ----------

def get_user_id(username: str, bearer: str) -> str:
    r = requests.get(
        f"{API}/users/by/username/{username}",
        headers={"Authorization": f"Bearer {bearer}"},
        params={"user.fields":"id"}
    )
    r.raise_for_status()
    return r.json()["data"]["id"]

def fetch_tweets_since(user_id: str, start_time_iso: str, bearer: str, include_replies: bool):
    """
    Generator yielding tweet dicts (newest->oldest order of each page).
    """
    params = {
        "max_results": 100,
        "start_time": start_time_iso,
        "tweet.fields": "id,text,created_at,entities,source,referenced_tweets,conversation_id,lang",
        # We do NOT set end_time; default = now
    }
    # Exclude RTs; optionally exclude replies
    exclude = ["retweets"]
    if not include_replies:
        exclude.append("replies")
    params["exclude"] = ",".join(exclude)

    next_token = None
    while True:
        if next_token:
            params["pagination_token"] = next_token
        r = requests.get(
            f"{API}/users/{user_id}/tweets",
            headers={"Authorization": f"Bearer {bearer}"},
            params=params,
            timeout=30
        )
        if r.status_code == 429:
            reset = int(r.headers.get("x-rate-limit-reset", "0"))
            wait = max(5, reset - int(time.time()))
            print(f"Rate limited. Sleeping {wait}s...", file=sys.stderr)
            time.sleep(wait)
            continue
        r.raise_for_status()
        data = r.json()
        for t in data.get("data", []):
            yield t
        next_token = data.get("meta", {}).get("next_token")
        if not next_token:
            break

# ---------- Conversion ----------

def is_retweet(t: Dict[str,Any]) -> bool:
    for ref in t.get("referenced_tweets", []) or []:
        if ref.get("type") == "retweeted":
            return True
    # fallback by text prefix
    return (t.get("text","").startswith("RT @"))

def is_quote(t: Dict[str,Any]) -> bool:
    for ref in t.get("referenced_tweets", []) or []:
        if ref.get("type") == "quoted":
            return True
    return False

def extract_topics(text: str, entities: Dict[str,Any]) -> List[str]:
    topics = set()
    for h in (entities or {}).get("hashtags", []):
        tag = h.get("tag") or h.get("text") or ""
        if tag: topics.add("#" + tag)
    t = re.sub(r"https?://\S+", "", text)
    t = re.sub(r"[@#]\w+", "", t)
    words = re.findall(r"[A-Za-z][A-Za-z0-9’'-]{2,}", t)
    for w in words[:6]:
        topics.add(w.lower())
    return list(topics)[:6] or ["general"]

def to_example(t: Dict[str,Any], min_len=10) -> Optional[Dict[str,Any]]:
    txt = normalize_text(t.get("text",""))
    if len(txt) < min_len:
        return None
    topics = extract_topics(txt, t.get("entities") or {})
    user_prompt = (
        "Write a concise tweet in my signature style about: "
        + ", ".join(topics)
        + ". Keep it under 280 characters. Use my tone and phrasing."
    )
    return {
        "tweet_id": t.get("id"),
        "created_at": t.get("created_at"),
        "source": t.get("source"),
        "messages": [
            {"role":"user","content": user_prompt},
            {"role":"model","content": txt}
        ]
    }

# ---------- Main ----------

def main():
    args = parse_args()
    if not args.bearer:
        print("Missing Bearer token. Provide --bearer or set TWITTER_BEARER_TOKEN.", file=sys.stderr)
        sys.exit(1)

    pathlib.Path("state").mkdir(exist_ok=True)
    pathlib.Path("dataset").mkdir(exist_ok=True)

    exclude_sources = set([s.strip() for s in re.split(r"[,\s]+", args.exclude_sources) if s.strip()])

    # 1) Determine start_time
    state = load_state(args.state)
    start_time = state.get("start_time")
    if not start_time:
        # fall back to scanning existing dataset by created_at
        max_dt, existing_ids, existing_texts = scan_dataset_for_existing(args.out)
        if max_dt:
            start_time = max_dt.isoformat().replace("+00:00", "Z")
        else:
            start_time = "1970-01-01T00:00:00Z"
    else:
        _, existing_ids, existing_texts = scan_dataset_for_existing(args.out)

    gen_ids, gen_texts = load_generated_registry(args.generated_registry)

    print(f"Start time: {start_time}")
    print(f"Loaded {len(existing_ids)} existing ids, {len(existing_texts)} existing texts.")
    if args.generated_registry:
        print(f"Loaded generated-registry: {len(gen_ids)} ids, {len(gen_texts)} texts.")
    if exclude_sources:
        print(f"Excluding sources: {', '.join(sorted(exclude_sources))}")

    # 2) Resolve user id
    user_id = get_user_id(args.username, args.bearer)
    print(f"User @{args.username} -> id {user_id}")

    # 3) Fetch & convert
    new_examples = []
    newest_time = start_time
    fetched = 0
    kept = 0

    for t in fetch_tweets_since(user_id, start_time, args.bearer, include_replies=args.include_replies):
        fetched += 1

        # basic filters
        if is_retweet(t): 
            continue
        if args.no_quotes and is_quote(t):
            continue

        txt = normalize_text(t.get("text",""))
        tid = str(t.get("id",""))
        src = (t.get("source") or "").strip()

        # skip generated by your app
        if src and src in exclude_sources:
            continue
        if tid and tid in gen_ids:
            continue
        if txt and looks_like_generated(txt, gen_texts):
            continue

        # dedup vs dataset
        if tid and tid in existing_ids:
            continue
        if txt and (txt.lower() in existing_texts):
            continue

        ex = to_example(t, min_len=args.min_len)
        if not ex:
            continue

        new_examples.append(ex)
        kept += 1

        # track newest created_at
        ca = t.get("created_at")
        if ca and ca > newest_time:
            newest_time = ca

    print(f"Fetched {fetched} tweets; appending {kept} new examples.")

    if args.dry_run:
        print("(dry-run) Not writing outputs.")
        return

    # 4) Append to train.jsonl
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("a", encoding="utf-8") as f:
        for ex in new_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # 5) Update state (next start_time one second after newest)
    try:
        dt = datetime.datetime.fromisoformat(newest_time.replace("Z","+00:00"))
        dt_next = (dt + datetime.timedelta(seconds=1)).isoformat().replace("+00:00","Z")
    except Exception:
        dt_next = newest_time

    state.update({
        "start_time": dt_next,
        "last_run_at": iso_now(),
        "last_appended": len(new_examples)
    })
    save_state(args.state, state)
    print(f"Updated state at {args.state}. Next start_time={dt_next}")

if __name__ == "__main__":
    main()

