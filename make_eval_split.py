#!/usr/bin/env python3
import argparse, json, pathlib, random
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

def parse_ts(s: Optional[str]) -> Optional[datetime]:
    """Parse many timestamp shapes; always return tz-aware (UTC) datetimes."""
    if not s:
        return None
    s = s.strip().replace("Z", "+00:00")
    fmts = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%a %b %d %H:%M:%S %z %Y",  # Twitter-ish fallback
    )
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            pass
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None

def to_epoch(dt: datetime) -> float:
    """Convert any datetime to epoch seconds; treat naive as UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()

def load_jsonl(p: pathlib.Path) -> List[Dict[str, Any]]:
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

def save_jsonl(p: pathlib.Path, rows: List[Dict[str,Any]]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Split a JSONL dataset into train/eval.")
    # input aliases
    ap.add_argument("--in", "--input", "-i", dest="inp", required=True, help="input JSONL (merged)")
    # outputs (aliases with hyphen/underscore)
    ap.add_argument("--out_train", "--out-train", dest="out_train", default="dataset/train.jsonl")
    ap.add_argument("--out_eval",  "--out-eval",  dest="out_eval",  default="dataset/train_eval.jsonl")
    # pct, strategy, aliases
    ap.add_argument("--eval_pct", "--eval-pct", dest="eval_pct", type=float, default=0.10)
    ap.add_argument("--strategy", choices=["auto","time","random"], default="auto",
                    help="auto: time if timestamps exist, else random")
    ap.add_argument("--time-order",  dest="strategy", action="store_const", const="time",
                    help="alias for --strategy time")
    ap.add_argument("--random-order",dest="strategy", action="store_const", const="random",
                    help="alias for --strategy random")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rows = load_jsonl(pathlib.Path(args.inp))
    if not rows:
        print(f"[split] no rows in {args.inp}"); return

    has_time = any(r.get("created_at") for r in rows)
    strat = "time" if (args.strategy=="time" or (args.strategy=="auto" and has_time)) else "random"
    n_eval = max(1, int(round(len(rows) * args.eval_pct)))
    if len(rows) > 1:
        n_eval = min(n_eval, len(rows) - 1)

    if strat == "time" and has_time:
        rows_with_ts, rows_no_ts = [], []
        for r in rows:
            ts = parse_ts(r.get("created_at"))
            if ts:
                rows_with_ts.append((to_epoch(ts), r))
            else:
                rows_no_ts.append(r)
        rows_with_ts.sort(key=lambda x: x[0])
        ordered = [r for _, r in rows_with_ts] + rows_no_ts
        eval_set = ordered[-n_eval:]
        train_set = ordered[:-n_eval]
    else:
        rnd = random.Random(args.seed)
        idx = list(range(len(rows)))
        rnd.shuffle(idx)
        eval_ix = set(idx[:n_eval])
        eval_set = [rows[i] for i in idx[:n_eval]]
        train_set = [rows[i] for i in range(len(rows)) if i not in eval_ix]

    save_jsonl(pathlib.Path(args.out_train), train_set)
    save_jsonl(pathlib.Path(args.out_eval),  eval_set)
    print(f"[split] strategy={strat} total={len(rows)} train={len(train_set)} eval={len(eval_set)}")
    print(f"[split] wrote: {args.out_train} and {args.out_eval}")

if __name__ == "__main__":
    main()

