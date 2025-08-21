#!/usr/bin/env python3
import argparse, sqlite3, re, html, json, pathlib, sys
from collections import defaultdict

def html_to_text(s: str) -> str:
    if not s:
        return ""
    # normalize paragraph/line breaks
    s = s.replace("<br />", "\n").replace("<br/>", "\n").replace("<br>", "\n")
    s = re.sub(r"</p>\s*<p>", "\n\n", s, flags=re.I)
    s = re.sub(r"</?p[^>]*>", "", s, flags=re.I)
    # drop all other tags
    s = re.sub(r"<[^>]+>", "", s)
    # unescape SQL-escaped slashes and HTML entities
    s = s.replace("\\/", "/")
    s = html.unescape(s)
    # collapse whitespace
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

def title_to_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s).strip()

def compose_content(title: str, body: str) -> str:
    t = title_to_text(title or "")
    b = html_to_text(body or "")
    return (t + ("\n\n" if t and b else "") + b) if (t or b) else ""

def connect_input(path: pathlib.Path) -> sqlite3.Connection:
    con = None
    if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        con = sqlite3.connect(str(path))
    else:
        # treat as .sql dump
        con = sqlite3.connect(":memory:")
        sql = path.read_text(encoding="utf-8", errors="ignore")
        con.executescript(sql)
    con.row_factory = sqlite3.Row
    return con

def ancestors(by_id, row):
    """Return list [root ... this_row] following parent_id chain."""
    chain = []
    cur = row
    seen = set()
    while cur and cur["id"] not in seen:
        chain.append(cur)
        seen.add(cur["id"])
        pid = cur["parent_id"]
        if not pid or pid == 0 or pid == cur["id"]:
            break
        cur = by_id.get(pid)
    chain.reverse()
    return chain

def main():
    ap = argparse.ArgumentParser(description="Convert forum SQL to Gemma SFT JSONL with thread context.")
    ap.add_argument("--input", required=True, help=".sql dump OR .db/.sqlite")
    ap.add_argument("--out", default="dataset/train.jsonl", help="output JSONL")
    ap.add_argument("--nick", required=True, help="your author nickname (e.g., 'NatiHatuka')")
    ap.add_argument("--max_context", type=int, default=8, help="max prior turns before your reply")
    ap.add_argument("--strip_self_context", action="store_true",
                    help="drop your earlier replies from context (keep only others)")
    ap.add_argument("--role_assistant", default="model", choices=["model","assistant"],
                    help="role name for your replies (Gemma often uses 'model')")
    args = ap.parse_args()

    inp = pathlib.Path(args.input)
    con = connect_input(inp)

    rows = list(con.execute("SELECT * FROM msg_tbl ORDER BY datetime(date) ASC, id ASC"))
    by_id = {r["id"]: r for r in rows}

    # Build examples: one per *your* reply
    me = args.nick
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with outp.open("a", encoding="utf-8") as f:
        for r in rows:
            # We emit only your replies; parents you authored are rare, but skip if type=='parent'
            if (r["nick"] or "") != me:
                continue
            if (r["type"] or "").lower() == "parent":
                continue

            chain = ancestors(by_id, r)
            if not chain:
                continue

            # Build messages list up to and including *this* reply
            # Optionally cap context to last K turns before final
            ctx = chain[:-1]
            if args.strip_self_context:
                ctx = [x for x in ctx if (x["nick"] or "") != me]

            if args.max_context and len(ctx) > args.max_context:
                ctx = ctx[-args.max_context:]

            msgs = []
            for m in ctx:
                role = args.role_assistant if (m["nick"] or "") == me else "user"
                content = compose_content(m["title"], m["body"])
                if not content:
                    continue
                msgs.append({"role": role, "content": content})

            # final assistant/model message = your reply
            final = chain[-1]
            content = compose_content(final["title"], final["body"])
            if not content:
                continue
            msgs.append({"role": args.role_assistant, "content": content})

            # metadata
            ex = {
                "thread_id": final["root_id"],
                "post_id": final["id"],
                "created_at": final["date"],
                "author": final["nick"],
                "messages": msgs
            }
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} samples to {outp}")

if __name__ == "__main__":
    main()

