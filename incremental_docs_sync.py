#!/usr/bin/env python3
"""
Incrementally ingest documents (txt, md, html, pdf, docx) into Gemma SFT JSONL.

Default objective = STYLE SFT:
  user:  "Write a paragraph in my signature style about: <topics> ..."
  model: <your original paragraph/section text>

Usage:
  python incremental_docs_sync.py --path docs/ --out dataset/train.jsonl --state state/docs_sync.json
  python incremental_docs_sync.py --path notes.md --out dataset/train.jsonl --dedup-dataset

Options:
  --mode style|qa_heading
    style      : (default) teach voice by using your paragraphs as targets
    qa_heading : build Q→A from headings (Q=heading, A=following paragraph)
  --min_chars / --max_chars : size bounds for chunks (defaults 80 / 1200)
  --lang_hint "..."         : prepend a language/style hint into the user prompt
  --dedup-dataset           : also dedup against existing OUT content (slower)
  --delete-missing          : prune state entries for files that no longer exist

Notes:
  • Only include documents you authored / have rights to use.
"""

import argparse, re, os, json, html, pathlib, datetime, hashlib
from typing import List, Dict, Iterable, Tuple

# -------- optional deps per format --------
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    import docx  # python-docx
except Exception:
    docx = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

SUPPORTED_EXT = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf", ".docx"}

# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="File or directory (recurses)")
    ap.add_argument("--out", default="dataset/train.jsonl", help="Output JSONL (append)")
    ap.add_argument("--state", default="state/docs_sync.json", help="State JSON file")
    ap.add_argument("--mode", choices=["style","qa_heading"], default="style")
    ap.add_argument("--min_chars", type=int, default=80)
    ap.add_argument("--max_chars", type=int, default=1200)
    ap.add_argument("--lang_hint", default="", help="e.g., 'Use Old Hebrew register.'")
    ap.add_argument("--dedup-dataset", action="store_true")
    ap.add_argument("--delete-missing", action="store_true")
    return ap.parse_args()

# ----------------- helpers -----------------
def norm_text(t: str) -> str:
    t = html.unescape(t or "")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def sha256_text(t: str) -> str:
    return hashlib.sha256(norm_text(t).encode("utf-8")).hexdigest()

def topic_keywords(t: str, limit=8) -> List[str]:
    t = re.sub(r"https?://\S+", "", t)
    words = re.findall(r"[A-Za-z\u0590-\u05FF][\w’׳״'-]{2,}", t)  # allow Hebrew
    out = []
    for w in words:
        lw = w.lower()
        if lw not in out:
            out.append(lw)
        if len(out) >= limit:
            break
    return out or ["general"]

def read_txt(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def strip_markdown(src: str) -> str:
    src = re.sub(r"```.*?```", "", src, flags=re.S)
    src = re.sub(r"`[^`]+`", "", src)
    src = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", src)
    src = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", src)
    src = re.sub(r"^\s{0,3}#{1,6}\s*", "", src, flags=re.M)
    return src

def read_md(path: pathlib.Path) -> str:
    return strip_markdown(read_txt(path))

def read_html(path: pathlib.Path) -> str:
    raw = read_txt(path)
    if not BeautifulSoup:
        return raw
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script","style","noscript"]):
        tag.decompose()
    return soup.get_text("\n")

def read_pdf(path: pathlib.Path) -> str:
    if not PdfReader:
        raise RuntimeError("pypdf not installed")
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)

def read_docx(path: pathlib.Path) -> str:
    if not docx:
        raise RuntimeError("python-docx not installed")
    d = docx.Document(str(path))
    return "\n".join(p.text for p in d.paragraphs)

def load_text(path: pathlib.Path) -> str:
    ext = path.suffix.lower()
    if ext == ".txt": return read_txt(path)
    if ext in {".md",".markdown"}: return read_md(path)
    if ext in {".html",".htm"}: return read_html(path)
    if ext == ".pdf": return read_pdf(path)
    if ext == ".docx": return read_docx(path)
    return ""

def split_paragraphs(text: str) -> List[str]:
    parts = [norm_text(p) for p in re.split(r"\n\s*\n", text) if norm_text(p)]
    return parts

def chunk_long(par: str, max_chars: int) -> List[str]:
    if len(par) <= max_chars:
        return [par]
    sents = re.split(r"(?<=[\.!?…\u05C3])\s+", par)
    out, buf = [], ""
    for s in sents:
        if len(buf) + 1 + len(s) <= max_chars:
            buf = (buf + " " + s).strip()
        else:
            if buf: out.append(buf)
            buf = s
    if buf: out.append(buf)
    return out

def iter_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    if root.is_file():
        if root.suffix.lower() in SUPPORTED_EXT:
            yield root
        return
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            yield p

def load_dataset_norms(out_path: pathlib.Path) -> set:
    if not out_path.exists():
        return set()
    norms = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                msgs = obj.get("messages", [])
                if msgs:
                    txt = msgs[-1].get("content","")
                    if txt:
                        norms.add(norm_text(txt).lower())
            except Exception:
                pass
    return norms

# ----------------- state -----------------
def load_state(path: pathlib.Path) -> Dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"version":1, "files":{}, "last_run_at": None}

def save_state(path: pathlib.Path, state: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")

# ----------------- example builders -----------------
def make_style_example(txt: str, src: pathlib.Path, chunk_ix: int, lang_hint: str) -> Dict:
    topics = topic_keywords(txt)
    hint = (lang_hint + " ").strip() if lang_hint else ""
    prompt = f"{hint}Write a paragraph in my signature style about: " + ", ".join(topics) + ". Aim for a similar length."
    mtime = datetime.datetime.fromtimestamp(src.stat().st_mtime, tz=datetime.timezone.utc).isoformat().replace("+00:00","Z")
    return {
        "source_path": str(src),
        "created_at": mtime,
        "chunk_ix": chunk_ix,
        "messages": [
            {"role":"user","content": prompt},
            {"role":"model","content": txt}
        ]
    }

def make_qa_example(heading: str, answer: str, src: pathlib.Path, chunk_ix: int, lang_hint: str) -> Dict:
    hint = (lang_hint + " ").strip() if lang_hint else ""
    question = f"{hint}{heading.strip().rstrip('?')}?"
    mtime = datetime.datetime.fromtimestamp(src.stat().st_mtime, tz=datetime.timezone.utc).isoformat().replace("+00:00","Z")
    return {
        "source_path": str(src),
        "created_at": mtime,
        "chunk_ix": chunk_ix,
        "messages": [
            {"role":"user","content": question},
            {"role":"model","content": answer.strip()}
        ]
    }

def extract_heading_pairs(text: str) -> List[Dict[str,str]]:
    lines = text.splitlines()
    items = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        is_head = bool(re.match(r"^\s{0,3}#{1,6}\s+.+", line)) or line.endswith(":")
        if is_head:
            j = i + 1
            para_lines = []
            while j < len(lines) and not lines[j].strip():
                j += 1
            while j < len(lines) and lines[j].strip():
                para_lines.append(lines[j].strip())
                j += 1
            if para_lines:
                head_txt = re.sub(r"^\s{0,3}#{1,6}\s+", "", line).rstrip(":").strip()
                ans = norm_text(" ".join(para_lines))
                items.append({"q": head_txt, "a": ans})
            i = j
        else:
            i += 1
    return items

# ----------------- main -----------------
def main():
    args = parse_args()
    root = pathlib.Path(args.path)
    outp = pathlib.Path(args.out)
    statep = pathlib.Path(args.state)
    outp.parent.mkdir(parents=True, exist_ok=True)
    state = load_state(statep)

    dataset_norms = load_dataset_norms(outp) if args.dedup_dataset else set()

    added = 0
    seen_files = set()

    with outp.open("a", encoding="utf-8") as out_f:
        for fp in iter_files(root):
            seen_files.add(str(fp))
            stat = fp.stat()
            key = str(fp.resolve())
            info = state["files"].get(key, {})
            changed = (info.get("mtime") != stat.st_mtime) or (info.get("size") != stat.st_size)

            # Prepare per-file chunk hash set (existing)
            known_hashes = set(info.get("chunks", []))

            if not changed and known_hashes:
                # Nothing changed on disk; skip
                continue

            # Read text and build chunks
            try:
                raw = load_text(fp)
            except Exception as e:
                print(f"[warn] failed to read {fp}: {e}")
                continue
            text = norm_text(raw)
            if not text:
                # record empty to avoid re-trying
                state["files"][key] = {"mtime": stat.st_mtime, "size": stat.st_size, "chunks": []}
                continue

            chunk_ix = 0
            new_hashes = set(known_hashes)

            if args.mode == "qa_heading":
                pairs = extract_heading_pairs(text)
                for k, pair in enumerate(pairs):
                    ans = pair["a"]
                    if len(ans) < args.min_chars: 
                        continue
                    if len(ans) > args.max_chars:
                        for j, chunk in enumerate(chunk_long(ans, args.max_chars)):
                            if len(chunk) < args.min_chars: 
                                continue
                            h = sha256_text(chunk)
                            if h in new_hashes or (dataset_norms and norm_text(chunk).lower() in dataset_norms):
                                continue
                            ex = make_qa_example(pair["q"], chunk, fp, k*10+j, args.lang_hint)
                            out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                            added += 1
                            new_hashes.add(h)
                            dataset_norms.add(norm_text(chunk).lower())
                    else:
                        h = sha256_text(ans)
                        if h in new_hashes or (dataset_norms and norm_text(ans).lower() in dataset_norms):
                            continue
                        ex = make_qa_example(pair["q"], ans, fp, k, args.lang_hint)
                        out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                        added += 1
                        new_hashes.add(h)
                        dataset_norms.add(norm_text(ans).lower())
            else:
                # style mode
                for p_ix, par in enumerate(split_paragraphs(text)):
                    if len(par) < args.min_chars: 
                        continue
                    for j, ch in enumerate(chunk_long(par, args.max_chars)):
                        if len(ch) < args.min_chars: 
                            continue
                        h = sha256_text(ch)
                        if h in new_hashes or (dataset_norms and norm_text(ch).lower() in dataset_norms):
                            continue
                        ex = make_style_example(ch, fp, p_ix*10+j, args.lang_hint)
                        out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                        added += 1
                        new_hashes.add(h)
                        dataset_norms.add(norm_text(ch).lower())

            # Update state for this file
            state["files"][key] = {"mtime": stat.st_mtime, "size": stat.st_size, "chunks": sorted(new_hashes)}

    # Optionally clean missing files from state
    if args.delete_missing:
        keys = set(state["files"].keys())
        for k in keys - set(seen_files):
            del state["files"][k]

    state["last_run_at"] = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    statep.parent.mkdir(parents=True, exist_ok=True)
    save_state(statep, state)
    print(f"Appended {added} examples to {outp}")

if __name__ == "__main__":
    main()

