#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incrementally ingest documents (.txt, .md/.markdown, .html/.htm, .docx) into
Gemma-style SFT JSONL.

Default objective = STYLE SFT:
  user : "Write a paragraph in my signature style about: <topics> ..."
  model: <your original paragraph/section text>

Usage:
  python incremental_docs_sync.py --path docs/ --out dataset/train.jsonl --state state/docs_sync.json
  python incremental_docs_sync.py --path notes.md --out dataset/train.jsonl --dedup-dataset

Options:
  --mode style|qa_heading
      style      : (default) teach voice by using your paragraphs as targets
      qa_heading : build Q→A from headings (Q = heading, A = following paragraph)
  --min_chars / --max_chars : size bounds for chunks (defaults 80 / 1200)
  --lang_hint "..."         : prefix inside the user prompt (e.g., "ענה בעברית. ")
  --prompt_lang en|he       : choose English (default) or Hebrew prompt templates
  --tag_lang "he"           : tag rows with a language code in metadata
  --dedup-dataset           : also dedup against existing OUT content (slower)
  --delete-missing          : prune state entries for files that no longer exist

Notes:
  • Only include documents you authored / have rights to use.
"""

import argparse
import datetime
import hashlib
import html
import json
import pathlib
import re
from typing import Dict, Iterable, List

# -------- optional deps per format (HTML/DOCX) --------
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None

try:
    import docx  # python-docx  # type: ignore
except Exception:  # pragma: no cover
    docx = None

SUPPORTED_EXT = {".txt", ".md", ".markdown", ".html", ".htm", ".docx"}


# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Incrementally turn docs into Gemma SFT JSONL")
    ap.add_argument("--path", required=True, help="File or directory (recurses)")
    ap.add_argument("--out", default="dataset/train.jsonl", help="Output JSONL (append)")
    ap.add_argument("--state", default="state/docs_sync.json", help="State JSON file")
    ap.add_argument("--mode", choices=["style", "qa_heading"], default="style")
    ap.add_argument("--min_chars", type=int, default=80)
    ap.add_argument("--max_chars", type=int, default=1200)
    ap.add_argument("--lang_hint", default="", help='Prefix inside prompts, e.g., "ענה בעברית. " (note trailing space).')
    ap.add_argument("--prompt_lang", choices=["en", "he"], default="en", help="Prompt language/template")
    ap.add_argument("--tag_lang", default="", help='If set (e.g., "he"), add {"lang": "..."} to each row')
    ap.add_argument("--dedup-dataset", action="store_true")
    ap.add_argument("--delete-missing", action="store_true")
    return ap.parse_args()


# ----------------- helpers -----------------
def norm_ws(s: str) -> str:
    s = (s or "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(t or "")).strip()


def sha256_text(t: str) -> str:
    return hashlib.sha256(norm_text(t).encode("utf-8")).hexdigest()


def topic_keywords(t: str, limit=8) -> List[str]:
    t = re.sub(r"https?://\S+", "", t)
    # allow Hebrew letters too
    words = re.findall(r"[A-Za-z\u0590-\u05FF][\w’׳״'-]{2,}", t)
    out: List[str] = []
    for w in words:
        lw = w.lower()
        if lw not in out:
            out.append(lw)
        if len(out) >= limit:
            break
    return out or ["general"]


# ---------- readers ----------
def read_txt(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # common Hebrew/Windows fallback
        return path.read_text(encoding="cp1255", errors="ignore")


_MD_CODE_BLOCK = re.compile(r"```.*?```", re.S)
_MD_INLINE_CODE = re.compile(r"`[^`]+`")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def strip_markdown(md: str, preserve_headers: bool) -> str:
    # remove code/content noise; optionally keep '#' headings for qa mode
    s = _MD_CODE_BLOCK.sub("", md)
    s = _MD_INLINE_CODE.sub("", s)
    s = _MD_IMAGE.sub("", s)
    s = _MD_LINK.sub(r"\1", s)
    if not preserve_headers:
        s = re.sub(r"^\s{0,3}#{1,6}\s*", "", s, flags=re.M)
    return s


def read_md(path: pathlib.Path, preserve_headers: bool) -> str:
    raw = read_txt(path)
    return strip_markdown(raw, preserve_headers=preserve_headers)


def read_html(path: pathlib.Path) -> str:
    raw = read_txt(path)
    if not BeautifulSoup:
        # naive fallback: strip tags crudely
        return norm_ws(re.sub(r"<[^>]+>", " ", raw))
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text("\n")


def read_docx(path: pathlib.Path) -> str:
    if not docx:
        raise RuntimeError("python-docx not installed")
    d = docx.Document(path.as_posix())
    return "\n".join(p.text for p in d.paragraphs)


def load_text(path: pathlib.Path, want_headings: bool) -> str:
    ext = path.suffix.lower()
    if ext == ".txt":
        return read_txt(path)
    if ext in {".md", ".markdown"}:
        return read_md(path, preserve_headers=want_headings)
    if ext in {".html", ".htm"}:
        return read_html(path)
    if ext == ".docx":
        try:
            return read_docx(path)
        except Exception:
            return ""
    return ""


# ---------- chunking ----------
def split_paragraphs(text: str) -> List[str]:
    parts = [norm_ws(p) for p in re.split(r"\n\s*\n", text)]
    return [p for p in parts if p]


def chunk_long(par: str, max_chars: int) -> List[str]:
    if len(par) <= max_chars:
        return [par]
    # split on sentence boundaries (incl. Hebrew sof pasuq U+05C3)
    sents = re.split(r"(?<=[\.!?…\u05C3])\s+", par)
    out: List[str] = []
    buf = ""
    for s in sents:
        if len(buf) + (1 if buf else 0) + len(s) <= max_chars:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                out.append(buf)
            buf = s
    if buf:
        out.append(buf)
    return out


# ---------- dataset IO ----------
def load_dataset_norms(out_path: pathlib.Path) -> set:
    if not out_path.exists():
        return set()
    norms = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                msgs = obj.get("messages", [])
                if msgs:
                    txt = msgs[-1].get("content", "")
                    if txt:
                        norms.add(norm_text(txt).lower())
            except Exception:
                pass
    return norms


# ---------- state ----------
def load_state(path: pathlib.Path) -> Dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"version": 1, "files": {}, "last_run_at": None}


def save_state(path: pathlib.Path, state: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------- builders ----------
def _prompt_style(topics: List[str], lang_hint: str, prompt_lang: str) -> str:
    hint = (f"{lang_hint} " if lang_hint else "")  # <-- no .strip(): fixes "heWrite" bug
    if prompt_lang == "he":
        return f"{hint}כתוב פסקה בסגנון החתימה שלי על: " + ", ".join(topics) + ". שמור על אורך דומה."
    # default English
    return f"{hint}Write a paragraph in my signature style about: " + ", ".join(topics) + ". Aim for a similar length."


def _prompt_question(heading: str, lang_hint: str, prompt_lang: str) -> str:
    hint = (f"{lang_hint} " if lang_hint else "")  # <-- no .strip()
    head = heading.strip().rstrip("?")
    if prompt_lang == "he":
        return f"{hint}{head}?"
    return f"{hint}{head}?"


def make_style_example(
    txt: str,
    src: pathlib.Path,
    chunk_ix: int,
    lang_hint: str,
    prompt_lang: str,
    tag_lang: str,
) -> Dict:
    topics = topic_keywords(txt)
    prompt = _prompt_style(topics, lang_hint, prompt_lang)
    mtime = datetime.datetime.fromtimestamp(src.stat().st_mtime, tz=datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    ex = {
        "source_path": str(src),
        "source_type": "doc",
        "created_at": mtime,
        "chunk_ix": chunk_ix,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "model", "content": txt},
        ],
    }
    if tag_lang:
        ex["lang"] = tag_lang
    return ex


def make_qa_example(
    heading: str,
    answer: str,
    src: pathlib.Path,
    chunk_ix: int,
    lang_hint: str,
    prompt_lang: str,
    tag_lang: str,
) -> Dict:
    question = _prompt_question(heading, lang_hint, prompt_lang)
    mtime = datetime.datetime.fromtimestamp(src.stat().st_mtime, tz=datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    ex = {
        "source_path": str(src),
        "source_type": "doc",
        "created_at": mtime,
        "chunk_ix": chunk_ix,
        "messages": [
            {"role": "user", "content": question},
            {"role": "model", "content": answer.strip()},
        ],
    }
    if tag_lang:
        ex["lang"] = tag_lang
    return ex


def extract_heading_pairs(text: str) -> List[Dict[str, str]]:
    """Find lines that look like headings (# H1 / ## H2 ...) or end with ':'.
    Pair each heading with the following non-empty paragraph block.
    """
    lines = text.splitlines()
    items: List[Dict[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        is_head = bool(re.match(r"^\s{0,3}#{1,6}\s+.+", line)) or line.endswith(":")
        if is_head:
            # advance to first non-empty line
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            # collect contiguous non-empty lines as the paragraph
            para_lines: List[str] = []
            while j < len(lines) and lines[j].strip():
                para_lines.append(lines[j].strip())
                j += 1
            if para_lines:
                head_txt = re.sub(r"^\s{0,3}#{1,6}\s+", "", line).rstrip(":").strip()
                ans = norm_text(" ".join(para_lines))
                if head_txt and ans:
                    items.append({"q": head_txt, "a": ans})
            i = j
        else:
            i += 1
    return items


# ---------- file iteration ----------
def iter_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    if root.is_file():
        if root.suffix.lower() in SUPPORTED_EXT:
            yield root
        return
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            yield p


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

            known_hashes = set(info.get("chunks", []))

            if not changed and known_hashes:
                # unchanged since last run → skip
                continue

            # read text
            try:
                raw = load_text(fp, want_headings=(args.mode == "qa_heading"))
            except Exception as e:  # read failure
                print(f"[warn] failed to read {fp}: {e}")
                state["files"][key] = {"mtime": stat.st_mtime, "size": stat.st_size, "chunks": sorted(known_hashes)}
                continue

            text = norm_ws(raw)
            if not text:
                state["files"][key] = {"mtime": stat.st_mtime, "size": stat.st_size, "chunks": sorted(known_hashes)}
                continue

            new_hashes = set(known_hashes)

            if args.mode == "qa_heading":
                pairs = extract_heading_pairs(text)
                for k, pair in enumerate(pairs):
                    ans = pair["a"]
                    if len(ans) < args.min_chars:
                        continue
                    if len(ans) > args.max_chars:
                        from_ix = 0
                        for j, chunk in enumerate(chunk_long(ans, args.max_chars)):
                            if len(chunk) < args.min_chars:
                                continue
                            h = sha256_text(chunk)
                            if h in new_hashes or (dataset_norms and norm_text(chunk).lower() in dataset_norms):
                                continue
                            ex = make_qa_example(pair["q"], chunk, fp, k * 10 + j, args.lang_hint, args.prompt_lang, args.tag_lang)
                            out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                            added += 1
                            new_hashes.add(h)
                            dataset_norms.add(norm_text(chunk).lower())
                            from_ix += 1
                    else:
                        h = sha256_text(ans)
                        if h in new_hashes or (dataset_norms and norm_text(ans).lower() in dataset_norms):
                            continue
                        ex = make_qa_example(pair["q"], ans, fp, k, args.lang_hint, args.prompt_lang, args.tag_lang)
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
                        ex = make_style_example(ch, fp, p_ix * 10 + j, args.lang_hint, args.prompt_lang, args.tag_lang)
                        out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                        added += 1
                        new_hashes.add(h)
                        dataset_norms.add(norm_text(ch).lower())

            # update state for this file
            state["files"][key] = {"mtime": stat.st_mtime, "size": stat.st_size, "chunks": sorted(new_hashes)}

    # optionally prune missing files from state
    if args.delete_missing:
        keys = set(state["files"].keys())
        for k in keys - set(seen_files):
            del state["files"][k]

    state["last_run_at"] = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    save_state(statep, state)
    print(f"[docs_sync] appended {added} examples to {outp}")

if __name__ == "__main__":
    main()

