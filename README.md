# Twitter → Gemma SFT Dataset

Convert a Twitter/X archive into a **Supervised Fine-Tuning (SFT)** dataset for **Gemma 3**.  
Includes a simple pipeline to format your tweets into `messages` pairs compatible with Gemma’s chat template, plus quick tips for training (TRL/QLoRA) and local inference with **Ollama**.

---

## Quickstart

**Prepare a Python env**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install datasets transformers trl peft accelerate bitsandbytes
Unzip your Twitter archive into a folder (e.g., ~/Downloads/twitter-archive).

** Convert archive → JSONL**

```bash
python twitter_to_jsonl.py path/to/twitter-archive --out path/to/dataset/train.jsonl
```

This writes:

train.jsonl — training split
train_eval.jsonl — small, chronological evaluation split

## Run a Gemma model locally with Ollama

```bash
ollama run gemma3:4b
```

Try:

```
“Write a concise tweet in my signature style about: robotics, SLAM, AR.”
```

## What this tool does

Parses classic Twitter/X exports (data/tweets.js or data/tweets/*.js) and raw JSON/JSONL.
Cleans text (HTML entities, whitespace); skips retweets; optional: include/exclude quote tweets.
Builds SFT pairs that teach the model your style/domain:

```json
{
  "messages": [
    {"role": "user",  "content": "Write a concise tweet in my signature style about: robotics, slam, ar. Keep it under 280 characters."},
    {"role": "model", "content": "Your original tweet text here"}
  ]
}
```

- De-duplicates near-identical tweets.
- Splits off a small hold-out set by time (last ~10%).

If your archive (or API exports) include parent tweets/replies, you can switch to true dialog SFT:

```json

{"messages":[
  {"role":"user","content":"@you what’s your take on visual SLAM vs LiDAR?"},
  {"role":"model","content":"Short, opinionated reply (your tweet)."}
]}
```

## CLI

```bash
python twitter_to_jsonl.py ARCHIVE_PATH --out DATASET_PATH [--min_len 10] [--no_quotes]
```

Args
- ARCHIVE_PATH – folder or file. Supports: data/tweets.js, data/tweets/*.js, tweets/*.js - raw .json or .jsonl
--out – output JSONL path (e.g., dataset/train.jsonl)
--min_len – drop very short tweets (default: 10)
--no_quotes – exclude quote tweets

