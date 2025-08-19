# Twitter to Gemma 3 SFT Dataset

Convert your Twitter/X archive into a supervised fine-tuning dataset for **Gemma 3**. This repo includes: (1) a converter that emits Gemma chat `messages` JSONL, (2) an optional **incremental sync** that pulls only new tweets since your last run, and (3) a lightweight **TRL + QLoRA** training setup. For quick smoke tests, you can run models locally with **Ollama**. Compatible with Gemma 3 sizes from **270M to 27B**.

Training produces a **LoRA adapter**—small plug-in weights that you can keep separate for fast iteration or **merge into the base model** to create a standalone fine-tuned checkpoint.

## What is a LoRA adapter?

**LoRA (Low-Rank Adaptation)** fine-tunes a large model by training a **small set of extra weights** (the *adapter*) while keeping the base model frozen.  
Why it’s useful:

- **Lightweight:** adapters are tiny compared to the base and are faster/cheaper to train & share.
- **Swappable:** keep the base the same and plug in different domain/style adapters.
- **Mergeable:** when you’re happy, you can **merge** the adapter into the base to get a standalone finetuned model (no PEFT needed at inference).

In this repo, training writes a LoRA adapter to `out/gemma3-twitter-lora/` by default.

---

## What this tool does

<details>
<summary><b>Dataset preparation from Twitter/X export</b></summary>

**Input:** Twitter/X export (`data/tweets.js`, `data/tweets/*.js`) or raw JSON/JSONL  
**Output:** Gemma-ready SFT in `dataset/train.jsonl` (+ time-held `dataset/train_eval.jsonl`)

### Pipeline
- **Parse** classic archives (strips the `window.YTD...` JS wrapper) and raw JSON/JSONL.
- **Clean** text (HTML unescape, whitespace), trim tracking params in URLs.
- **Filter**: drop retweets; optionally skip quote tweets (`--no_quotes`).
- **Build SFT pairs** (teach tone/domain):
  ```json
  {
    "messages": [
      {"role": "user",  "content": "Write a concise tweet in my signature style about: robotics, slam, ar. Keep it under 280 characters."},
      {"role": "model", "content": "Your original tweet text here"}
    ],
    "tweet_id": "1234567890",
    "created_at": "2025-08-18T15:00:12Z"
  }
  ```
- **Topics prompt**: inferred from hashtags + light keywording.
- **De-duplicate** near-identical tweets (normalized text).
- **Eval split**: last \~10% by `created_at`.

### Dialog SFT (if parent tweets/replies exist)

```json
{"messages":[
  {"role":"user","content":"@you what’s your take on visual SLAM vs LiDAR?"},
  {"role":"model","content":"Short, opinionated reply (your tweet)."}
]}
```

### Converter CLI flags

* `--out PATH` — output JSONL (e.g., `dataset/train.jsonl`)
* `--min_len N` — skip tweets shorter than *N* chars (default: `10`)
* `--no_quotes` — exclude quote tweets
  *(auto-detects `data/tweets.js`, `data/tweets/*.js`, raw `.json`, `.jsonl`)*

> **Notes:** Only include content you own (avoid DMs / third-party text).
> For ongoing updates, use **incremental sync** to append only new tweets, dedup by ID/text, and filter bot-generated posts/sources.

</details>

---

## Quickstart

### 1) Set up a Python environment
Recommended (uses `requirements.txt`):
```bash
./run.sh setup
```
Which performs:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

2) Unzip your Twitter archive
Unzip the export to a folder, e.g. ~/Downloads/twitter-archive.

3) Convert archive to JSONL
Using the helper script:

```bash
./run.sh convert ~/Downloads/twitter-archive dataset/train.jsonl
```

Which performs in the `venv`:

```bash
python twitter_to_jsonl.py ~/Downloads/twitter-archive --out dataset/train.jsonl
```

This writes:

* `dataset/train.jsonl` — training split
* `dataset/train_eval.jsonl` — small, chronological evaluation split

## CLI for `run.sh`

`run.sh` is a thin wrapper that **auto-activates your venv** and (optionally) reads config from a `.env` file.  
General form:

```bash
./run.sh <command> [options]
````

### Commands
<details>
<summary><b>Usage</b></summary>

* **setup**
  Create `.venv/` and install `requirements.txt`.

  ```bash
  ./run.sh setup
  ```

* **shell**
  Open a subshell with the venv activated.

  ```bash
  ./run.sh shell
  ```

* **convert `<ARCHIVE>` \[OUT]**
  Convert a Twitter/X export to Gemma-style JSONL (defaults to `dataset/train.jsonl`).
  Auto-detects `twitter_to_jsonl.py` in repo root or `dataset/`.

  ```bash
  ./run.sh convert ~/Downloads/twitter-archive dataset/train.jsonl
  ```

* **sync \[args…]**
  Incrementally fetch **new tweets since last run** and append to the dataset.
  If you **pass args**, they are forwarded to `incremental_sync.py`.
  If you **pass no args**, it uses `.env` (see below).

  ```bash
  # pass-through
  ./run.sh sync --username your_handle --out dataset/train.jsonl --state state/last_sync.json

  # .env-driven (no args)
  ./run.sh sync
  ```

* **train \[args…]**
  Run `train_gemma3.py`. Arguments are forwarded (e.g., `--epochs`, `--resume`, `--model`).

  ```bash
  ./run.sh train --epochs 1 --resume
  ```

* **daily**
  Convenience command: **sync** (using `.env`) → **train** (auto-resume).
  Uses `EPOCHS` from `.env` (default `1`).

  ```bash
  ./run.sh daily
  ```

* **clean**
  Remove the virtual environment.

  ```bash
  ./run.sh clean
  ```

### Configuration via `.env`

`run.sh` loads `.env` automatically (override with `ENV_FILE=/path/to/.env`).
Common keys:

| Key                                                     | Purpose                                              |
| ------------------------------------------------------- | ---------------------------------------------------- |
| `TWITTER_BEARER_TOKEN`                                  | **Required** for API (or pass `--bearer` to `sync`)  |
| `TWITTER_USERNAME`                                      | Your handle **without** `@` (used by `sync`/`daily`) |
| `EXCLUDE_SOURCES`                                       | App sources to skip (e.g., `MyBotApp AnotherApp`)    |
| `INCLUDE_REPLIES=1`                                     | Include replies in sync                              |
| `NO_QUOTES=1`                                           | Exclude quote tweets                                 |
| `EPOCHS=1`                                              | Epochs used by `daily`                               |
| `MODEL_NAME`                                            | Default model for `train_gemma3.py`                  |
| `PY_BIN`                                                | Python executable (e.g., `python3.11`)               |
| `VENV_DIR`, `DEFAULT_OUT`, `STATE_FILE`, `TRAIN_SCRIPT` | Optional path overrides                              |

Example `.env`:

```bash
TWITTER_BEARER_TOKEN=YOUR_TOKEN_HERE
TWITTER_USERNAME=your_handle_without_at
EXCLUDE_SOURCES=MyBotApp
INCLUDE_REPLIES=1
NO_QUOTES=1
EPOCHS=1
MODEL_NAME=google/gemma-3-4b-it
PY_BIN=python3.11
```

> **Note:** `.env` (and `state/`) should be in `.gitignore`. All commands that run Python call the venv **activate** internally; your parent shell remains unchanged.

</details>

## Inference

<details>
<summary><b>Inference with a LoRA adapter</b></summary>

Run your base Gemma model + your adapter (uses `infer_adapter.py` under the hood).

```bash
# Using explicit args:
./run.sh infer google/gemma-3-4b-it out/gemma3-twitter-lora \
  "Write a concise tweet in my signature style about: robotics, SLAM, AR."

# Or pipe the prompt:
echo "Write a concise tweet in my signature style about: SLAM vs LiDAR." \
  | ./run.sh infer google/gemma-3-4b-it out/gemma3-twitter-lora
```

**Defaults:** If you set these in `.env`, you can omit them on the CLI.

```bash
MODEL_NAME=google/gemma-3-4b-it
ADAPTER_DIR=out/gemma3-twitter-lora
```

Then:

```bash
./run.sh infer "" "" "Write a concise tweet about: AR tracking."
# or
echo "Write a concise tweet about: AR tracking." | ./run.sh infer
```
</details>
<details>
<summary><b>Merge the adapter into standalone weights</b></summary>

Bake the adapter into a new folder so you can run it **without** PEFT (uses `merge_adapter.py`).

```bash
./run.sh merge_adapter google/gemma-3-4b-it out/gemma3-twitter-lora out/gemma3-merged
```

After merging, the folder `out/gemma3-merged/` contains a standalone Gemma checkpoint + tokenizer.

You can set a default in `.env`:

```bash
MERGED_DIR=out/gemma3-merged
```
</details>
<details>
<summary><b>Inference with the merged model</b></summary>

Run the merged model directly (uses `infer_merged.py`).

```bash
./run.sh infer_merged out/gemma3-merged \
  "Write a concise tweet in my signature style about: UAV SLAM."
# or via stdin
echo "Write a concise tweet in my signature style about: UAV SLAM." \
  | ./run.sh infer_merged out/gemma3-merged
```

With `.env`:

```bash
MERGED_DIR=out/gemma3-merged
```

…you can simply do:

```bash
./run.sh infer_merged "" "Write a concise tweet about: AR occlusion."
# or
echo "Write a concise tweet about: AR occlusion." | ./run.sh infer_merged
```
</details>
<details>
<summary><b> Tips & gotchas</b></summary>

* **Pick the right folder:** pointing `infer` at the adapter **root** (`out/gemma3-twitter-lora/`) usually gives you the best weights (if you trained with `load_best_model_at_end`). Otherwise, try a specific `checkpoint-XXXX/`.
* **Memory:** for GPU inference the scripts default to `bfloat16` when available. On CPU, dtype falls back to default float; it’s slower, so prefer smaller Gemma sizes.
* **Ollama:** Ollama cannot load a Hugging Face **adapter** directly. If you want to run your finetuned weights in Ollama, **merge** first, then follow Ollama’s custom-model guidance.

</details>

## Run a Gemma model locally with Ollama

To use our finetuned model, it needs to be merged and converted to **GGUF** and potentially **Quantized**

<details>
<summary><b> To use an off-the-shelf Gemma 3 build</b></summary>

```bash
ollama pull gemma3:4b
ollama run gemma3:4b
````

One-liner:

```bash
ollama run gemma3:4b -p "Write a concise tweet in my signature style about: robotics, SLAM, AR."
```

> Gemma 3 models are available in Ollama 

</details>

<details>
<summary><b> Run <i>your merged fine-tuned</i> model in Ollama</b></summary>

Ollama runs models in **GGUF**. For your fine-tune, first merge the LoRA adapter into the base (you already have a command for this), then convert to GGUF, optionally quantize, and create a small Modelfile.

1. **Merge your adapter → standalone HF folder**

```bash
./run.sh merge_adapter google/gemma-3-4b-it out/gemma3-twitter-lora out/gemma3-merged
```

2. **Convert the merged model to GGUF**

```bash
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements.txt

# The script name may be `convert_hf_to_gguf.py` (used to be `convert-hf-to-gguf.py`).
python llama.cpp/convert_hf_to_gguf.py out/gemma3-merged --outfile out/gemma3-merged-f16.gguf
```

> Tuned Gemma models must be converted to GGUF to run in Ollama. 
> If you see references to different script names, that’s due to a rename in llama.cpp. 

3. **(Optional) Quantize** for smaller size/speed (popular choice: `Q4_K_M`)

```bash
./llama.cpp/quantize out/gemma3-merged-f16.gguf out/gemma3-merged-Q4_K_M.gguf Q4_K_M
```

> Ollama imports GGUF directly; quantized variants are common for local use. 

4. **Create a Modelfile** (minimal)

```bash
# Modelfile
FROM ./out/gemma3-merged-Q4_K_M.gguf
```

5. **Create and run your model in Ollama**

```bash
ollama create my-gemma -f Modelfile
ollama run my-gemma -p "Write a concise tweet in my signature style about: AR tracking."
```

> Modelfiles let you import a local GGUF and run it via `ollama create` / `ollama run`. 

</details>
