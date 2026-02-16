# Running Week 2 Training via Modal (from Cursor / terminal)

No browser needed. Runs on a cloud A100, streams logs back to your terminal.

---

## 1. Install & Authenticate

```bash
pip install modal
modal setup          # opens browser once for auth, then done
```

---

## 2. Add Secrets (one-time)

```bash
# HuggingFace token (required for gated models like gemma-2, llama-3)
modal secret create huggingface HF_TOKEN=hf_your_token_here

# OpenAI (optional — only needed for NB03 VJ scoring)
modal secret create openai OPENAI_API_KEY=sk-your_key_here
```

Verify secrets exist:
```bash
modal secret list
```

---

## 3. Run Training

```bash
cd 'agentic_learning/Projects/week2'

# Default: gemma-2-2b-it, beta=0.2, 1000 pairs
modal run train_modal.py

# Custom model
modal run train_modal.py --model "Qwen/Qwen2.5-1.5B-Instruct"

# Custom beta and pair count
modal run train_modal.py --beta 0.15 --max-pairs 500

# Skip data prep if already done (saves ~5 min)
modal run train_modal.py --skip-data

# Just list saved checkpoints
modal run train_modal.py --download-only
```

Logs stream live to your terminal. Loss / perplexity / norm_perplexity printed every 5 steps.
After training completes, learning curves are saved to `data/training_curves.png`.

---

## 4. Download Trained Model

Checkpoints are stored in a persistent Modal Volume (`week2-models`).

```bash
# List what's in the volume
modal volume ls week2-models

# Download the DPO checkpoint locally
modal volume get week2-models /vol/gemma-2-2b-it-dpo/dpo-final ./models/gemma-2-2b-it-dpo
```

---

## 5. GPU & Cost

| GPU | VRAM | $/hr (approx) | Notes |
|-----|------|---------------|-------|
| A100 | 40 GB | ~$3.00 | Default. Fastest. |
| A10G | 24 GB | ~$1.10 | Change `gpu="A10G"` in script |
| T4   | 16 GB | ~$0.60 | Slow; only for 1.5B models |

A full run (data prep + SFT + DPO on 1000 pairs with gemma-2-2b-it) takes ~20–30 min on A100.

---

## 6. Change GPU

Edit `train_modal.py` line with `@app.function(gpu=...)`:

```python
@app.function(gpu="A10G", ...)   # cheaper
@app.function(gpu="H100", ...)   # fastest
```
