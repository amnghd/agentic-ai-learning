"""
Week 2 — Modal Evaluation Script
==================================
Generates responses from base, SFT, and DPO checkpoints on A100.
Saves responses.jsonl to the Modal volume for local VJ scoring in NB03.

Usage
-----
    # Run generation (takes ~10 min on A100)
    modal run eval_modal.py

    # Download results for NB03
    modal volume get week2-models /responses.jsonl ./data/eval_results/responses.jsonl

    # Optional: fewer prompts for a quick test
    modal run eval_modal.py --n-prompts 50
"""

import os
from pathlib import Path

import modal

# ── Reuse the same image and volume as train_modal.py ─────────────────────────

IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0",
    "transformers==4.44.2",
    "peft==0.12.0",
    "trl==0.11.4",
    "accelerate>=0.28.0",
    "datasets>=2.18.0",
    "pandas>=2.2.0",
    "pyarrow>=15.0.0",
    "huggingface_hub>=0.21.0",
    "rich>=13.0.0",
)

volume = modal.Volume.from_name("week2-models", create_if_missing=True)
VOLUME_PATH = Path("/vol")

app = modal.App("week2-eval", image=IMAGE)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODEL = "google/gemma-2-2b-it"
RUN_NAME = "gemma-2-2b-it-dpo"

# Paths inside the volume (written by train_modal.py)
SFT_ADAPTER = str(VOLUME_PATH / RUN_NAME / "sft-final")
DPO_ADAPTER = str(VOLUME_PATH / RUN_NAME / "dpo-final")

# Output path in the volume
RESPONSES_OUT = str(VOLUME_PATH / "responses.jsonl")


# ── Generation function ───────────────────────────────────────────────────────


@app.function(
    gpu="A100",
    timeout=3600,
    volumes={str(VOLUME_PATH): volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def generate_responses(n_prompts: int = 200) -> str:
    """Load base/SFT/DPO checkpoints, generate responses, save to volume."""
    import torch
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import login

    login(token=os.environ["HF_TOKEN"])
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load test prompts from dpo_floor.jsonl ─────────────────────────────
    data_path = VOLUME_PATH / "dpo_floor.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(
            "dpo_floor.jsonl not in volume. Run: modal run train_modal.py"
        )

    df = pd.read_json(str(data_path), lines=True)
    df = df.sample(min(n_prompts, len(df)), random_state=99).reset_index(drop=True)
    prompts = df["prompt"].tolist()
    print(f"Generating responses for {len(prompts)} prompts")

    # ── Helper: load model (base only, or base + adapter) ─────────────────
    def load_model(adapter_path=None):
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            attn_implementation="eager",
        )
        if adapter_path and Path(adapter_path).exists():
            model = PeftModel.from_pretrained(model, adapter_path)
            print(f"  Loaded adapter: {adapter_path}")
        else:
            print("  Base model (no adapter)")
        model.eval()
        return model, tokenizer

    # ── Helper: generate one response ─────────────────────────────────────
    def generate(model, tokenizer, prompt, max_new_tokens=200):
        text = f"### User:\n{prompt}\n\n### Assistant:\n"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    # ── Run all three checkpoints ──────────────────────────────────────────
    checkpoints = {
        "base": None,
        "sft": SFT_ADAPTER,
        "dpo": DPO_ADAPTER,
    }

    rows = []
    for ckpt_name, adapter_path in checkpoints.items():
        print(f"\n=== Checkpoint: {ckpt_name} ===")
        model, tokenizer = load_model(adapter_path)

        for i, prompt in enumerate(prompts):
            response = generate(model, tokenizer, prompt)
            rows.append(
                {"checkpoint": ckpt_name, "prompt": prompt, "response": response}
            )
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(prompts)} done")

        # Free VRAM before loading next checkpoint
        del model
        torch.cuda.empty_cache()
        print(f"  {ckpt_name} complete.")

    # ── Save ──────────────────────────────────────────────────────────────
    df_out = pd.DataFrame(rows)
    df_out.to_json(RESPONSES_OUT, orient="records", lines=True)
    volume.commit()
    print(f"\nSaved {len(df_out):,} rows → {RESPONSES_OUT}")
    return RESPONSES_OUT


# ── Local entrypoint ──────────────────────────────────────────────────────────

_DEFAULT_N: int = 200


@app.local_entrypoint()
def main(n_prompts: int = _DEFAULT_N) -> None:
    print(f"Generating responses for {n_prompts} prompts per checkpoint...")
    out_path = generate_responses.remote(n_prompts)
    print(f"\nDone. Responses saved to volume at: {out_path}")
    print("\nDownload with:")
    print(
        "  modal volume get week2-models /responses.jsonl"
        " ./data/eval_results/responses.jsonl"
    )
