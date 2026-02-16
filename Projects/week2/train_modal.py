"""
Week 2 — Modal Training Script
================================
Runs the full SFT warm-up → DPO pipeline on a cloud GPU without opening a browser.

Usage
-----
    # One-time setup
    pip install modal
    modal setup                        # authenticates with your Modal account

    # Add secrets (once)
    modal secret create huggingface HF_TOKEN=hf_...
    modal secret create openai OPENAI_API_KEY=sk-...  # optional, for eval

    # Run training
    modal run train_modal.py

    # Override model or config
    modal run train_modal.py --model "Qwen/Qwen2.5-1.5B-Instruct" --beta 0.15

    # Download trained model to local ./models/
    modal run train_modal.py --download-only
"""

import math
import os
from pathlib import Path

import modal

# ── Modal infrastructure ──────────────────────────────────────────────────────

IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.2.0",
    "transformers==4.44.2",
    "peft==0.12.0",
    "trl==0.11.4",
    "bitsandbytes>=0.43.0",
    "accelerate>=0.28.0",
    "datasets>=2.18.0",
    "pandas>=2.2.0",
    "pyarrow>=15.0.0",
    "wandb>=0.16.0",
    "matplotlib>=3.8.0",
    "huggingface_hub>=0.21.0",
    "rich>=13.0.0",
)

# Persistent volume — model checkpoints survive across runs
volume = modal.Volume.from_name("week2-models", create_if_missing=True)
VOLUME_PATH = Path("/vol")

app = modal.App("week2-finetune", image=IMAGE)

# ── Config (override via CLI args in local_entrypoint) ────────────────────────

DEFAULT_CONFIG = {
    "base_model": "google/gemma-2-2b-it",
    # Other good options:
    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "meta-llama/Llama-3.2-1B-Instruct"
    # "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_targets": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    # SFT
    "sft_epochs": 1,
    "sft_batch_size": 4,
    "sft_lr": 2e-4,
    "sft_max_seq_len": 1024,
    "sft_score_floor": 0.80,
    # DPO
    "dpo_beta": 0.2,
    "dpo_epochs": 1,
    "dpo_batch_size": 2,
    "dpo_lr": 5e-5,
    "dpo_max_length": 1024,
    # Data
    "chosen_floor": 0.70,
    "delta_min": 0.15,
    "max_pairs": 1000,
    "log_every": 5,
}


# ── Data preparation (NB01 logic) ─────────────────────────────────────────────


@app.function(
    cpu=4,
    memory=8192,
    timeout=1800,
    volumes={str(VOLUME_PATH): volume},
)
def prepare_data(cfg: dict) -> str:
    """Download UltraFeedback, filter, build DPO pairs. Returns path to JSONL."""
    import pandas as pd
    from datasets import load_dataset

    print("Loading UltraFeedback binarized...")
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    df = dataset.to_pandas()
    print(f"Raw rows: {len(df):,}")

    # Extract text
    def extract_last_assistant(messages) -> str:
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    return msg.get("content", "")
        return str(messages)

    df["prompt_clean"] = df["prompt"].astype(str).str.strip()
    df["response_chosen"] = df["chosen"].apply(extract_last_assistant)
    df["response_rejected"] = df["rejected"].apply(extract_last_assistant)

    # Normalise scores (1–5 → 0–1)
    SCORE_MAX = 5.0
    df["score_chosen"] = pd.to_numeric(df["score_chosen"], errors="coerce") / SCORE_MAX
    df["score_rejected"] = (
        pd.to_numeric(df["score_rejected"], errors="coerce") / SCORE_MAX
    )
    df = df.dropna(subset=["score_chosen", "score_rejected"])
    df["score_delta"] = df["score_chosen"] - df["score_rejected"]

    # Quality filter
    df = df[df["prompt_clean"].str.split().str.len().between(5, 2000)]
    df = df[~df.duplicated(subset=["prompt_clean"], keep="first")]

    # Floor + delta filter
    df_floor = df[
        (df["score_chosen"] >= cfg["chosen_floor"])
        & (df["score_delta"] >= cfg["delta_min"])
    ].copy()

    print(f"After filtering: {len(df_floor):,} pairs")

    # Cap at max_pairs
    if cfg["max_pairs"] and len(df_floor) > cfg["max_pairs"]:
        df_floor = df_floor.sample(cfg["max_pairs"], random_state=42).reset_index(
            drop=True
        )
        print(f"Sampled to {cfg['max_pairs']} pairs")

    # Save
    out = pd.DataFrame(
        {
            "prompt": df_floor["prompt_clean"],
            "chosen": df_floor["response_chosen"],
            "rejected": df_floor["response_rejected"],
            "score_chosen": df_floor["score_chosen"],
            "score_rejected": df_floor["score_rejected"],
            "score_delta": df_floor["score_delta"],
        }
    )

    out_path = str(VOLUME_PATH / "dpo_floor.jsonl")
    out.to_json(out_path, orient="records", lines=True)
    volume.commit()
    print(f"Saved {len(out):,} pairs → {out_path}")
    return out_path


# ── Metrics callback ──────────────────────────────────────────────────────────


def make_metrics_callback(log_every: int = 5):
    from transformers import TrainerCallback

    class MetricsCallback(TrainerCallback):
        def __init__(self):
            self.steps, self.losses, self.perplexities = [], [], []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None or state.global_step % log_every != 0:
                return
            loss = logs.get("loss") or logs.get("train_loss")
            if loss is None:
                return
            ppl = math.exp(min(loss, 20))
            self.steps.append(state.global_step)
            self.losses.append(loss)
            self.perplexities.append(ppl)
            norm = ppl / (self.perplexities[0] or 1.0)
            print(
                f"  step {state.global_step:>4} | loss {loss:.4f} | "
                f"ppl {ppl:.2f} | norm_ppl {norm:.3f}"
            )

        def summary(self):
            if not self.losses:
                return
            norm = [p / (self.perplexities[0] or 1.0) for p in self.perplexities]
            print(
                f"\nFinal → loss: {self.losses[-1]:.4f} | "
                f"ppl: {self.perplexities[-1]:.2f} | "
                f"norm_ppl: {norm[-1]:.3f}"
            )
            return {
                "steps": self.steps,
                "losses": self.losses,
                "perplexities": self.perplexities,
                "norm_ppl": norm,
            }

    return MetricsCallback()


# ── Training function ─────────────────────────────────────────────────────────


@app.function(
    gpu="A100",
    timeout=7200,
    volumes={str(VOLUME_PATH): volume},
    secrets=[
        modal.Secret.from_name("huggingface"),
    ],
)
def train(cfg: dict) -> dict:
    import torch
    import pandas as pd
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
    )
    from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
    from huggingface_hub import login

    # Auth
    login(token=os.environ["HF_TOKEN"])
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    run_name = cfg["base_model"].split("/")[-1] + "-dpo"
    model_dir = VOLUME_PATH / run_name

    # ── Load data ──────────────────────────────────────────────────────────
    data_path = VOLUME_PATH / "dpo_floor.jsonl"
    if not data_path.exists():
        raise FileNotFoundError("dpo_floor.jsonl not found — run prepare_data first.")

    df_dpo = pd.read_json(str(data_path), lines=True)
    print(f"Loaded {len(df_dpo):,} DPO pairs")

    # ── Load model ─────────────────────────────────────────────────────────
    # A100 40 GB has plenty of room for a 2B model in bf16 (~4 GB).
    # Skipping 4-bit quantization avoids bitsandbytes/accelerate version conflicts.
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_targets"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Phase 1: SFT warm-up ───────────────────────────────────────────────
    print("\n=== Phase 1: SFT warm-up ===")
    df_sft = (
        df_dpo[df_dpo["score_chosen"] >= cfg["sft_score_floor"]].copy()
        if "score_chosen" in df_dpo.columns
        else df_dpo.sample(min(700, len(df_dpo)), random_state=42)
    )
    df_sft["text"] = df_sft.apply(
        lambda r: f"### User:\n{r['prompt']}\n\n### Assistant:\n{r['chosen']}", axis=1
    )
    ds_sft = Dataset.from_pandas(df_sft[["text"]])
    print(f"SFT samples: {len(ds_sft):,}")

    sft_cb = make_metrics_callback(cfg["log_every"])
    sft_args = SFTConfig(
        output_dir=str(model_dir / "sft"),
        num_train_epochs=cfg["sft_epochs"],
        per_device_train_batch_size=cfg["sft_batch_size"],
        gradient_accumulation_steps=4,
        learning_rate=cfg["sft_lr"],
        bf16=True,
        logging_steps=cfg["log_every"],
        save_strategy="epoch",
        optim="adamw_torch",
        report_to="none",
        dataset_text_field="text",
        max_seq_length=cfg["sft_max_seq_len"],
    )
    sft_trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds_sft,
        processing_class=tokenizer,
        callbacks=[sft_cb],
    )
    sft_trainer.train()
    sft_trainer.save_model(str(model_dir / "sft-final"))
    sft_metrics = sft_cb.summary()

    # ── Phase 2: DPO ──────────────────────────────────────────────────────
    print(f"\n=== Phase 2: DPO (β={cfg['dpo_beta']}) ===")
    ds_dpo = Dataset.from_pandas(df_dpo[["prompt", "chosen", "rejected"]])
    split = ds_dpo.train_test_split(test_size=0.05, seed=42)
    print(f"DPO train: {len(split['train']):,} | eval: {len(split['test']):,}")

    dpo_cb = make_metrics_callback(cfg["log_every"])
    dpo_config = DPOConfig(
        beta=cfg["dpo_beta"],
        output_dir=str(model_dir / "dpo"),
        num_train_epochs=cfg["dpo_epochs"],
        per_device_train_batch_size=cfg["dpo_batch_size"],
        per_device_eval_batch_size=cfg["dpo_batch_size"],
        gradient_accumulation_steps=8,
        learning_rate=cfg["dpo_lr"],
        bf16=True,
        logging_steps=cfg["log_every"],
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        max_length=cfg["dpo_max_length"],
        max_prompt_length=512,
        optim="adamw_torch",
        report_to="none",
    )
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        callbacks=[dpo_cb],
    )
    dpo_trainer.train()
    dpo_trainer.save_model(str(model_dir / "dpo-final"))
    dpo_metrics = dpo_cb.summary()

    volume.commit()
    print(f"\nModel saved to volume: {model_dir}")

    return {"sft": sft_metrics, "dpo": dpo_metrics, "model_dir": str(model_dir)}


# ── Download helper ───────────────────────────────────────────────────────────


@app.function(volumes={str(VOLUME_PATH): volume})
def list_checkpoints():
    import subprocess  # nosec B404

    result = subprocess.run(  # nosec B603, B607
        ["find", str(VOLUME_PATH), "-name", "config.json"],
        capture_output=True,
        text=True,
    )
    print(result.stdout or "No checkpoints found yet.")


# ── Local entrypoint ──────────────────────────────────────────────────────────


_DEFAULT_MODEL: str = "google/gemma-2-2b-it"
_DEFAULT_BETA: float = 0.2
_DEFAULT_MAX_PAIRS: int = 1000


@app.local_entrypoint()
def main(
    model: str = _DEFAULT_MODEL,
    beta: float = _DEFAULT_BETA,
    max_pairs: int = _DEFAULT_MAX_PAIRS,
    skip_data: bool = False,
    download_only: bool = False,
):
    cfg = {
        **DEFAULT_CONFIG,
        "base_model": model,
        "dpo_beta": beta,
        "max_pairs": max_pairs,
    }

    if download_only:
        list_checkpoints.remote()
        return

    if not skip_data:
        print("Step 1/2 — Preparing data...")
        prepare_data.remote(cfg)

    print("Step 2/2 — Training...")
    results = train.remote(cfg)

    # Plot metrics locally after training
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        fig.suptitle(f"Training Curves — {model}", fontsize=13)

        for row, (phase, metrics) in enumerate(
            [("SFT", results["sft"]), ("DPO", results["dpo"])]
        ):
            if not metrics:
                continue
            norm = metrics["norm_ppl"]
            for ax, values, ylabel, color in zip(
                axes[row],
                [metrics["losses"], metrics["perplexities"], norm],
                ["Loss", "Perplexity", "Norm Perplexity"],
                ["steelblue", "darkorange", "seagreen"],
            ):
                ax.plot(metrics["steps"], values, color=color, linewidth=1.8)
                ax.set_title(f"{phase} — {ylabel}")
                ax.set_xlabel("Step")
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        plt.tight_layout()
        out_png = Path("Projects/week2/data/training_curves.png")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Curves saved → {out_png}")
        plt.show()
    except Exception as e:
        print(f"Plot skipped: {e}")

    print("\nDone. Checkpoints are in the Modal volume.")
    print("To download a checkpoint:")
    print("  modal volume get week2-models <remote-path> <local-path>")
