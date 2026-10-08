"""
evaluate.py – VQA evaluation, question-type analysis, and AdaLoRA rank analysis
"""

import os
import re
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm

from src.data  import load_docvqa, DocVQADataset
from src.model import load_processor


# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────
RESULTS_DIR   = "results"
BATCH_SIZE    = 8
MAX_NEW_TOKENS = 64


# ────────────────────────────────────────────────────────────────────────────
# VQA exact-match metric (lowercased, punctuation stripped)
# ────────────────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def vqa_score(prediction: str, ground_truth: str) -> float:
    """Return 1 if normalised prediction == normalised answer, else 0."""
    return float(_normalise(prediction) == _normalise(ground_truth))


# ────────────────────────────────────────────────────────────────────────────
# Question-type classifier (simple heuristics for DocVQA)
# ────────────────────────────────────────────────────────────────────────────
_NUMERICAL_RE = re.compile(r"\b(how many|how much|what (is the )?(number|amount|total|count|year|date|price|cost))\b", re.I)
_LAYOUT_RE    = re.compile(r"\b(where|location|position|address|table|column|row|section)\b", re.I)

def classify_question(question: str) -> str:
    q = question.lower()
    if _NUMERICAL_RE.search(q):
        return "Numerical"
    if _LAYOUT_RE.search(q):
        return "Layout"
    # Fallback: assume text / OCR type
    return "Text/OCR"


# ────────────────────────────────────────────────────────────────────────────
# Generation-based evaluation
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(model, processor, test_hf, device: str,
                   max_new_tokens: int = MAX_NEW_TOKENS) -> pd.DataFrame:
    """
    Generate answers for each test example and compute per-sample VQA scores.

    Returns
    -------
    DataFrame with columns: question, ground_truth, prediction, score, q_type
    """
    from PIL import Image

    model.eval()
    records = []

    for item in tqdm(test_hf, desc="Evaluating"):
        image    = item["image"]
        question = item["question"]
        answers  = item.get("answers") or item.get("answer") or [""]
        gt       = answers[0] if isinstance(answers, list) else answers

        prompt = f"Question: {question}\nAnswer:"

        encoding = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        output_ids = model.generate(
            **encoding,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # Decode only the newly generated tokens
        gen_ids   = output_ids[:, encoding["input_ids"].shape[-1]:]
        prediction = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        score  = vqa_score(prediction, gt)
        q_type = classify_question(question)

        records.append({
            "question":     question,
            "ground_truth": gt,
            "prediction":   prediction,
            "score":        score,
            "q_type":       q_type,
        })

    return pd.DataFrame(records)


# ────────────────────────────────────────────────────────────────────────────
# Comparison helpers
# ────────────────────────────────────────────────────────────────────────────

def build_results_table(lora_df: pd.DataFrame, adalora_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-question-type VQA scores for both adapters.

    Returns
    -------
    DataFrame with columns: Question Type | LoRA | AdaLoRA | Difference
    """
    rows = []
    for q_type in ["Text/OCR", "Numerical", "Layout"]:
        lora_score    = lora_df[lora_df["q_type"] == q_type]["score"].mean()
        adalora_score = adalora_df[adalora_df["q_type"] == q_type]["score"].mean()
        rows.append({
            "Question Type": q_type,
            "LoRA":          round(lora_score,    4),
            "AdaLoRA":       round(adalora_score, 4),
            "Difference":    round(adalora_score - lora_score, 4),
        })

    # Overall row
    rows.append({
        "Question Type": "Overall",
        "LoRA":          round(lora_df["score"].mean(),    4),
        "AdaLoRA":       round(adalora_df["score"].mean(), 4),
        "Difference":    round(adalora_df["score"].mean() - lora_df["score"].mean(), 4),
    })

    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# AdaLoRA rank analysis
# ────────────────────────────────────────────────────────────────────────────

def get_adalora_ranks(model) -> dict:
    """
    Extract the effective rank of every AdaLoRA module by counting the
    non-zero singular values in the ``lora_E`` vector.

    Returns
    -------
    dict mapping module name → effective rank (int)
    """
    ranks = {}
    for name, param in model.named_parameters():
        if "lora_E" in name and param.requires_grad:
            # effective rank = number of non-zero (> 1e-6) singular values
            effective_rank = (param.abs() > 1e-6).sum().item()
            clean_name = name.replace(".lora_E", "").replace("base_model.model.", "")
            ranks[clean_name] = int(effective_rank)
    return ranks


# ────────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────────

def plot_training_curves(lora_history: dict, adalora_history: dict,
                         save_path: str = "results/training_curves.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(lora_history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, lora_history["train_loss"],    label="LoRA train",    marker="o")
    ax1.plot(epochs, lora_history["val_loss"],      label="LoRA val",      marker="o", ls="--")
    ax1.plot(epochs, adalora_history["train_loss"], label="AdaLoRA train", marker="s")
    ax1.plot(epochs, adalora_history["val_loss"],   label="AdaLoRA val",   marker="s", ls="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss"); ax1.legend()

    q_types  = ["Text/OCR", "Numerical", "Layout", "Overall"]
    x        = np.arange(len(q_types))
    width    = 0.35
    results  = pd.read_csv("results/results.csv")
    lora_s   = [results.loc[results["Question Type"] == qt, "LoRA"].values[0]    for qt in q_types]
    ada_s    = [results.loc[results["Question Type"] == qt, "AdaLoRA"].values[0] for qt in q_types]

    ax2.bar(x - width / 2, lora_s,  width, label="LoRA")
    ax2.bar(x + width / 2, ada_s,   width, label="AdaLoRA")
    ax2.set_xticks(x); ax2.set_xticklabels(q_types, rotation=15)
    ax2.set_ylabel("VQA Score")
    ax2.set_title("VQA Score by Question Type"); ax2.legend()
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Training curves saved → {save_path}")


def plot_rank_distribution(ranks: dict,
                           save_path: str = "results/rank_analysis.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    modules = list(ranks.keys())[:30]   # clip to first 30 for readability
    values  = [ranks[m] for m in modules]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(modules)), values, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(modules)))
    ax.set_xticklabels(modules, rotation=90, fontsize=7)
    ax.set_ylabel("Effective Rank")
    ax.set_title("AdaLoRA – Final Rank Allocation per Module")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Rank analysis saved → {save_path}")


# ────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from peft import PeftModel
    from src.model import load_base_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", choices=["lora", "adalora", "both"],
                        default="both")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_hf = load_docvqa()

    results = {}
    for name in (["lora", "adalora"] if args.adapter == "both" else [args.adapter]):
        ckpt = f"checkpoints/{name}"
        print(f"\nLoading {name.upper()} from {ckpt} …")
        processor  = load_processor()
        base_model = load_base_model()
        model      = PeftModel.from_pretrained(base_model, ckpt).to(device)
        df         = evaluate_model(model, processor, test_hf, device)
        results[name] = df

    if "lora" in results and "adalora" in results:
        table = build_results_table(results["lora"], results["adalora"])
        os.makedirs(RESULTS_DIR, exist_ok=True)
        table.to_csv(f"{RESULTS_DIR}/results.csv", index=False)
        print("\n=== Results ===")
        print(table.to_string(index=False))
