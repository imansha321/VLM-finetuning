# Quickstart — VLM Fine-Tuning (LoRA vs AdaLoRA)

Get from zero to running experiments in **5 steps**.

---

## Prerequisites

| Requirement | Minimum version |
|---|---|
| Python | 3.10 |
| CUDA (optional but recommended) | 11.8 |
| GPU VRAM | ≥ 8 GB (16 GB recommended) |
| Disk space | ~5 GB (model + dataset cache) |

---

## Step 1 — Clone the repo

```bash
git clone https://github.com/imansha321/VLM-finetuning.git
cd VLM-finetuning
```

---

## Step 2 — Create a virtual environment

```bash
# Using conda (recommended)
conda create -n vlm_ft python=3.10 -y
conda activate vlm_ft

# OR using venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

---

## Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Google Colab users** — run the install cell at the top of
> `notebooks/VLM_finetuning.ipynb` instead; skip Steps 2–3.

---

## Step 4 — Run an experiment

### Option A — Jupyter Notebook (recommended)

```bash
jupyter notebook notebooks/VLM_finetuning.ipynb
```

Run all cells top-to-bottom. The notebook covers:

```
Install → Load DocVQA → Build LoRA & AdaLoRA
→ Train both → Evaluate → Plot results → Rank analysis
```

### Option B — Command line

**Train LoRA baseline**

```bash
python -m src.train --adapter lora
```

**Train AdaLoRA**

```bash
python -m src.train --adapter adalora
```

**Evaluate both and generate plots**

```bash
python -m src.evaluate --adapter both
```

Checkpoints are saved to `checkpoints/lora/` and `checkpoints/adalora/`.  
Results are written to `results/results.csv` and plots to `results/`.

---

## Step 5 — Inspect the outputs

| Output file | Description |
|---|---|
| `results/results.csv` | VQA scores per question type |
| `results/training_curves.png` | Loss curves + bar chart comparison |
| `results/rank_analysis.png` | AdaLoRA per-module rank allocation |
| `checkpoints/lora/` | Best LoRA checkpoint |
| `checkpoints/adalora/` | Best AdaLoRA checkpoint |

---

## Key hyperparameters

| Parameter | Value | File |
|---|---|---|
| `SEED` | 42 | `src/train.py` |
| `BATCH_SIZE` | 4 | `src/train.py` |
| `LEARNING_RATE` | 2e-4 | `src/train.py` |
| `NUM_EPOCHS` | 3 | `src/train.py` |
| `LORA_R` | 16 | `src/model.py` |
| `ADA_TARGET_R` | 16 | `src/model.py` |
| `TRAIN_SIZE` | 2 000 | `src/data.py` |
| `VAL_SIZE` | 500 | `src/data.py` |
| `TEST_SIZE` | 500 | `src/data.py` |

---

## Running on Google Colab

1. Open [notebooks/VLM_finetuning.ipynb](notebooks/VLM_finetuning.ipynb) in Colab.
2. Set **Runtime → Change runtime type → T4 GPU**.
3. Run all cells (`Runtime → Run all`).

> Training 3 epochs on a T4 GPU takes approximately **60–90 minutes** depending on dataset size.

---

## Project layout

```
VLM_finetuning/
├── QUICKSTART.md          ← you are here
├── README.md
├── requirements.txt
├── notebooks/
│   └── VLM_finetuning.ipynb
├── src/
│   ├── data.py            ← DocVQA loading & Dataset class
│   ├── model.py           ← SmolVLM-256M + LoRA / AdaLoRA builders
│   ├── train.py           ← training loop
│   └── evaluate.py        ← evaluation, plots, rank analysis
└── results/
    ├── results.csv
    ├── training_curves.png
    └── rank_analysis.png
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | Reduce `BATCH_SIZE` to 2 or 1 in `src/train.py` |
| `trust_remote_code` error | Pass `trust_remote_code=True` to `load_dataset` (already set) |
| Model download is slow | Run `huggingface-cli login` and set `HF_HUB_ENABLE_HF_TRANSFER=1` |
| `ModuleNotFoundError: peft` | Re-run `pip install -r requirements.txt` |
| AdaLoRA rank not decreasing | Increase `NUM_EPOCHS` or lower `ADA_TARGET_R` in `src/model.py` |
