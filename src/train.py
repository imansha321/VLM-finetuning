"""
train.py – Training loop for LoRA and AdaLoRA experiments on DocVQA
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.data  import load_docvqa, DocVQADataset
from src.model import load_processor, load_base_model, build_lora_model, build_adalora_model


# ────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ────────────────────────────────────────────────────────────────────────────
SEED           = 42
BATCH_SIZE     = 4
LEARNING_RATE  = 2e-4
NUM_EPOCHS     = 3
WARMUP_RATIO   = 0.1
MAX_GRAD_NORM  = 1.0
MAX_SEQ_LEN    = 256
SAVE_DIR       = "checkpoints"


# ────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ────────────────────────────────────────────────────────────────────────────
# Collate function
# ────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """Stack tensor fields; keep text fields as lists."""
    keys = ["input_ids", "attention_mask", "pixel_values", "labels"]
    out  = {k: torch.stack([item[k] for item in batch]) for k in keys}
    out["answer_text"]   = [item["answer_text"]   for item in batch]
    out["question_text"] = [item["question_text"] for item in batch]
    return out


# ────────────────────────────────────────────────────────────────────────────
# Training step
# ────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for step, batch in enumerate(pbar):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values   = batch["pixel_values"].to(device)
        labels         = batch["labels"].to(device)

        # Replace padding token id in labels with -100 (ignored by CE loss)
        labels[labels == 0] = -100

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # AdaLoRA: update the singular-value masks
        if hasattr(model, "update_and_allocate"):
            model.update_and_allocate(step)

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


# ────────────────────────────────────────────────────────────────────────────
# Validation step
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Validating", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values   = batch["pixel_values"].to(device)
        labels         = batch["labels"].to(device)
        labels[labels == 0] = -100

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
        )
        total_loss += outputs.loss.item()

    return total_loss / len(loader)


# ────────────────────────────────────────────────────────────────────────────
# Main training entry-point
# ────────────────────────────────────────────────────────────────────────────

def train(adapter: str = "lora"):
    """
    Run one complete training experiment.

    Parameters
    ----------
    adapter : str
        ``"lora"`` or ``"adalora"``
    """
    assert adapter in ("lora", "adalora"), "adapter must be 'lora' or 'adalora'"

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Adapter  : {adapter.upper()}")
    print(f"  Device   : {device}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    processor = load_processor()
    train_hf, val_hf, _ = load_docvqa()

    train_ds = DocVQADataset(train_hf, processor, max_length=MAX_SEQ_LEN)
    val_ds   = DocVQADataset(val_hf,   processor, max_length=MAX_SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  collate_fn=collate_fn,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    base_model = load_base_model(device="auto")
    if adapter == "lora":
        model = build_lora_model(base_model)
    else:
        model = build_adalora_model(base_model)

    model.to(device)

    # ── Optimiser & scheduler ─────────────────────────────────────────────
    optimizer  = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )
    total_steps   = NUM_EPOCHS * len(train_loader)
    warmup_steps  = int(WARMUP_RATIO * total_steps)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ── Training loop ─────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     scheduler, device, epoch)
        val_loss   = validate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(SAVE_DIR, adapter)
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"  ✓ Checkpoint saved → {save_path}")

    print(f"\nTraining complete. Best val_loss = {best_val_loss:.4f}")
    return model, processor, history


# ────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", choices=["lora", "adalora"],
                        default="lora", help="Which adapter to train")
    args = parser.parse_args()
    train(adapter=args.adapter)
