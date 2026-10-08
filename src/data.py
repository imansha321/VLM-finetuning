"""
data.py – Dataset loading and preprocessing for DocVQA
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
import io


# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────
SEED = 42

TRAIN_SIZE = 2000
VAL_SIZE   = 500
TEST_SIZE  = 500


# ────────────────────────────────────────────────────────────────────────────
# Raw dataset loader
# ────────────────────────────────────────────────────────────────────────────

def load_docvqa(train_size: int = TRAIN_SIZE,
                val_size:   int = VAL_SIZE,
                test_size:  int = TEST_SIZE):
    """
    Load DocVQA from Hugging Face and return fixed-size subsets.

    Returns
    -------
    train_ds, val_ds, test_ds : HuggingFace Dataset objects
    """
    print("Loading DocVQA …")
    ds = load_dataset("HuggingFaceM4/DocumentVQA", trust_remote_code=True)

    train_ds = ds["train"].shuffle(seed=SEED).select(range(train_size))
    val_ds   = ds["validation"].shuffle(seed=SEED).select(range(val_size))
    test_ds  = ds["test"].shuffle(seed=SEED).select(range(min(test_size, len(ds["test"]))))

    print(f"  Train : {len(train_ds):,} examples")
    print(f"  Val   : {len(val_ds):,} examples")
    print(f"  Test  : {len(test_ds):,} examples")
    return train_ds, val_ds, test_ds


# ────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset wrapper
# ────────────────────────────────────────────────────────────────────────────

class DocVQADataset(Dataset):
    """
    Wraps a HuggingFace DocVQA split for use with a PyTorch DataLoader.

    Each item returns:
        image    – PIL.Image
        question – str
        answer   – str  (first answer in the list)
    """

    def __init__(self, hf_dataset, processor, max_length: int = 256):
        self.data       = hf_dataset
        self.processor  = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # ── image ──────────────────────────────────────────────────────────
        image = item["image"]
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = image.convert("RGB")

        # ── text ───────────────────────────────────────────────────────────
        question = item["question"]
        answers  = item.get("answers") or item.get("answer") or [""]
        answer   = answers[0] if isinstance(answers, list) else answers

        # ── processor ──────────────────────────────────────────────────────
        # SmolVLM uses a chat template; build the prompt string directly
        prompt = f"Question: {question}\nAnswer:"

        encoding = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        # Labels: encode the answer
        label_encoding = self.processor.tokenizer(
            answer,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "pixel_values":   encoding["pixel_values"].squeeze(0),
            "labels":         label_encoding["input_ids"].squeeze(0),
            "answer_text":    answer,
            "question_text":  question,
        }


# ────────────────────────────────────────────────────────────────────────────
# Quick-check helper
# ────────────────────────────────────────────────────────────────────────────

def check_dataset(dataset: DocVQADataset, n: int = 3):
    """Print a few samples for a sanity check."""
    print(f"\nDataset length: {len(dataset)}")
    for i in range(min(n, len(dataset))):
        item = dataset[i]
        print(f"\n[{i}]")
        print(f"  question : {item['question_text']}")
        print(f"  answer   : {item['answer_text']}")
        print(f"  ids shape: {item['input_ids'].shape}")
