"""
model.py – Model loading, LoRA & AdaLoRA adapter creation for SmolVLM-256M
"""

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import (
    LoraConfig,
    AdaLoraConfig,
    get_peft_model,
    TaskType,
)


# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

# Target modules inside the language-model backbone of SmolVLM
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Approximate trainable-parameter budget shared by both adapters
LORA_R      = 16          # LoRA rank
LORA_ALPHA  = 32          # LoRA scaling factor
LORA_DROPOUT = 0.05

# AdaLoRA: initial rank = 2 × target rank so the pruning has room to work
ADA_INIT_R   = 32
ADA_TARGET_R = 16         # should match LORA_R for a fair comparison
ADA_DELTA_T  = 10         # update frequency (steps)
ADA_BETA1    = 0.85
ADA_BETA2    = 0.85


# ────────────────────────────────────────────────────────────────────────────
# Processor & base model loader
# ────────────────────────────────────────────────────────────────────────────

def load_processor():
    """Return the SmolVLM processor (tokenizer + image processor)."""
    print(f"Loading processor for {MODEL_ID} …")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return processor


def load_base_model(device: str = "auto"):
    """
    Load SmolVLM-256M in fp32 (or bf16 if CUDA is available).

    Parameters
    ----------
    device : str
        ``"auto"`` lets Accelerate decide; pass ``"cpu"`` to force CPU.
    """
    print(f"Loading base model {MODEL_ID} …")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=device,
    )
    # Disable cache during fine-tuning (required by gradient-checkpointing)
    model.config.use_cache = False
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ────────────────────────────────────────────────────────────────────────────
# LoRA adapter
# ────────────────────────────────────────────────────────────────────────────

def build_lora_model(base_model):
    """Wrap *base_model* with standard LoRA adapters and return the PEFT model."""
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(base_model, config)
    model.print_trainable_parameters()
    return model


# ────────────────────────────────────────────────────────────────────────────
# AdaLoRA adapter
# ────────────────────────────────────────────────────────────────────────────

def build_adalora_model(base_model):
    """Wrap *base_model* with AdaLoRA adapters and return the PEFT model."""
    config = AdaLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        init_r=ADA_INIT_R,
        target_r=ADA_TARGET_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        deltaT=ADA_DELTA_T,
        beta1=ADA_BETA1,
        beta2=ADA_BETA2,
        orth_reg_weight=0.5,
    )
    model = get_peft_model(base_model, config)
    model.print_trainable_parameters()
    return model


# ────────────────────────────────────────────────────────────────────────────
# Utility: count trainable parameters
# ────────────────────────────────────────────────────────────────────────────

def count_trainable_parameters(model) -> dict:
    """Return a dict with total, trainable, and trainable-% counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total":      total,
        "trainable":  trainable,
        "pct":        100.0 * trainable / total,
    }
