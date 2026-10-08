# VLM_finetuning

## LoRA Finetuning for Vision-Language Models

This project investigates whether the main idea behind **AdaLoRA (Adaptive Low-Rank Adaptation)** transfers from language models to the vision-language domain.

The experiment uses **SmolVLM-256M** and compares **AdaLoRA** with standard **LoRA** on a Visual Question Answering (VQA) task.

---

## Project Overview

The main research question is:

> **Does AdaLoRA improve parameter-efficient fine-tuning of a small vision-language model compared with standard LoRA under a similar trainable-parameter budget?**

### Experiment

```text
                 SmolVLM-256M
                       │
              ┌────────┴────────┐
              │                 │
           LoRA              AdaLoRA
              │                 │
              └────────┬────────┘
                       │
                      VQA
                       │
                  Comparison
```

---

## Repository Structure

```text
VLM_finetuning/
│
├── README.md
│
├── notebooks/
│   └── VLM_finetuning.ipynb
│
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── results/
│   ├── results.csv
│   ├── training_curves.png
│   └── rank_analysis.png
│
└── requirements.txt
```

For the assignment, the main implementation can also be provided as a single Google Colab notebook.

---

# 1. LoRA Variant

## AdaLoRA

The selected LoRA variant is **AdaLoRA**.

AdaLoRA extends standard LoRA by dynamically allocating the available low-rank parameter budget between different layers.

### Standard LoRA

Standard LoRA normally uses a fixed rank:

```text
Layer 1 → rank 8
Layer 2 → rank 8
Layer 3 → rank 8
Layer 4 → rank 8
```

### AdaLoRA

AdaLoRA can allocate different ranks:

```text
Layer 1 → rank 4
Layer 2 → rank 12
Layer 3 → rank 2
Layer 4 → rank 14
```

The idea is that different layers may have different importance for the downstream task.

---

# 2. Vision-Language Model

## SmolVLM-256M

The selected VLM is:

**SmolVLM-256M**

The model accepts both an image and text input:

```text
Image + Question
       ↓
 SmolVLM-256M
       ↓
     Answer
```

---

# 3. Research Hypothesis

The central concept taken from AdaLoRA is:

> Different model components do not contribute equally to adaptation, so the available parameter budget should be allocated according to the importance of different layers.

This is translated into the following vision-language hypothesis:

> **AdaLoRA will achieve better VQA performance than standard LoRA under a similar trainable-parameter budget because it can allocate more adaptation capacity to the VLM components that are important for the VQA task.**

---

# 4. Dataset

## DocVQA

The selected dataset is **DocVQA**.

DocVQA contains document images together with questions and answers.

Example:

```text
Document Image
      +
"What is the invoice number?"
      ↓
"INV-10245"
```

---

## Dataset Subset

Initial target:

```text
Training:    2,000 examples
Validation:    500 examples
Testing:      500 examples
```

The same dataset subset will be used for both LoRA and AdaLoRA.

If GPU memory or training time becomes a limitation, the number of examples can be reduced while keeping the same subset for both experiments.

---

# 5. Experimental Setup

Two main fine-tuning experiments will be performed.

## Experiment 1 – LoRA

The pretrained SmolVLM-256M is loaded and standard LoRA adapters are added.

```text
SmolVLM-256M
     │
     ├── Frozen weights
     │
     └── LoRA adapters
             │
             ↓
          Training
```

---

## Experiment 2 – AdaLoRA

The same pretrained SmolVLM-256M is used with AdaLoRA.

```text
SmolVLM-256M
     │
     ├── Frozen weights
     │
     └── AdaLoRA adapters
             │
             ↓
    Adaptive rank allocation
             │
             ↓
          Training
```

---

# 6. Fair Comparison

To make the comparison meaningful, both experiments will use the same:

* pretrained model
* training dataset
* validation dataset
* test dataset
* training procedure
* batch size
* number of training steps
* learning-rate schedule
* target modules
* approximate trainable parameter budget

The main difference is:

```text
LoRA     → Fixed rank allocation
AdaLoRA  → Adaptive rank allocation
```

---

# 7. Evaluation

The experiment will evaluate more than just the overall VQA score.

## Overall Performance

The main comparison will be:

| Model                  | VQA Score |
| ---------------------- | --------: |
| SmolVLM-256M           |       TBD |
| SmolVLM-256M + LoRA    |       TBD |
| SmolVLM-256M + AdaLoRA |       TBD |

---

## Question-Type Analysis

Where possible, the test examples will be grouped according to question characteristics.

For example:

| Question Type | LoRA | AdaLoRA | Difference |
| ------------- | ---: | ------: | ---------: |
| Text/OCR      |  TBD |     TBD |        TBD |
| Numerical     |  TBD |     TBD |        TBD |
| Layout        |  TBD |     TBD |        TBD |
| Other         |  TBD |     TBD |        TBD |
| Overall       |  TBD |     TBD |        TBD |

This helps determine whether AdaLoRA provides improvements for particular visual-language capabilities rather than only measuring the overall score.

---

# 8. AdaLoRA Rank Analysis

An additional analysis will investigate how AdaLoRA distributes its parameter budget.

For example:

```text
Module                    Final Rank
-------------------------------------
Layer 1                       4
Layer 2                      12
Layer 3                       6
Layer 4                      14
...
```

The final rank allocation can then be compared with the performance of different question types.

This provides a more specific interpretation of the AdaLoRA concept in the vision-language setting.

---

# 9. Expected Outcomes

The experiment does not assume that AdaLoRA will definitely outperform LoRA.

Possible results include:

### AdaLoRA performs better

```text
AdaLoRA > LoRA
```

This would support the hypothesis that adaptive parameter allocation is useful for VLM fine-tuning.

### Similar performance

```text
AdaLoRA ≈ LoRA
```

This would suggest that adaptive allocation provides limited benefit for this particular VLM and dataset.

### LoRA performs better

```text
AdaLoRA < LoRA
```

This would suggest that the AdaLoRA advantage does not necessarily transfer to this particular vision-language setting.

All three outcomes are useful for testing the research hypothesis.

---

# 10. Technologies

The project uses:

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face PEFT
* Hugging Face Datasets
* Google Colab

Main libraries:

```text
torch
transformers
peft
datasets
accelerate
Pillow
```

---

# 11. Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/imansha321/VLM-finetuning
cd VLM-finetuning
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

```text
notebooks/VLM_finetuning.ipynb
```

The notebook contains the complete pipeline:

```text
Install dependencies
        ↓
Load DocVQA
        ↓
Select dataset subset
        ↓
Load SmolVLM-256M
        ↓
Train LoRA baseline
        ↓
Evaluate LoRA
        ↓
Train AdaLoRA
        ↓
Evaluate AdaLoRA
        ↓
Compare results
        ↓
Analyze AdaLoRA rank allocation
```

---

# 12. Reproducibility

A fixed random seed will be used where possible:

```python
SEED = 42
```

The following will also be recorded:

* dataset split
* number of examples
* model checkpoint
* LoRA configuration
* AdaLoRA configuration
* learning rate
* batch size
* number of epochs/steps
* random seed
* evaluation settings

This allows the main experimental result to be reproduced.

---

# 14. Research Question

The final research question is:

> **Does adaptive rank allocation in AdaLoRA improve parameter-efficient fine-tuning of SmolVLM-256M for Visual Question Answering compared with standard LoRA under a similar parameter budget?**

---

## Summary

| Component           | Selection                                |
| ------------------- | ---------------------------------------- |
| Assignment          | EN4554 – Assignment 1 Part 1             |
| LoRA Variant        | AdaLoRA                                  |
| VLM                 | SmolVLM-256M                             |
| Task                | Visual Question Answering                |
| Dataset             | DocVQA                                   |
| Baseline            | Standard LoRA                            |
| Evaluation          | VQA performance + question-type analysis |
| Additional Analysis | AdaLoRA rank allocation                  |
| Platform            | Google Colab                             |
| Repository          | `VLM_finetuning`                         |
