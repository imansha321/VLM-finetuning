"""Generate the VLM_finetuning Jupyter notebook."""
import json, os, uuid

def uid(): return uuid.uuid4().hex[:8]
def md(src): return {"cell_type":"markdown","id":uid(),"metadata":{},"source":src}
def code(src): return {"cell_type":"code","id":uid(),"metadata":{},"source":src,"outputs":[],"execution_count":None}

CELLS = []

# ── Title ──────────────────────────────────────────────────────────────────
CELLS.append(md(
    "# VLM Fine-Tuning: LoRA vs AdaLoRA on DocVQA\n\n"
    "**Model**: SmolVLM-256M  \n"
    "**Task**: Visual Question Answering (VQA)  \n"
    "**Dataset**: DocVQA  \n"
    "**Research Question**: Does AdaLoRA improve parameter-efficient fine-tuning "
    "of SmolVLM-256M compared with standard LoRA under a similar parameter budget?"
))

# ── 1. Install ─────────────────────────────────────────────────────────────
CELLS.append(md("## 1 — Install Dependencies"))
CELLS.append(code(
    "!pip install torch torchvision transformers peft datasets accelerate "
    "Pillow evaluate pandas matplotlib tqdm scikit-learn -q"
))

# ── 2. Imports ─────────────────────────────────────────────────────────────
CELLS.append(md("## 2 — Imports & Reproducibility"))
CELLS.append(code(
    "import os, sys, random, json, re\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.ticker as mticker\n"
    "import torch\n"
    "from tqdm import tqdm\n"
    "from PIL import Image\n"
    "from datasets import load_dataset\n"
    "from transformers import (\n"
    "    AutoProcessor,\n"
    "    AutoModelForVision2Seq,\n"
    "    get_linear_schedule_with_warmup,\n"
    ")\n"
    "from peft import (\n"
    "    LoraConfig, AdaLoraConfig,\n"
    "    get_peft_model, PeftModel, TaskType,\n"
    ")\n"
    "from torch.utils.data import Dataset, DataLoader\n"
    "\n"
    "SEED = 42\n"
    "random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)\n"
    "if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)\n"
    "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    "print('Device:', DEVICE)"
))

# ── 3. Load DocVQA ─────────────────────────────────────────────────────────
CELLS.append(md("## 3 — Load DocVQA"))
CELLS.append(code(
    "TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 2000, 500, 500\n"
    "\n"
    "ds = load_dataset('HuggingFaceM4/DocumentVQA', trust_remote_code=True)\n"
    "train_hf = ds['train'].shuffle(seed=SEED).select(range(TRAIN_SIZE))\n"
    "val_hf   = ds['validation'].shuffle(seed=SEED).select(range(VAL_SIZE))\n"
    "test_hf  = ds['test'].shuffle(seed=SEED).select(range(min(TEST_SIZE, len(ds['test']))))\n"
    "\n"
    "print('Train:', len(train_hf), '| Val:', len(val_hf), '| Test:', len(test_hf))\n"
    "\n"
    "# Quick preview\n"
    "sample = train_hf[0]\n"
    "print('Question:', sample['question'])\n"
    "print('Answer  :', sample['answers'][0])\n"
    "sample['image']"
))

# ── 4. Dataset class ───────────────────────────────────────────────────────
CELLS.append(md("## 4 — Dataset Class"))
CELLS.append(code(
    "class DocVQADataset(Dataset):\n"
    "    def __init__(self, hf_dataset, processor, max_length=256):\n"
    "        self.data, self.processor, self.max_length = hf_dataset, processor, max_length\n"
    "\n"
    "    def __len__(self): return len(self.data)\n"
    "\n"
    "    def __getitem__(self, idx):\n"
    "        item     = self.data[idx]\n"
    "        image    = item['image']\n"
    "        question = item['question']\n"
    "        answers  = item.get('answers') or item.get('answer') or ['']\n"
    "        answer   = answers[0] if isinstance(answers, list) else answers\n"
    "        prompt   = f'Question: {question}\\nAnswer:'\n"
    "\n"
    "        enc = self.processor(\n"
    "            images=image, text=prompt,\n"
    "            return_tensors='pt', padding='max_length',\n"
    "            truncation=True, max_length=self.max_length)\n"
    "        lbl = self.processor.tokenizer(\n"
    "            answer, return_tensors='pt',\n"
    "            padding='max_length', truncation=True, max_length=64)\n"
    "        return {\n"
    "            'input_ids':      enc['input_ids'].squeeze(0),\n"
    "            'attention_mask': enc['attention_mask'].squeeze(0),\n"
    "            'pixel_values':   enc['pixel_values'].squeeze(0),\n"
    "            'labels':         lbl['input_ids'].squeeze(0),\n"
    "            'answer_text':    answer,\n"
    "            'question_text':  question,\n"
    "        }\n"
    "\n"
    "def collate_fn(batch):\n"
    "    keys = ['input_ids','attention_mask','pixel_values','labels']\n"
    "    out  = {k: torch.stack([b[k] for b in batch]) for k in keys}\n"
    "    out['answer_text']   = [b['answer_text']   for b in batch]\n"
    "    out['question_text'] = [b['question_text'] for b in batch]\n"
    "    return out"
))

# ── 5. Load model ──────────────────────────────────────────────────────────
CELLS.append(md("## 5 — Load SmolVLM-256M"))
CELLS.append(code(
    "MODEL_ID  = 'HuggingFaceTB/SmolVLM-256M-Instruct'\n"
    "DTYPE     = torch.bfloat16 if torch.cuda.is_available() else torch.float32\n"
    "processor = AutoProcessor.from_pretrained(MODEL_ID)\n"
    "\n"
    "def fresh_model():\n"
    "    m = AutoModelForVision2Seq.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map='auto')\n"
    "    m.config.use_cache = False\n"
    "    return m\n"
    "\n"
    "print('Processor loaded.')"
))

# ── 6. LoRA config ─────────────────────────────────────────────────────────
CELLS.append(md("## 6 — LoRA Configuration"))
CELLS.append(code(
    "TARGET_MODULES = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']\n"
    "LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05\n"
    "\n"
    "def build_lora():\n"
    "    base = fresh_model()\n"
    "    cfg  = LoraConfig(\n"
    "        task_type=TaskType.CAUSAL_LM, r=LORA_R,\n"
    "        lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,\n"
    "        target_modules=TARGET_MODULES, bias='none')\n"
    "    model = get_peft_model(base, cfg)\n"
    "    model.print_trainable_parameters()\n"
    "    return model\n"
    "\n"
    "lora_model = build_lora()"
))

# ── 7. AdaLoRA config ──────────────────────────────────────────────────────
CELLS.append(md("## 7 — AdaLoRA Configuration"))
CELLS.append(code(
    "ADA_INIT_R, ADA_TARGET_R = 32, 16\n"
    "\n"
    "def build_adalora():\n"
    "    base = fresh_model()\n"
    "    cfg  = AdaLoraConfig(\n"
    "        task_type=TaskType.CAUSAL_LM, init_r=ADA_INIT_R,\n"
    "        target_r=ADA_TARGET_R, lora_alpha=LORA_ALPHA,\n"
    "        lora_dropout=LORA_DROPOUT, target_modules=TARGET_MODULES,\n"
    "        deltaT=10, beta1=0.85, beta2=0.85, orth_reg_weight=0.5)\n"
    "    model = get_peft_model(base, cfg)\n"
    "    model.print_trainable_parameters()\n"
    "    return model\n"
    "\n"
    "adalora_model = build_adalora()"
))

# ── 8. Training helper ─────────────────────────────────────────────────────
CELLS.append(md("## 8 — Training Helper"))
CELLS.append(code(
    "BATCH_SIZE    = 4\n"
    "LR            = 2e-4\n"
    "NUM_EPOCHS    = 3\n"
    "WARMUP_RATIO  = 0.1\n"
    "MAX_GRAD_NORM = 1.0\n"
    "\n"
    "def train_model(model, name='lora'):\n"
    "    train_ds = DocVQADataset(train_hf, processor)\n"
    "    val_ds   = DocVQADataset(val_hf,   processor)\n"
    "    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)\n"
    "    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)\n"
    "\n"
    "    optimizer = torch.optim.AdamW(\n"
    "        filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)\n"
    "    total_steps = NUM_EPOCHS * len(train_dl)\n"
    "    scheduler   = get_linear_schedule_with_warmup(\n"
    "        optimizer, int(WARMUP_RATIO * total_steps), total_steps)\n"
    "\n"
    "    history = {'train_loss': [], 'val_loss': []}\n"
    "\n"
    "    for epoch in range(1, NUM_EPOCHS + 1):\n"
    "        model.train(); t_loss = 0.0\n"
    "        for step, batch in enumerate(tqdm(train_dl, desc=f'[{name}] Epoch {epoch} train')):\n"
    "            ids  = batch['input_ids'].to(DEVICE)\n"
    "            mask = batch['attention_mask'].to(DEVICE)\n"
    "            pix  = batch['pixel_values'].to(DEVICE)\n"
    "            lbl  = batch['labels'].to(DEVICE); lbl[lbl == 0] = -100\n"
    "            out  = model(input_ids=ids, attention_mask=mask, pixel_values=pix, labels=lbl)\n"
    "            out.loss.backward()\n"
    "            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)\n"
    "            optimizer.step(); scheduler.step(); optimizer.zero_grad()\n"
    "            if hasattr(model, 'update_and_allocate'): model.update_and_allocate(step)\n"
    "            t_loss += out.loss.item()\n"
    "\n"
    "        model.eval(); v_loss = 0.0\n"
    "        with torch.no_grad():\n"
    "            for batch in tqdm(val_dl, desc=f'[{name}] Epoch {epoch} val'):\n"
    "                ids  = batch['input_ids'].to(DEVICE)\n"
    "                mask = batch['attention_mask'].to(DEVICE)\n"
    "                pix  = batch['pixel_values'].to(DEVICE)\n"
    "                lbl  = batch['labels'].to(DEVICE); lbl[lbl == 0] = -100\n"
    "                v_loss += model(\n"
    "                    input_ids=ids, attention_mask=mask,\n"
    "                    pixel_values=pix, labels=lbl).loss.item()\n"
    "\n"
    "        tl = t_loss / len(train_dl); vl = v_loss / len(val_dl)\n"
    "        history['train_loss'].append(tl); history['val_loss'].append(vl)\n"
    "        print(f'  Epoch {epoch} | train_loss={tl:.4f} | val_loss={vl:.4f}')\n"
    "\n"
    "        ckpt = f'checkpoints/{name}'\n"
    "        os.makedirs(ckpt, exist_ok=True)\n"
    "        model.save_pretrained(ckpt); processor.save_pretrained(ckpt)\n"
    "\n"
    "    return history"
))

# ── 9. Train LoRA ──────────────────────────────────────────────────────────
CELLS.append(md("## 9 — Train LoRA Baseline"))
CELLS.append(code("lora_history = train_model(lora_model, name='lora')"))

# ── 10. Train AdaLoRA ──────────────────────────────────────────────────────
CELLS.append(md("## 10 — Train AdaLoRA"))
CELLS.append(code("adalora_history = train_model(adalora_model, name='adalora')"))

# ── 11. Eval helper ────────────────────────────────────────────────────────
CELLS.append(md("## 11 — Evaluation Helper"))
CELLS.append(code(
    "def normalise(text):\n"
    "    text = text.lower().strip()\n"
    "    text = re.sub(r'[^a-z0-9\\s]', ' ', text)\n"
    "    return ' '.join(text.split())\n"
    "\n"
    "def classify_question(q):\n"
    "    q = q.lower()\n"
    "    if re.search(r'how many|how much|number|amount|total|count|year|date|price|cost', q):\n"
    "        return 'Numerical'\n"
    "    if re.search(r'where|location|position|address|table|column|row|section', q):\n"
    "        return 'Layout'\n"
    "    return 'Text/OCR'\n"
    "\n"
    "@torch.no_grad()\n"
    "def evaluate(model, hf_data):\n"
    "    model.eval(); records = []\n"
    "    for item in tqdm(hf_data, desc='Evaluating'):\n"
    "        image    = item['image']\n"
    "        question = item['question']\n"
    "        answers  = item.get('answers') or item.get('answer') or ['']\n"
    "        gt       = answers[0] if isinstance(answers, list) else answers\n"
    "        prompt   = f'Question: {question}\\nAnswer:'\n"
    "        enc      = processor(\n"
    "            images=image, text=prompt, return_tensors='pt',\n"
    "            padding=True, truncation=True, max_length=256).to(DEVICE)\n"
    "        gen_ids  = model.generate(**enc, max_new_tokens=64, do_sample=False)\n"
    "        gen_ids  = gen_ids[:, enc['input_ids'].shape[-1]:]\n"
    "        pred     = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()\n"
    "        score    = float(normalise(pred) == normalise(gt))\n"
    "        records.append({'question': question, 'ground_truth': gt,\n"
    "                        'prediction': pred, 'score': score,\n"
    "                        'q_type': classify_question(question)})\n"
    "    return pd.DataFrame(records)"
))

# ── 12. Eval LoRA ──────────────────────────────────────────────────────────
CELLS.append(md("## 12 — Evaluate LoRA"))
CELLS.append(code(
    "lora_df = evaluate(lora_model, test_hf)\n"
    "print('LoRA overall VQA score:', round(lora_df['score'].mean(), 4))"
))

# ── 13. Eval AdaLoRA ───────────────────────────────────────────────────────
CELLS.append(md("## 13 — Evaluate AdaLoRA"))
CELLS.append(code(
    "adalora_df = evaluate(adalora_model, test_hf)\n"
    "print('AdaLoRA overall VQA score:', round(adalora_df['score'].mean(), 4))"
))

# ── 14. Results table ──────────────────────────────────────────────────────
CELLS.append(md("## 14 — Results Table"))
CELLS.append(code(
    "rows = []\n"
    "for qt in ['Text/OCR', 'Numerical', 'Layout']:\n"
    "    ls  = lora_df[lora_df['q_type'] == qt]['score'].mean()\n"
    "    als = adalora_df[adalora_df['q_type'] == qt]['score'].mean()\n"
    "    rows.append({'Question Type': qt,\n"
    "                 'LoRA': round(ls, 4), 'AdaLoRA': round(als, 4),\n"
    "                 'Difference': round(als - ls, 4)})\n"
    "rows.append({'Question Type': 'Overall',\n"
    "             'LoRA':    round(lora_df['score'].mean(), 4),\n"
    "             'AdaLoRA': round(adalora_df['score'].mean(), 4),\n"
    "             'Difference': round(adalora_df['score'].mean() - lora_df['score'].mean(), 4)})\n"
    "results_table = pd.DataFrame(rows)\n"
    "os.makedirs('results', exist_ok=True)\n"
    "results_table.to_csv('results/results.csv', index=False)\n"
    "results_table"
))

# ── 15. Training curves ────────────────────────────────────────────────────
CELLS.append(md("## 15 — Training Curves"))
CELLS.append(code(
    "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
    "epochs = range(1, NUM_EPOCHS + 1)\n"
    "\n"
    "ax = axes[0]\n"
    "ax.plot(epochs, lora_history['train_loss'],    'b-o',  label='LoRA train')\n"
    "ax.plot(epochs, lora_history['val_loss'],      'b--o', label='LoRA val')\n"
    "ax.plot(epochs, adalora_history['train_loss'], 'r-s',  label='AdaLoRA train')\n"
    "ax.plot(epochs, adalora_history['val_loss'],   'r--s', label='AdaLoRA val')\n"
    "ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')\n"
    "ax.set_title('Training and Validation Loss'); ax.legend()\n"
    "\n"
    "ax = axes[1]\n"
    "q_types = ['Text/OCR', 'Numerical', 'Layout', 'Overall']\n"
    "x = np.arange(len(q_types)); w = 0.35\n"
    "ax.bar(x - w/2,\n"
    "    [results_table.loc[results_table['Question Type'] == qt, 'LoRA'].values[0]    for qt in q_types],\n"
    "    w, label='LoRA')\n"
    "ax.bar(x + w/2,\n"
    "    [results_table.loc[results_table['Question Type'] == qt, 'AdaLoRA'].values[0] for qt in q_types],\n"
    "    w, label='AdaLoRA')\n"
    "ax.set_xticks(x); ax.set_xticklabels(q_types, rotation=15)\n"
    "ax.set_ylabel('VQA Score'); ax.set_title('VQA Score by Question Type'); ax.legend()\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/training_curves.png', dpi=150)\n"
    "plt.show()\n"
    "print('Saved -> results/training_curves.png')"
))

# ── 16. Rank analysis ──────────────────────────────────────────────────────
CELLS.append(md("## 16 — AdaLoRA Rank Analysis"))
CELLS.append(code(
    "ranks = {}\n"
    "for name, param in adalora_model.named_parameters():\n"
    "    if 'lora_E' in name and param.requires_grad:\n"
    "        eff = (param.abs() > 1e-6).sum().item()\n"
    "        clean = name.replace('.lora_E', '').replace('base_model.model.', '')\n"
    "        ranks[clean] = int(eff)\n"
    "\n"
    "print('Total AdaLoRA modules:', len(ranks))\n"
    "rank_df = pd.DataFrame(ranks.items(), columns=['Module', 'Effective Rank'])\n"
    "rank_df = rank_df.sort_values('Effective Rank', ascending=False)\n"
    "print(rank_df.head(15).to_string(index=False))"
))
CELLS.append(code(
    "fig, ax = plt.subplots(figsize=(14, 5))\n"
    "modules = list(ranks.keys())[:30]\n"
    "values  = [ranks[m] for m in modules]\n"
    "ax.bar(range(len(modules)), values, color='steelblue', edgecolor='black', linewidth=0.5)\n"
    "ax.set_xticks(range(len(modules)))\n"
    "ax.set_xticklabels(modules, rotation=90, fontsize=7)\n"
    "ax.set_ylabel('Effective Rank')\n"
    "ax.set_title('AdaLoRA - Final Rank Allocation per Module')\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/rank_analysis.png', dpi=150)\n"
    "plt.show()\n"
    "print('Saved -> results/rank_analysis.png')"
))

# ── 17. Summary ────────────────────────────────────────────────────────────
CELLS.append(md("## 17 — Experiment Summary"))
CELLS.append(code(
    "print('=' * 60)\n"
    "print('  EXPERIMENT SUMMARY')\n"
    "print('=' * 60)\n"
    "print('  Model   : SmolVLM-256M')\n"
    "print('  Dataset : DocVQA', TRAIN_SIZE, 'train /', VAL_SIZE, 'val /', TEST_SIZE, 'test')\n"
    "print('  Epochs  :', NUM_EPOCHS, ' | Batch size:', BATCH_SIZE, ' | LR:', LR)\n"
    "print('  SEED    :', SEED)\n"
    "print()\n"
    "print(results_table.to_string(index=False))\n"
    "print()\n"
    "diff = results_table.loc[results_table['Question Type'] == 'Overall', 'Difference'].values[0]\n"
    "if diff > 0:\n"
    "    print('CONCLUSION: AdaLoRA outperforms LoRA by', round(diff, 4), '(supports hypothesis)')\n"
    "elif diff < 0:\n"
    "    print('CONCLUSION: LoRA outperforms AdaLoRA by', round(abs(diff), 4))\n"
    "else:\n"
    "    print('CONCLUSION: Both methods perform equally')"
))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": CELLS,
}

os.makedirs("notebooks", exist_ok=True)
with open("notebooks/VLM_finetuning.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook written to notebooks/VLM_finetuning.ipynb")
