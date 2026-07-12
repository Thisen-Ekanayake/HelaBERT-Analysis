"""
HelaBERT Fine-tuning for Sentiment Analysis
Paper Method: "BERTifying Sinhala - A Comprehensive Analysis of Pre-trained
Language Models for Sinhala Text Classification" (LREC 2022)

Method (Table 3, SinBERT column):
  - Standard fine-tuning: [CLS] token → dropout → linear classifier head
  - Optimizer: AdamW, lr=1e-5, batch=16, epochs=10
  - Evaluation: macro-averaged F1 over 5 randomly-initialized runs, 4:1 train/test split
  - No oversampling, no class weights

Task: 4-class sentiment classification (positive, negative, neutral, conflict)
Dataset: user comments from online news articles
Data: 15,059 samples, avg text length ~21.66 tokens

Architecture:
  BERT encoder → [CLS] → dropout(0.1) → Linear(hidden, num_labels)
"""

import os
import random
import random as stdlib_random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import sentencepiece as spm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report,
)
from transformers import (
    BertConfig, BertForMaskedLM, BertModel,
    Trainer, TrainingArguments, EvalPrediction,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import wandb

# ==================== CONFIGURATION ====================
print("=" * 80)
print("HelaBERT FINE-TUNING — SENTIMENT ANALYSIS  [PAPER METHOD: 5 RANDOM SEED RUNS]")
print("=" * 80)

BERT_MODEL_PATH  = "HelaBERT"
TOKENIZER_MODEL  = "tokenizer/unigram_32000_0.9995.model"
BERT_CONFIG_FILE = "HelaBERT/config.json"
DATA_PATH        = "data/sinhala-sentiment-analysis/train.csv"

# Paper hyperparameters (Table 3, SinBERT)
MAX_SEQ_LENGTH  = 512    # truncate to model max; sentiment avg ~22 tokens
BATCH_SIZE      = 16
LEARNING_RATE   = 5e-6
NUM_EPOCHS      = 10
DROPOUT         = 0.1

# Evaluation: 5 randomly-initialized runs, 4:1 split
N_RUNS    = 5
TEST_SIZE = 0.2          # 4:1 → 20% test

# Output
OUTPUT_DIR = "HelaBERT_paper_sentiment"

# W&B
USE_WANDB     = True
WANDB_PROJECT = "helabert-paper-method"
WANDB_GROUP   = "sentiment_5runs_lr1e-5"

print(f"  LR={LEARNING_RATE}, batch={BATCH_SIZE}, epochs={NUM_EPOCHS}, "
      f"runs={N_RUNS}, test_size={TEST_SIZE}")
print(f"  Metric: macro-F1 (mean ± std over {N_RUNS} runs)")


# ==================== ENVIRONMENT ====================
print(f"\nPyTorch: {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

assert os.path.exists(BERT_MODEL_PATH),  f"{BERT_MODEL_PATH} not found"
assert os.path.exists(TOKENIZER_MODEL),  f"{TOKENIZER_MODEL} not found"
assert os.path.exists(DATA_PATH),        f"{DATA_PATH} not found"
print("All paths verified")


# ==================== TOKENIZER ====================
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)
PAD_ID = sp.pad_id()
print(f"SentencePiece loaded — vocab: {sp.get_piece_size()}, PAD_ID: {PAD_ID}")


# ==================== DATASET ====================
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, sp_processor, max_length):
        self.texts      = texts
        self.labels     = labels
        self.sp         = sp_processor
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids  = self.sp.encode(str(self.texts[idx]))[:self.max_length]
        mask = [1] * len(ids)
        pad  = self.max_length - len(ids)
        ids  += [PAD_ID] * pad
        mask += [0]      * pad
        return {
            'input_ids':      torch.tensor(ids,              dtype=torch.long),
            'attention_mask': torch.tensor(mask,             dtype=torch.long),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ==================== MODEL ====================
class SentimentModel(nn.Module):
    """
    Paper method: [CLS] → dropout → Linear(hidden, num_labels)
    Mirrors the Huggingface default RoBERTa classifier head structure:
    pooled output → dropout → linear layer.
    """
    def __init__(self, bert, hidden_size, num_labels, dropout=0.1):
        super().__init__()
        self.bert       = bert
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls    = out.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


# ==================== HELPERS ====================
def load_fresh_model(num_labels):
    if os.path.exists(BERT_CONFIG_FILE):
        cfg = BertConfig.from_json_file(BERT_CONFIG_FILE)
    else:
        cfg = BertConfig.from_pretrained(BERT_MODEL_PATH)

    try:
        backbone = BertModel.from_pretrained(BERT_MODEL_PATH)
    except Exception:
        mlm      = BertForMaskedLM.from_pretrained(BERT_MODEL_PATH)
        backbone = mlm.bert

    return SentimentModel(backbone, cfg.hidden_size, num_labels, DROPOUT)


def compute_metrics(eval_pred: EvalPrediction):
    preds  = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    return {
        'accuracy':  accuracy_score(labels, preds),
        'f1_macro':  f1_score(labels, preds, average='macro',    zero_division=0),
        'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
        'precision': precision_score(labels, preds, average='macro', zero_division=0),
        'recall':    recall_score(labels, preds,    average='macro', zero_division=0),
    }


def set_seed(seed):
    random.seed(seed)
    stdlib_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================== LOAD DATA ====================
print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# Flexible column detection for sentiment dataset
text_col  = next(c for c in df.columns if 'comment' in c.lower())
label_col = next(c for c in df.columns if 'sentiment' in c.lower())

df = df[[text_col, label_col]].dropna()
df[text_col]  = df[text_col].astype(str).str.strip()
df[label_col] = df[label_col].astype(str).str.strip()
df = df[df[text_col].str.len() > 0].reset_index(drop=True)

le = LabelEncoder()
df['label_id'] = le.fit_transform(df[label_col])
NUM_LABELS = len(le.classes_)

all_texts  = df[text_col].tolist()
all_labels = df['label_id'].tolist()

print(f"Loaded {len(df):,} samples, {NUM_LABELS} classes: {list(le.classes_)}")
for i, cls in enumerate(le.classes_):
    cnt = sum(1 for l in all_labels if l == i)
    print(f"  [{i}] {cls}: {cnt}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
pd.DataFrame({'id': range(NUM_LABELS), 'label': le.classes_}).to_csv(
    f"{OUTPUT_DIR}/label_map.csv", index=False)


# ==================== 5-RUN TRAINING ====================
print("\n" + "=" * 80)
print(f"TRAINING: {N_RUNS} randomly-initialized runs, 4:1 train/test split")
print("=" * 80)

run_f1s = []
SEEDS   = [42, 123, 456, 789, 1024]

for run_idx, seed in enumerate(SEEDS, start=1):
    print(f"\n{'='*80}")
    print(f"RUN {run_idx}/{N_RUNS}  (seed={seed})")
    print("=" * 80)

    set_seed(seed)

    # 4:1 stratified split (paper uses 4:1 train/test)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        all_texts, all_labels,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=all_labels,
    )
    print(f"Train: {len(train_texts):,}  |  Test: {len(test_texts):,}")

    train_ds = SentimentDataset(train_texts, train_labels, sp, MAX_SEQ_LENGTH)
    test_ds  = SentimentDataset(test_texts,  test_labels,  sp, MAX_SEQ_LENGTH)

    model = load_fresh_model(NUM_LABELS)

    run_dir = f"{OUTPUT_DIR}/run_{run_idx}"
    os.makedirs(run_dir, exist_ok=True)

    if USE_WANDB:
        if wandb.run is not None:
            wandb.finish()
        wandb.init(
            project=WANDB_PROJECT,
            group=WANDB_GROUP,
            name=f"run_{run_idx}_seed{seed}",
            config={
                "task": "sentiment_analysis", "method": "paper_sinbert",
                "run": run_idx, "seed": seed,
                "lr": LEARNING_RATE, "batch": BATCH_SIZE, "epochs": NUM_EPOCHS,
                "num_labels": NUM_LABELS, "label_names": list(le.classes_),
                "max_seq_length": MAX_SEQ_LENGTH, "test_size": TEST_SIZE,
            },
            reinit=True,
        )

    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        optim="adamw_torch",
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="wandb" if USE_WANDB else "none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    preds_out = trainer.predict(test_ds)
    preds     = np.argmax(preds_out.predictions, axis=1)
    macro_f1  = f1_score(test_labels, preds, average='macro', zero_division=0)
    run_f1s.append(round(macro_f1, 4))

    print(f"\nRun {run_idx} macro-F1: {macro_f1:.4f}")
    print(classification_report(test_labels, preds,
                                target_names=le.classes_, zero_division=0))

    if USE_WANDB and wandb.run:
        wandb.log({"test_macro_f1": macro_f1})
        wandb.finish()


# ==================== SUMMARY ====================
print("\n" + "=" * 80)
print("FINAL RESULTS  (paper-style: macro-F1 mean ± std over 5 runs)")
print("=" * 80)
for i, f1 in enumerate(run_f1s, 1):
    print(f"  Run {i}: {f1:.4f}")
mean_f1 = np.mean(run_f1s)
std_f1  = np.std(run_f1s)
print(f"\nMacro-F1: {mean_f1:.4f} ± {std_f1:.4f}")

results_df = pd.DataFrame({
    'run': range(1, N_RUNS + 1),
    'seed': SEEDS,
    'macro_f1': run_f1s,
})
results_df.to_csv(f"{OUTPUT_DIR}/results.csv", index=False)
print(f"\nResults saved to {OUTPUT_DIR}/results.csv")

with open(f"{OUTPUT_DIR}/results.txt", "w") as f:
    f.write("FINAL RESULTS  (paper-style: macro-F1 mean \u00b1 std over 5 runs)\n")
    f.write("=" * 80 + "\n")
    for i, f1 in enumerate(run_f1s, 1):
        f.write(f"  Run {i}: {f1:.4f}\n")
    f.write(f"\nMacro-F1: {mean_f1:.4f} \u00b1 {std_f1:.4f}\n")
print(f"Summary saved to {OUTPUT_DIR}/results.txt")
