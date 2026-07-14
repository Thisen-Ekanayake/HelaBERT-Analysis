"""
HelaBERT Fine-tuning for News Category Classification — WITH CO-ATTENTION
Paper Method: "BERTifying Sinhala - A Comprehensive Analysis of Pre-trained
Language Models for Sinhala Text Classification" (LREC 2022)

Co-Attention head (same design as sentiment_co_attn.py):
  BERT encoder
    cls_vec   (B, H)    = last_hidden_state[:, 0, :]
    token_seq (B, T, H) = last_hidden_state[:, 1:, :]
  CoAttention([CLS] ↔ token_seq) → concat[c; c̃; T̃] (raw CLS + attended) → MLP

All hyperparameters (lr, batch, epochs, dropout, etc.) are UNCHANGED from
the paper baseline (news_category.py).

Task: 5-class sentence classification
Classes: Business, Political, Entertainment, Science and Technology, Sports
Dataset: 3,327 samples, avg text length ~23.49 tokens
Architecture: HelaBERT_small (hidden=384) + co-attention head
"""

import os
import random
import random as stdlib_random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
print("HelaBERT FINE-TUNING — NEWS CATEGORY  [CO-ATTENTION | 5 SEED RUNS]")
print("=" * 80)

BERT_MODEL_PATH  = "HelaBERT_small"
TOKENIZER_MODEL  = "tokenizer/unigram_32000_0.9995.model"
BERT_CONFIG_FILE = "HelaBERT_small/config.json"
DATA_PATH        = "data/Sinhala-News-Category-classification/train/news_train.csv"

# Paper hyperparameters — UNCHANGED
MAX_SEQ_LENGTH = 512
BATCH_SIZE     = 16
LEARNING_RATE  = 3e-5
NUM_EPOCHS     = 10
DROPOUT        = 0.1
NUM_LABELS     = 5

N_RUNS    = 5
TEST_SIZE = 0.2

OUTPUT_DIR    = "HelaBERT_small_coattention_news_category_clsconcat"
USE_WANDB     = True
WANDB_PROJECT = "helabert-coattention"
WANDB_GROUP   = "news_category_small_5runs_coattn_clsconcat"

print(f"  LR={LEARNING_RATE}, batch={BATCH_SIZE}, epochs={NUM_EPOCHS}, "
      f"runs={N_RUNS}, test_size={TEST_SIZE}")
print(f"  Head: Co-Attention → concat[c; c̃; T̃] (raw CLS + attended CLS + attended tokens) → MLP")


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
class NewsCategoryDataset(Dataset):
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


# ==================== CO-ATTENTION MODULE ====================
class CoAttention(nn.Module):
    """
    Parallel co-attention between [CLS] and the token sequence.

    Direction 1 — [CLS] attends over tokens  → attended_cls    (B, H)
    Direction 2 — tokens attend over [CLS]   → attended_tokens (B, H)
      Uses sigmoid gate + renorm; fp16-safe -1e4 fill prevents NaN.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.W_cls   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_token = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v       = nn.Linear(hidden_size, 1,           bias=False)
        self.scale   = hidden_size ** 0.5

    @staticmethod
    def softmax_safe(logits, mask=None):
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e4)
        return F.softmax(logits, dim=-1)

    def forward(self, cls_vec, token_seq, key_mask=None):
        B, T, H = token_seq.shape

        cls_proj   = self.W_cls(cls_vec).unsqueeze(1).expand(-1, T, -1)
        token_proj = self.W_token(token_seq)
        affinity   = self.v(torch.tanh(cls_proj + token_proj)).squeeze(-1) / self.scale

        alpha        = self.softmax_safe(affinity, key_mask)
        attended_cls = torch.bmm(alpha.unsqueeze(1), token_seq).squeeze(1)

        beta = torch.sigmoid(affinity)
        if key_mask is not None:
            beta = beta * key_mask.float()
        beta_norm       = beta / (beta.sum(dim=-1, keepdim=True) + 1e-9)
        attended_tokens = torch.bmm(beta_norm.unsqueeze(1), token_seq).squeeze(1)

        return attended_cls, attended_tokens


# ==================== MODEL ====================
class NewsCategoryModelCoAttention(nn.Module):
    """
    Co-Attention classifier on top of HelaBERT (small).

    Pipeline:
        BERT encoder → CoAttention → LayerNorm → concat[c; c̃; T̃] (3H) → dropout → MLP → num_labels
    """

    def __init__(self, bert, hidden_size, num_labels, dropout=0.1):
        super().__init__()
        self.bert     = bert
        self.co_attn  = CoAttention(hidden_size)

        self.norm_cls    = nn.LayerNorm(hidden_size)
        self.norm_tokens = nn.LayerNorm(hidden_size)

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state

        cls_vec    = hidden[:, 0, :]
        token_seq  = hidden[:, 1:, :]
        token_mask = attention_mask[:, 1:]

        attended_cls, attended_tokens = self.co_attn(
            cls_vec, token_seq, key_mask=token_mask
        )

        attended_cls    = self.norm_cls(attended_cls)
        attended_tokens = self.norm_tokens(attended_tokens)

        # [c; c̃; T̃]: retain the raw [CLS] vector alongside the two attended
        # streams so the head strictly generalises the plain-[CLS] baseline
        # (reviewer fix for the discarded-[CLS] issue). c is left un-normalised
        # — BERT's final layer already ends in LayerNorm so scales match, and
        # keeping c raw lets the classifier recover the baseline exactly.
        combined = torch.cat([cls_vec, attended_cls, attended_tokens], dim=-1)
        logits   = self.classifier(self.dropout(combined))

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

    return NewsCategoryModelCoAttention(backbone, cfg.hidden_size, num_labels, DROPOUT)


def compute_metrics(eval_pred: EvalPrediction):
    preds  = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    return {
        'accuracy':    accuracy_score(labels, preds),
        'f1_macro':    f1_score(labels, preds, average='macro',    zero_division=0),
        'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
        'precision':   precision_score(labels, preds, average='macro', zero_division=0),
        'recall':      recall_score(labels, preds,    average='macro', zero_division=0),
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

text_col  = 'comments'
label_col = 'labels'

df = df[[text_col, label_col]].dropna()
df[text_col]  = df[text_col].astype(str).str.strip()
df[label_col] = df[label_col].astype(str).str.strip()
df = df[df[text_col].str.len() > 0].reset_index(drop=True)

# Paper pre-processing: remove sentences with fewer than 3 words
df = df[df[text_col].str.split().str.len() >= 3].reset_index(drop=True)

le = LabelEncoder()
df['label_id'] = le.fit_transform(df[label_col])
num_labels_actual = len(le.classes_)
assert num_labels_actual == NUM_LABELS, (
    f"Expected {NUM_LABELS} classes, got {num_labels_actual}: {list(le.classes_)}"
)

all_texts  = df[text_col].tolist()
all_labels = df['label_id'].tolist()

print(f"Loaded {len(df):,} samples, {num_labels_actual} classes: {list(le.classes_)}")
for i, cls in enumerate(le.classes_):
    cnt = sum(1 for l in all_labels if l == i)
    print(f"  [{i}] {cls}: {cnt}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
pd.DataFrame({'id': range(num_labels_actual), 'label': le.classes_}).to_csv(
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

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        all_texts, all_labels,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=all_labels,
    )
    print(f"Train: {len(train_texts):,}  |  Test: {len(test_texts):,}")

    train_ds = NewsCategoryDataset(train_texts, train_labels, sp, MAX_SEQ_LENGTH)
    test_ds  = NewsCategoryDataset(test_texts,  test_labels,  sp, MAX_SEQ_LENGTH)

    model   = load_fresh_model(NUM_LABELS)
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
                "task": "news_category", "method": "co_attention_clsconcat",
                "model": "HelaBERT_small",
                "run": run_idx, "seed": seed,
                "lr": LEARNING_RATE, "batch": BATCH_SIZE, "epochs": NUM_EPOCHS,
                "num_labels": NUM_LABELS, "label_names": list(le.classes_),
                "max_seq_length": MAX_SEQ_LENGTH, "test_size": TEST_SIZE,
                "head": "CoAttention -> concat[raw_cls; attended_cls; attended_tokens] (3H) -> MLP",
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
print(f"Paper baseline (SinBERT-small): 84.75")

results_df = pd.DataFrame({
    'run':      range(1, N_RUNS + 1),
    'seed':     SEEDS,
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
