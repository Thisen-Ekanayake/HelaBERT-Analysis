"""
Unified 5-seed fine-tuning sweep for the mean±std fix (review Weakness 1).

Fixes two issues raised by review 1:
  - Headline results were "best of 5 seeds selected by TEST macro-F1"
    (model selection on the test set, upward-biased).
  - Per-run checkpoint selection (metric_for_best_model) was also evaluated
    on the test split, not a held-out validation split.

Protocol here, for every (model_size, task, head) combination:
  1. Stratified 80/20 train_full/test split (random_state=seed) -- identical
     boundary to the original protocol, for comparability with baselines.
  2. Stratified 90/10 split of train_full into train/val (random_state=seed)
     -- val is used ONLY for checkpoint selection during training.
  3. Train with eval_strategy="epoch" on val, metric_for_best_model=
     "eval_f1_macro", load_best_model_at_end=True.
  4. After training, a single trainer.predict(test_ds) on the untouched
     test split gives the one reported number for that seed.
  5. Report mean ± std of the 5 seeds' test macro-F1 (no cherry-picking).

Architectures (standard / co-attention head) are copied verbatim from the
existing finetune/helabert_{small,large}/*.py scripts to preserve exact
fidelity with what is described in the paper.

A third head, "coattn_clsconcat", implements the reviewer-requested [c; c̃; T̃]
variant: it concatenates the raw [CLS] vector (c) with the two attended
streams so the head strictly generalises the plain-[CLS] baseline. It is added
as a separate arm so the standard / coattn / coattn_clsconcat results can be
compared side by side.
"""

import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import gc
import csv
import random
import random as stdlib_random
import argparse

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
)
from transformers import (
    BertConfig, BertForMaskedLM, BertModel,
    Trainer, TrainingArguments, EvalPrediction,
)
from transformers.modeling_outputs import SequenceClassifierOutput

REPO_ROOT = "/ml/HelaBERT_Analysis"
os.chdir(REPO_ROOT)

TOKENIZER_MODEL = "tokenizer/unigram_32000_0.9995.model"
SEEDS = [42, 123, 456, 789, 1024]
RESULTS_CSV = "five_seed_sweep_results.csv"
LOG_DIR = "five_seed_sweep_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ==================== TASK DEFINITIONS ====================
def load_news_category(sp):
    df = pd.read_csv("data/Sinhala-News-Category-classification/train/news_train.csv")
    df.columns = df.columns.str.strip()
    text_col, label_col = "comments", "labels"
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 0].reset_index(drop=True)
    df = df[df[text_col].str.split().str.len() >= 3].reset_index(drop=True)
    return df, text_col, label_col

def load_news_source(sp):
    df = pd.read_csv("data/Sinhala-News-Source-classification/train/news_source_train.csv")
    df.columns = df.columns.str.strip()
    text_col, label_col = "comment", "label"
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 0].reset_index(drop=True)
    return df, text_col, label_col

def load_sentiment(sp):
    df = pd.read_csv("data/sinhala-sentiment-analysis/train.csv")
    df.columns = df.columns.str.strip()
    text_col = next(c for c in df.columns if "comment" in c.lower())
    label_col = next(c for c in df.columns if "sentiment" in c.lower())
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 0].reset_index(drop=True)
    return df, text_col, label_col

def load_writing_style(sp):
    df = pd.read_csv("data/Writing-style-classification/train/writing_style_train.csv")
    df.columns = df.columns.str.strip()
    text_col, label_col = "comments", "labels"
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 0].reset_index(drop=True)
    df = df[df[text_col].str.len() <= 3500].reset_index(drop=True)
    return df, text_col, label_col

TASKS = {
    "news_category": dict(loader=load_news_category, num_labels=5),
    "news_source":   dict(loader=load_news_source,   num_labels=9),
    "sentiment":     dict(loader=load_sentiment,      num_labels=3),
    "writing_style": dict(loader=load_writing_style, num_labels=4),
}

# LR / epochs per Table 6 of the paper (per model size x task)
HPARAMS = {
    ("small", "news_category"): dict(lr=1e-5, epochs=10),
    ("small", "news_source"):   dict(lr=5e-5, epochs=3),
    ("small", "sentiment"):     dict(lr=3e-5, epochs=6),
    ("small", "writing_style"): dict(lr=1e-5, epochs=3),
    ("large", "news_category"): dict(lr=3e-5, epochs=3),
    ("large", "news_source"):   dict(lr=5e-5, epochs=3),
    ("large", "sentiment"):     dict(lr=5e-6, epochs=10),
    ("large", "writing_style"): dict(lr=1e-5, epochs=3),
}

MODEL_PATHS = {"small": "HelaBERT_small", "large": "HelaBERT_large"}
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 512
DROPOUT = 0.1

# ==================== DATASET ====================
class TextClsDataset(Dataset):
    def __init__(self, texts, labels, sp_processor, max_length):
        self.texts, self.labels = texts, labels
        self.sp, self.max_length = sp_processor, max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = self.sp.encode(str(self.texts[idx]))[: self.max_length]
        mask = [1] * len(ids)
        pad = self.max_length - len(ids)
        ids += [0] * pad
        mask += [0] * pad
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ==================== HEADS (verbatim from finetune/helabert_*/*.py) ====================
class StandardHeadModel(nn.Module):
    def __init__(self, bert, hidden_size, num_labels, dropout=0.1):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


class CoAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W_cls = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_token = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.scale = hidden_size ** 0.5

    @staticmethod
    def softmax_safe(logits, mask=None):
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e4)
        return F.softmax(logits, dim=-1)

    def forward(self, cls_vec, token_seq, key_mask=None):
        B, T, H = token_seq.shape
        cls_proj = self.W_cls(cls_vec).unsqueeze(1).expand(-1, T, -1)
        token_proj = self.W_token(token_seq)
        affinity = self.v(torch.tanh(cls_proj + token_proj)).squeeze(-1) / self.scale

        alpha = self.softmax_safe(affinity, key_mask)
        attended_cls = torch.bmm(alpha.unsqueeze(1), token_seq).squeeze(1)

        beta = torch.sigmoid(affinity)
        if key_mask is not None:
            beta = beta * key_mask.float()
        beta_norm = beta / (beta.sum(dim=-1, keepdim=True) + 1e-9)
        attended_tokens = torch.bmm(beta_norm.unsqueeze(1), token_seq).squeeze(1)

        return attended_cls, attended_tokens


class CoAttentionHeadModel(nn.Module):
    def __init__(self, bert, hidden_size, num_labels, dropout=0.1):
        super().__init__()
        self.bert = bert
        self.co_attn = CoAttention(hidden_size)
        self.norm_cls = nn.LayerNorm(hidden_size)
        self.norm_tokens = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        cls_vec = hidden[:, 0, :]
        token_seq = hidden[:, 1:, :]
        token_mask = attention_mask[:, 1:]

        attended_cls, attended_tokens = self.co_attn(cls_vec, token_seq, key_mask=token_mask)
        attended_cls = self.norm_cls(attended_cls)
        attended_tokens = self.norm_tokens(attended_tokens)
        combined = torch.cat([attended_cls, attended_tokens], dim=-1)
        logits = self.classifier(self.dropout(combined))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


class CoAttentionClsConcatHeadModel(nn.Module):
    """Co-attention head that also concatenates the raw [CLS] vector: [c; c̃; T̃].

    Reviewer fix for the discarded-[CLS] issue. The plain [c̃; T̃] head feeds the
    classifier only the two *attended* streams and drops the raw [CLS] vector
    (c) that the standard baseline classifies on. Concatenating c back in lets
    this head strictly generalise the baseline: it can zero out the attention
    streams and recover the plain-[CLS] classifier exactly. c is left
    un-normalised (BERT's final layer already ends in LayerNorm) so that exact
    recovery is possible.
    """
    def __init__(self, bert, hidden_size, num_labels, dropout=0.1):
        super().__init__()
        self.bert = bert
        self.co_attn = CoAttention(hidden_size)
        self.norm_cls = nn.LayerNorm(hidden_size)
        self.norm_tokens = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        cls_vec = hidden[:, 0, :]
        token_seq = hidden[:, 1:, :]
        token_mask = attention_mask[:, 1:]

        attended_cls, attended_tokens = self.co_attn(cls_vec, token_seq, key_mask=token_mask)
        attended_cls = self.norm_cls(attended_cls)
        attended_tokens = self.norm_tokens(attended_tokens)
        combined = torch.cat([cls_vec, attended_cls, attended_tokens], dim=-1)
        logits = self.classifier(self.dropout(combined))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


def load_fresh_model(model_path, head, hidden_size, num_labels):
    cfg_file = f"{model_path}/config.json"
    if os.path.exists(cfg_file):
        BertConfig.from_json_file(cfg_file)
    try:
        backbone = BertModel.from_pretrained(model_path)
    except Exception:
        mlm = BertForMaskedLM.from_pretrained(model_path)
        backbone = mlm.bert
    HEADS = {
        "standard": StandardHeadModel,
        "coattn": CoAttentionHeadModel,
        "coattn_clsconcat": CoAttentionClsConcatHeadModel,
    }
    return HEADS[head](backbone, hidden_size, num_labels, DROPOUT)


def compute_metrics(eval_pred: EvalPrediction):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
    }


def set_seed(seed):
    random.seed(seed)
    stdlib_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_result(row):
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_combo(model_size, task, head, only_seed=None):
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)

    task_cfg = TASKS[task]
    df, text_col, label_col = task_cfg["loader"](sp)
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df[label_col])
    num_labels = len(le.classes_)

    all_texts = df[text_col].tolist()
    all_labels = df["label_id"].tolist()

    hp = HPARAMS[(model_size, task)]
    model_path = MODEL_PATHS[model_size]
    cfg = BertConfig.from_json_file(f"{model_path}/config.json")
    hidden_size = cfg.hidden_size

    seeds = [only_seed] if only_seed is not None else SEEDS
    test_f1s = []

    for run_idx, seed in enumerate(seeds, start=1):
        combo_name = f"{model_size}_{task}_{head}_seed{seed}"
        print(f"\n{'='*80}\n{combo_name}\n{'='*80}")
        set_seed(seed)

        train_full_texts, test_texts, train_full_labels, test_labels = train_test_split(
            all_texts, all_labels, test_size=0.2, random_state=seed, stratify=all_labels,
        )
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_full_texts, train_full_labels, test_size=0.1,
            random_state=seed, stratify=train_full_labels,
        )
        print(f"Train: {len(train_texts)}  Val: {len(val_texts)}  Test: {len(test_texts)}")

        train_ds = TextClsDataset(train_texts, train_labels, sp, MAX_SEQ_LENGTH)
        val_ds = TextClsDataset(val_texts, val_labels, sp, MAX_SEQ_LENGTH)
        test_ds = TextClsDataset(test_texts, test_labels, sp, MAX_SEQ_LENGTH)

        model = load_fresh_model(model_path, head, hidden_size, num_labels)

        run_dir = f"{LOG_DIR}/{combo_name}"
        os.makedirs(run_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=run_dir,
            num_train_epochs=hp["epochs"],
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE * 2,
            learning_rate=hp["lr"],
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
            report_to="none",
            seed=seed,
            disable_tqdm=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,          # validation, NOT test -- fixes model-selection-on-test
            compute_metrics=compute_metrics,
        )
        trainer.train()
        best_val_f1 = trainer.state.best_metric

        preds_out = trainer.predict(test_ds)
        preds = np.argmax(preds_out.predictions, axis=1)
        test_f1 = f1_score(test_labels, preds, average="macro", zero_division=0)
        test_acc = accuracy_score(test_labels, preds)
        test_f1w = f1_score(test_labels, preds, average="weighted", zero_division=0)
        test_prec = precision_score(test_labels, preds, average="macro", zero_division=0)
        test_rec = recall_score(test_labels, preds, average="macro", zero_division=0)

        test_f1s.append(test_f1)
        print(f"  seed={seed}  val_f1_macro(best)={best_val_f1:.4f}  test_f1_macro={test_f1:.4f}")

        append_result(dict(
            model=model_size, task=task, head=head, seed=seed,
            val_f1_macro=round(best_val_f1, 4), test_f1_macro=round(test_f1, 4),
            test_accuracy=round(test_acc, 4), test_f1_weighted=round(test_f1w, 4),
            test_precision=round(test_prec, 4), test_recall=round(test_rec, 4),
        ))

        # cleanup between runs -- long sweep on a single 8GB GPU
        del model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if test_f1s:
        mean_f1, std_f1 = float(np.mean(test_f1s)), float(np.std(test_f1s))
        print(f"\n>>> {model_size}/{task}/{head}: mean={mean_f1:.4f} std={std_f1:.4f} "
              f"over {len(test_f1s)} seed(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["small", "large"], required=False)
    parser.add_argument("--task", choices=list(TASKS.keys()), required=False)
    parser.add_argument("--head", choices=["standard", "coattn", "coattn_clsconcat"], required=False)
    parser.add_argument("--seed", type=int, default=None, help="run only this single seed (smoke test)")
    args = parser.parse_args()

    combos = []
    for model_size in (["small", "large"] if not args.model else [args.model]):
        for task in (list(TASKS.keys()) if not args.task else [args.task]):
            for head in (["standard", "coattn", "coattn_clsconcat"] if not args.head else [args.head]):
                combos.append((model_size, task, head))

    print(f"Running {len(combos)} combo(s), {len(SEEDS) if args.seed is None else 1} seed(s) each")
    for model_size, task, head in combos:
        run_combo(model_size, task, head, only_seed=args.seed)

    print("\nAll done. Results in", RESULTS_CSV)
