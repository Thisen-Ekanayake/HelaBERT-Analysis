"""
Fine-tuning XLM-R_large for News Category Classification — 5-Fold Cross Validation
— Balanced training via oversampling + weighted loss —
— Comparison baseline against HelaBERT —

Architecture:
    text → XLM-R_large encoder → [CLS] → LayerNorm → Dropout → Linear → num_labels
    (full fine-tuning)

Mirrors the exact training/evaluation pipeline used for HelaBERT:
  • StratifiedKFold(n_splits=5) — same splits strategy
  • Oversampling applied ONLY to training split
  • Class weights recomputed per fold from raw (pre-oversample) distribution
  • Same Trainer, TrainingArguments, EarlyStoppingCallback settings
  • Same metrics: accuracy, macro-F1, weighted-F1, precision, recall
  • OOF report generated at end
  • W&B logging enabled

Model:   FacebookAI/xlm-roberta-large
Task:    News Category Classification
Data:    data/Sinhala-News-Category-classification/train/news_train.csv
"""

import os
import traceback
import json
from collections import Counter
import random as stdlib_random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    EarlyStoppingCallback,
)
import random
import wandb

# ==================== CONFIGURATION ====================
print("=" * 80)
print("XLM-R_LARGE FINE-TUNING — 5-FOLD CV  [NEWS CATEGORY CLASSIFICATION]")
print("=" * 80)

MODEL_NAME       = "FacebookAI/xlm-roberta-large"
DATA_PATH        = "data/Sinhala-News-Category-classification/train/news_train.csv"

NUM_LABELS                   = 5
COMMENT_MAX_LENGTH           = 512
TRAIN_BATCH_SIZE             = 4
EVAL_BATCH_SIZE              = 8
LEARNING_RATE                = 2e-05
NUM_EPOCHS                   = 6
WARMUP_RATIO                 = 0.05
WEIGHT_DECAY                 = 0.05
GRADIENT_ACCUMULATION_STEPS  = 4    # effective batch = 16
EARLY_STOPPING_PATIENCE      = 3

N_FOLDS           = 5
OVERSAMPLE_TRAIN  = True
USE_CLASS_WEIGHTS = True

OUTPUT_DIR     = "XLM_R_large_finetuned_news_category_cv"
BEST_MODEL_DIR = f"{OUTPUT_DIR}/best_model"

RANDOM_SEED = 42
USE_FP16    = True
NUM_WORKERS = 2

USE_WANDB      = True
WANDB_PROJECT  = "XLM_R_large-news-category-finetuning"
WANDB_GROUP    = f"5fold_cv_lr{LEARNING_RATE}_bs{TRAIN_BATCH_SIZE}"
WANDB_ENTITY   = None

print(f"\n✓ Model:           {MODEL_NAME}")
print(f"  Task:            News Category Classification")
print(f"  Frozen encoder:  No")
print(f"  Max length:      {COMMENT_MAX_LENGTH}")
print(f"  Effective batch: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")


# ==================== SEEDS ====================
random.seed(RANDOM_SEED)
stdlib_random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
print("✓ Random seeds set")


# ==================== ENVIRONMENT ====================
print("\n" + "=" * 80)
print("ENVIRONMENT")
print("=" * 80)
print(f"PyTorch: {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CPU only — training will be slow")
assert os.path.exists(DATA_PATH), f"Data not found: {DATA_PATH}"
print("All paths verified")


# ==================== TOKENIZER ====================
print("\n" + "=" * 80)
print("LOADING TOKENIZER")
print("=" * 80)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"✓ Tokenizer loaded — vocab size: {tokenizer.vocab_size}")


# ==================== DATASET ====================
class TextClassificationDataset(Dataset):
    """
    Tokenises each text sample with AutoTokenizer.
    Mirrors HelaBERT's COMMENT_MAX_LENGTH truncation/padding approach.
    """

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ==================== HELPERS ====================
def oversample(texts, labels, seed=42):
    """Oversample minority classes to match majority class count.
    Identical to HelaBERT implementation."""
    stdlib_random.seed(seed)
    counts    = Counter(labels)
    max_count = max(counts.values())

    bal_texts, bal_labels = list(texts), list(labels)
    for label, count in counts.items():
        needed = max_count - count
        if needed == 0:
            continue
        indices = [i for i, l in enumerate(labels) if l == label]
        extras  = stdlib_random.choices(indices, k=needed)
        bal_texts  += [texts[i] for i in extras]
        bal_labels += [labels[i] for i in extras]

    combined = list(zip(bal_texts, bal_labels))
    stdlib_random.shuffle(combined)
    bal_texts, bal_labels = zip(*combined)
    return list(bal_texts), list(bal_labels)


def compute_class_weights(labels, num_labels):
    """Inverse-frequency weights normalised so they sum to num_labels.
    Identical to HelaBERT implementation."""
    counts  = Counter(labels)
    weights = torch.tensor(
        [1.0 / counts.get(i, 1) for i in range(num_labels)],
        dtype=torch.float,
    )
    weights = weights / weights.sum() * num_labels
    return weights


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    """Identical metric set to HelaBERT: accuracy, macro-F1, weighted-F1, precision, recall."""
    preds  = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    return {
        'accuracy':    accuracy_score(labels, preds),
        'precision':   precision_score(labels, preds, average='macro',    zero_division=0),
        'recall':      recall_score(labels,    preds, average='macro',    zero_division=0),
        'f1':          f1_score(labels,        preds, average='macro',    zero_division=0),
        'f1_weighted': f1_score(labels,        preds, average='weighted', zero_division=0),
    }


def print_metric(key, value):
    if isinstance(value, float):
        print(f"    {key}: {value:.4f}")
    else:
        print(f"    {key}: {value}")


# ==================== MODEL ====================
class ClassificationModel(nn.Module):
    """
    XLM-R_large encoder → [CLS] → LayerNorm → Dropout → Linear → num_labels

    Mirrors HelaBERT's BaselineModel architecture exactly.
    Full fine-tuning end-to-end.
    """

    def __init__(self, encoder, hidden_size, num_labels, dropout=0.1):
        super().__init__()
        self.encoder    = encoder
        self.hidden_size = hidden_size
        self.norm        = nn.LayerNorm(hidden_size)
        self.dropout     = nn.Dropout(dropout)
        self.classifier  = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        out     = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = out.last_hidden_state[:, 0, :]   # [B, H]
        logits  = self.classifier(self.dropout(self.norm(cls_vec)))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {'loss': loss, 'logits': logits}


# ==================== CUSTOM TRAINER ====================
class WeightedTrainer(Trainer):
    """Trainer subclass that applies class-weight loss.
    Identical pattern to HelaBERT's BaselineTrainer."""

    def __init__(self, class_weights: torch.Tensor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.get("logits")
        if self.class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def load_fresh_model(num_labels):
    """Load a fresh model for each fold (same pattern as HelaBERT's load_fresh_model)."""
    encoder = AutoModel.from_pretrained(MODEL_NAME)
    hidden_size = encoder.config.hidden_size
    model = ClassificationModel(
        encoder=encoder,
        hidden_size=hidden_size,
        num_labels=num_labels,
    )
    return model, hidden_size


# ==================== LOAD DATA ====================
print("\n" + "=" * 80)
print("LOADING DATASET")
print("=" * 80)

try:
    df = pd.read_csv(DATA_PATH)
except pd.errors.ParserError:
    df = pd.read_csv(DATA_PATH, engine='python', on_bad_lines='skip')

    df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

    possible_comment_cols = [c for c in df.columns if 'comments' in c.lower()]
    possible_label_cols   = [c for c in df.columns if 'labels'   in c.lower()]
    if possible_comment_cols and possible_label_cols:
        df = df.rename(columns={possible_comment_cols[0]: 'comments', possible_label_cols[0]: 'labels'})
    else:
        df = df.iloc[:, -2:].copy()
        df.columns = ['comments', 'labels']

df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c], errors='ignore')
df = df.dropna(subset=['comments', 'labels'])
df['comments'] = df['comments'].astype(str).str.strip()
df['labels'] = df['labels'].astype(str).str.strip().astype(int)
df = df[df['comments'].str.len() > 0].reset_index(drop=True)
print(f"✓ Loaded {len(df):,} samples")

# ==================== LABEL SETUP ====================
actual_num_labels = df['labels'].nunique()
if actual_num_labels != NUM_LABELS:
    print(f"Updating NUM_LABELS: {NUM_LABELS} → {actual_num_labels}")
    NUM_LABELS = actual_num_labels

print(f"Loaded {len(df)} samples, {NUM_LABELS} classes")
print("\nFull dataset label distribution:")
for lbl, cnt in df['labels'].value_counts().sort_index().items():
    print(f"  Label {lbl}: {cnt:6d} ({100 * cnt / len(df):.1f}%)")

all_texts  = df['comments'].tolist()
all_labels = df['labels'].tolist()


# ==================== PROBE MODEL (for hidden size / W&B config) ====================
print("\n" + "=" * 80)
print("PROBING MODEL ARCHITECTURE")
print("=" * 80)
_probe, hidden_size = load_fresh_model(NUM_LABELS)
total_p     = sum(p.numel() for p in _probe.parameters())
trainable_p = sum(p.numel() for p in _probe.parameters() if p.requires_grad)
print(f"Total params    : {total_p:,}")
print(f"Trainable params: {trainable_p:,}  ({100*trainable_p/total_p:.1f}%)")
del _probe


# ==================== CROSS-VALIDATION LOOP ====================
print("\n" + "=" * 80)
print(f"STARTING {N_FOLDS}-FOLD CROSS VALIDATION")
print("=" * 80)

os.makedirs(OUTPUT_DIR, exist_ok=True)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

fold_metrics:  list = []
all_y_true:    list = []
all_y_pred:    list = []
all_oof_texts: list = []
best_fold_f1   = -1.0
best_fold_idx  = -1
wandb_group_url = None

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_texts, all_labels), start=1):
    print("\n" + "=" * 80)
    print(f"FOLD {fold_idx} / {N_FOLDS}")
    print("=" * 80)

    fold_output_dir = f"{OUTPUT_DIR}/fold_{fold_idx}"
    os.makedirs(fold_output_dir, exist_ok=True)

    # ── Split ───────────────────────────────────────────────────────────────
    fold_train_texts  = [all_texts[i]  for i in train_idx]
    fold_train_labels = [all_labels[i] for i in train_idx]
    fold_val_texts    = [all_texts[i]  for i in val_idx]
    fold_val_labels   = [all_labels[i] for i in val_idx]

    print(f"Train: {len(fold_train_texts)} samples  |  Val: {len(fold_val_texts)} samples")
    print("Train label distribution (before oversampling):")
    for lbl, cnt in sorted(Counter(fold_train_labels).items()):
        print(f"  Label {lbl}: {cnt}")

    # ── Oversample (train split only) ───────────────────────────────────────
    if OVERSAMPLE_TRAIN:
        fold_train_texts, fold_train_labels = oversample(
            fold_train_texts, fold_train_labels, seed=RANDOM_SEED + fold_idx
        )
        print(f"After oversampling: {len(fold_train_texts)} train samples")

    # ── Class weights (raw pre-oversample distribution) ─────────────────────
    raw_fold_labels = [all_labels[i] for i in train_idx]
    fold_weights    = compute_class_weights(raw_fold_labels, NUM_LABELS)
    if USE_CLASS_WEIGHTS:
        print("Class weights:")
        for i, w in enumerate(fold_weights):
            print(f"  Label {i}: {w.item():.4f}")

    # ── Datasets ────────────────────────────────────────────────────────────
    train_ds = TextClassificationDataset(
        fold_train_texts, fold_train_labels, tokenizer, COMMENT_MAX_LENGTH
    )
    val_ds = TextClassificationDataset(
        fold_val_texts, fold_val_labels, tokenizer, COMMENT_MAX_LENGTH
    )

    # ── Fresh model ─────────────────────────────────────────────────────────
    print(f"Loading fresh model for fold {fold_idx}...")
    model, hidden_size = load_fresh_model(NUM_LABELS)
    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_p:,} total, {trainable_p:,} trainable")

    # ── W&B ─────────────────────────────────────────────────────────────────
    wandb_run_name = f"fold_{fold_idx}_of_{N_FOLDS}"
    if USE_WANDB:
        if wandb.run is not None:
            wandb.finish()
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            group=WANDB_GROUP,
            name=wandb_run_name,
            config={
                "model":                  MODEL_NAME,
                "fold":                   fold_idx,
                "n_folds":                N_FOLDS,
                "frozen_encoder":         False,
                "comment_max_length":     COMMENT_MAX_LENGTH,
                "learning_rate":          LEARNING_RATE,
                "epochs":                 NUM_EPOCHS,
                "train_batch_size":       TRAIN_BATCH_SIZE,
                "effective_batch":        TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
                "warmup_ratio":           WARMUP_RATIO,
                "weight_decay":           WEIGHT_DECAY,
                "oversample":             OVERSAMPLE_TRAIN,
                "class_weights":          USE_CLASS_WEIGHTS,
                "train_samples_balanced": len(fold_train_texts),
                "val_samples":            len(fold_val_texts),
                **{f"class_weight_{i}": fold_weights[i].item() for i in range(NUM_LABELS)},
            },
            reinit=True,
        )
        wandb.log({
            "train_label_dist": wandb.Histogram(fold_train_labels),
            "val_label_dist":   wandb.Histogram(fold_val_labels),
        })
        if fold_idx == 1:
            wandb_group_url = f"https://wandb.ai/{run.entity}/{WANDB_PROJECT}/groups/{WANDB_GROUP}"
        print(f"W&B run: {run.get_url()}")

    # ── Training args ────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_steps=100,
        save_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=50,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=USE_FP16 and torch.cuda.is_available(),
        dataloader_num_workers=NUM_WORKERS,
        seed=RANDOM_SEED + fold_idx,
        report_to="wandb" if USE_WANDB else "none",
        run_name=wandb_run_name if USE_WANDB else None,
        push_to_hub=False,
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    early_stop = EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)

    trainer = WeightedTrainer(
        class_weights=fold_weights if USE_CLASS_WEIGHTS else None,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[early_stop],
    )

    # ── Train ────────────────────────────────────────────────────────────────
    try:
        train_result = trainer.train()
        print(f"\n  Fold {fold_idx} training complete.")
        for k, v in train_result.metrics.items():
            print_metric(k, v)
    except KeyboardInterrupt:
        print(f"\nInterrupted at fold {fold_idx}.")
        trainer.save_model(f"{fold_output_dir}/interrupted_model")
    except Exception as e:
        print(f"\nFold {fold_idx} failed: {e}\n{traceback.format_exc()}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print(f"\nEvaluating fold {fold_idx}...")
    eval_results = trainer.evaluate()
    print(f"Fold {fold_idx} eval metrics:")
    for k, v in eval_results.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    # ── Predictions & per-class report ──────────────────────────────────────
    preds_out = trainer.predict(val_ds)
    y_pred    = np.argmax(preds_out.predictions, axis=-1)
    y_true    = np.array(fold_val_labels)

    all_y_true.extend(y_true.tolist())
    all_y_pred.extend(y_pred.tolist())
    all_oof_texts.extend(fold_val_texts)

    print(f"\nClassification report — Fold {fold_idx}:")
    print(classification_report(y_true, y_pred, digits=4))

    # ── Save per-fold predictions ────────────────────────────────────────────
    fold_conf = [
        torch.softmax(torch.tensor(preds_out.predictions[i]), dim=0)[y_pred[i]].item()
        for i in range(len(y_pred))
    ]
    pd.DataFrame({
        'text':            fold_val_texts,
        'true_label':      y_true,
        'predicted_label': y_pred,
        'correct':         y_true == y_pred,
        'confidence':      fold_conf,
    }).to_csv(f"{fold_output_dir}/predictions.csv", index=False)

    # ── Per-class metrics ────────────────────────────────────────────────────
    prec_pc, rec_pc, f1_pc, sup_pc = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    pd.DataFrame({
        'Class':     range(NUM_LABELS),
        'Precision': prec_pc,
        'Recall':    rec_pc,
        'F1-Score':  f1_pc,
        'Support':   sup_pc,
    }).to_csv(f"{fold_output_dir}/per_class_metrics.csv", index=False)

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm).to_csv(f"{fold_output_dir}/confusion_matrix.csv")

    # ── Store fold metrics ───────────────────────────────────────────────────
    fold_m = {
        'fold':        fold_idx,
        'accuracy':    eval_results['eval_accuracy'],
        'precision':   eval_results['eval_precision'],
        'recall':      eval_results['eval_recall'],
        'f1':          eval_results['eval_f1'],
        'f1_weighted': eval_results['eval_f1_weighted'],
    }
    fold_metrics.append(fold_m)

    # ── W&B fold summary ─────────────────────────────────────────────────────
    if USE_WANDB:
        wandb.log({
            "fold_summary/accuracy":    fold_m['accuracy'],
            "fold_summary/f1":          fold_m['f1'],
            "fold_summary/f1_weighted": fold_m['f1_weighted'],
            "fold_summary/precision":   fold_m['precision'],
            "fold_summary/recall":      fold_m['recall'],
        })
        try:
            import plotly.figure_factory as ff
            fig = ff.create_annotated_heatmap(
                z=cm, colorscale='Blues', showscale=True
            )
            fig.update_layout(title=f"Confusion Matrix — Fold {fold_idx}")
            wandb.log({"confusion_matrix": fig})
        except ImportError:
            pass

    # ── Save best model ──────────────────────────────────────────────────────
    if fold_m['f1'] > best_fold_f1:
        best_fold_f1  = fold_m['f1']
        best_fold_idx = fold_idx
        os.makedirs(BEST_MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), f"{BEST_MODEL_DIR}/pytorch_model.bin")
        with open(f"{BEST_MODEL_DIR}/model_info.json", "w") as fh:
            json.dump({
                "model_name":         MODEL_NAME,
                "task":               "News Category Classification",
                "hidden_size":        hidden_size,
                "num_labels":         NUM_LABELS,
                "comment_max_length": COMMENT_MAX_LENGTH,
                "frozen_encoder":     False,
            }, fh, indent=2)
        print(f"\nNew best model saved (fold {fold_idx}, F1={best_fold_f1:.4f}) → {BEST_MODEL_DIR}")

    print(f"\nFold {fold_idx} done — macro F1: {fold_m['f1']:.4f}")

    if USE_WANDB:
        wandb.finish()
        print(f"W&B fold {fold_idx} run finished")

    model.train()


# ==================== CROSS-VALIDATION SUMMARY ====================
print("\n" + "=" * 80)
print("CROSS-VALIDATION RESULTS")
print("=" * 80)

metrics_df = pd.DataFrame(fold_metrics)
metrics_df.to_csv(f"{OUTPUT_DIR}/cv_fold_metrics.csv", index=False)

print("\nPer-fold results:")
print(metrics_df.to_string(index=False))

mean_metrics = metrics_df.drop(columns=['fold']).mean()
std_metrics  = metrics_df.drop(columns=['fold']).std()

print("\n" + "-" * 60)
print(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'95% CI':>20}")
print("-" * 60)
for metric in ['accuracy', 'precision', 'recall', 'f1', 'f1_weighted']:
    m     = mean_metrics[metric]
    s     = std_metrics[metric]
    ci_lo = m - 1.96 * s / (N_FOLDS ** 0.5)
    ci_hi = m + 1.96 * s / (N_FOLDS ** 0.5)
    print(f"  {metric:<18} {m:>10.4f} {s:>10.4f}   [{ci_lo:.4f}, {ci_hi:.4f}]")
print("-" * 60)
print(f"\nBest single fold: Fold {best_fold_idx}  (macro F1 = {best_fold_f1:.4f})")


# ==================== OUT-OF-FOLD (OOF) REPORT ====================
print("\n" + "=" * 80)
print("OUT-OF-FOLD (OOF) REPORT — FULL DATASET")
print("=" * 80)

oof_y_true = np.array(all_y_true)
oof_y_pred = np.array(all_y_pred)
print(classification_report(oof_y_true, oof_y_pred, digits=4))

oof_f1   = f1_score(oof_y_true, oof_y_pred, average='macro',    zero_division=0)
oof_f1_w = f1_score(oof_y_true, oof_y_pred, average='weighted', zero_division=0)
oof_acc  = accuracy_score(oof_y_true, oof_y_pred)

print(f"OOF macro F1:    {oof_f1:.4f}")
print(f"OOF weighted F1: {oof_f1_w:.4f}")
print(f"OOF accuracy:    {oof_acc:.4f}")

oof_cm = confusion_matrix(oof_y_true, oof_y_pred)
pd.DataFrame(oof_cm).to_csv(f"{OUTPUT_DIR}/oof_confusion_matrix.csv")
pd.DataFrame({
    'text':            all_oof_texts,
    'true_label':      oof_y_true,
    'predicted_label': oof_y_pred,
    'correct':         oof_y_true == oof_y_pred,
}).to_csv(f"{OUTPUT_DIR}/oof_predictions.csv", index=False)


# ==================== W&B CV SUMMARY RUN ====================
if USE_WANDB:
    print("\nLogging CV summary to W&B...")
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        group=WANDB_GROUP,
        name="cv_summary",
        reinit=True,
    )
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'f1_weighted']:
        wandb.log({
            f"cv_mean/{metric}": mean_metrics[metric],
            f"cv_std/{metric}":  std_metrics[metric],
        })
    wandb.log({
        "oof/f1":          oof_f1,
        "oof/f1_weighted": oof_f1_w,
        "oof/accuracy":    oof_acc,
        "best_fold":       best_fold_idx,
        "best_fold_f1":    best_fold_f1,
        "cv_fold_metrics": wandb.Table(dataframe=metrics_df),
    })
    try:
        import plotly.figure_factory as ff
        fig = ff.create_annotated_heatmap(z=oof_cm, colorscale='Blues', showscale=True)
        fig.update_layout(title="OOF Confusion Matrix — All Folds")
        wandb.log({"oof_confusion_matrix": fig})
    except ImportError:
        pass
    wandb.finish()
    print(f"CV summary logged — group: {WANDB_GROUP}")


# ==================== FINAL SUMMARY ====================
cv_summary = {
    'Model':                "XLM-R_large",
    'HuggingFace ID':       MODEL_NAME,
    'Frozen Encoder':       "No",
    'Task':                 "News Category Classification",
    'N Folds':              N_FOLDS,
    'Num Classes':          NUM_LABELS,
    'Total Samples':        len(all_texts),
    'Max Length':           COMMENT_MAX_LENGTH,
    'Learning Rate':        LEARNING_RATE,
    'Effective Batch Size': TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
    'Best Fold':            best_fold_idx,
    'Best Fold F1':         f'{best_fold_f1:.4f}',
    'Mean F1 (macro)':      f'{mean_metrics["f1"]:.4f} ± {std_metrics["f1"]:.4f}',
    'Mean F1 (weighted)':   f'{mean_metrics["f1_weighted"]:.4f} ± {std_metrics["f1_weighted"]:.4f}',
    'Mean Accuracy':        f'{mean_metrics["accuracy"]:.4f} ± {std_metrics["accuracy"]:.4f}',
    'OOF F1 (macro)':       f'{oof_f1:.4f}',
    'OOF F1 (weighted)':    f'{oof_f1_w:.4f}',
    'OOF Accuracy':         f'{oof_acc:.4f}',
}

pd.DataFrame([cv_summary]).to_csv(f'{OUTPUT_DIR}/cv_summary.csv', index=False)

final_summary_lines = []
final_summary_lines.append("\n" + "=" * 80)
final_summary_lines.append("FINAL CROSS-VALIDATION SUMMARY")
final_summary_lines.append("=" * 80)
for k, v in cv_summary.items():
    final_summary_lines.append(f"  {k:<28}: {v}")
final_summary_lines.append("\n" + "=" * 80)
final_summary_lines.append(f"5-FOLD CV COMPLETE — XLM-R_large / News Category Classification")
final_summary_lines.append("=" * 80)
final_summary_lines.append(f"\nOutputs saved to: {OUTPUT_DIR}/")
if USE_WANDB and wandb_group_url:
    final_summary_lines.append(f"W&B group: {wandb_group_url}")

final_summary_text = "\n".join(final_summary_lines)
print(final_summary_text)

# ==================== SAVE EVAL RESULTS ====================
_eval_results_dir = os.path.join("eval_results", "XLM-R_large")
os.makedirs(_eval_results_dir, exist_ok=True)
_eval_results_path = os.path.join(_eval_results_dir, "news_category.txt")
with open(_eval_results_path, "w", encoding="utf-8") as _f:
    _f.write(final_summary_text + "\n")
print(f"\nEval results saved to: {_eval_results_path}")
