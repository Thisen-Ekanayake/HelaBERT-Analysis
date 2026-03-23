"""
Fine-tuning LASER for Writing Style Classification — 5-Fold Cross Validation
— Balanced training via oversampling + weighted loss —
— Comparison baseline against HelaBERT —

Architecture:
    text → LASER encoder (frozen) → sentence embedding → LayerNorm → Dropout → Linear → num_labels

LASER is used as a frozen sentence-embedding encoder. Only the classification
head (LayerNorm → Dropout → Linear) is trained — identical to the LaBSE approach.

Mirrors the exact training/evaluation pipeline used for HelaBERT:
  • StratifiedKFold(n_splits=5) — same splits strategy
  • Oversampling applied ONLY to training split
  • Class weights recomputed per fold from raw (pre-oversample) distribution
  • Same Trainer, TrainingArguments, EarlyStoppingCallback settings
  • Same metrics: accuracy, macro-F1, weighted-F1, precision, recall
  • OOF report generated at end
  • W&B logging enabled

Requires: pip install laserembeddings
          python -m laserembeddings download-models

Task:    Writing Style Classification
Data:    data/Writing-style-classification/train/writing_style_train.csv
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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
    EarlyStoppingCallback,
)
import random
import wandb

# LASER import — requires laserembeddings package
try:
    from laserembeddings import Laser
    _laser_available = True
except ImportError:
    raise ImportError(
        "laserembeddings is not installed.\n"
        "Install with: pip install laserembeddings\n"
        "Then download models: python -m laserembeddings download-models"
    )

# ==================== CONFIGURATION ====================
print("=" * 80)
print("LASER FINE-TUNING — 5-FOLD CV  [WRITING STYLE CLASSIFICATION]")
print("=" * 80)

DATA_PATH        = "data/Writing-style-classification/train/writing_style_train.csv"
LASER_LANG       = "si"    # Sinhala ISO 639-1 code

NUM_LABELS                   = None  # resolved from data
# LASER produces 1024-dim embeddings
LASER_EMBED_DIM              = 1024
TRAIN_BATCH_SIZE             = 16   # larger batch OK since no backprop through encoder
EVAL_BATCH_SIZE              = 32
LEARNING_RATE                = 5e-05
NUM_EPOCHS                   = 5
WARMUP_RATIO                 = 0.05
WEIGHT_DECAY                 = 0.05
GRADIENT_ACCUMULATION_STEPS  = 1    # no encoder — lighter compute
EARLY_STOPPING_PATIENCE      = 3

N_FOLDS           = 5
OVERSAMPLE_TRAIN  = True
USE_CLASS_WEIGHTS = True

OUTPUT_DIR     = "LASER_finetuned_writing_style_cv"
BEST_MODEL_DIR = f"{OUTPUT_DIR}/best_model"

RANDOM_SEED = 42
USE_FP16    = False   # LASER embeddings are float32; FP16 not needed
NUM_WORKERS = 2

USE_WANDB      = True
WANDB_PROJECT  = "LASER-writing-style-finetuning"
WANDB_GROUP    = f"5fold_cv_lr{LEARNING_RATE}_bs{TRAIN_BATCH_SIZE}"
WANDB_ENTITY   = None

print(f"\n✓ Model:           LASER (laserembeddings)")
print(f"  Language:        {LASER_LANG}")
print(f"  Task:            Writing Style Classification")
print(f"  Embedding dim:   {LASER_EMBED_DIM}")
print(f"  Frozen encoder:  Yes")
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
    print("CPU only")
assert os.path.exists(DATA_PATH), f"Data not found: {DATA_PATH}"
print("All paths verified")


# ==================== LOAD LASER ====================
print("\n" + "=" * 80)
print("LOADING LASER ENCODER")
print("=" * 80)
laser = Laser()
print("✓ LASER loaded")


def encode_texts(texts, batch_size=256):
    """Encode a list of texts to LASER embeddings [N, 1024]."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb   = laser.embed_sentences(batch, lang=LASER_LANG)
        embeddings.append(emb)
    return np.vstack(embeddings).astype(np.float32)


# ==================== DATASET ====================
class LaserEmbeddingDataset(Dataset):
    """
    Pre-computed LASER embeddings — avoids re-encoding each sample every epoch.
    Mirrors HelaBERT's Dataset interface for Trainer compatibility.
    """

    def __init__(self, embeddings: np.ndarray, labels: list):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels     = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_embeds': self.embeddings[idx],   # [1024]
            'labels':       torch.tensor(self.labels[idx], dtype=torch.long),
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
    """Identical metric set to HelaBERT."""
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
class LaserClassificationModel(nn.Module):
    """
    LASER frozen encoder → pre-computed embedding → LayerNorm → Dropout → Linear

    Mirrors HelaBERT's classifier head exactly.
    Input is a pre-computed embedding vector [B, 1024] rather than token IDs.
    """

    def __init__(self, embed_dim, num_labels, dropout=0.1):
        super().__init__()
        self.embed_dim  = embed_dim
        self.norm       = nn.LayerNorm(embed_dim)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(self, input_embeds, labels=None, **kwargs):
        logits = self.classifier(self.dropout(self.norm(input_embeds)))

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
    """Load a fresh classifier head for each fold."""
    model = LaserClassificationModel(embed_dim=LASER_EMBED_DIM, num_labels=num_labels)
    return model


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
df['labels'] = df['labels'].astype(str).str.strip()
df = df[df['comments'].str.len() > 0].reset_index(drop=True)
print(f"✓ Loaded {len(df):,} samples")

# ==================== ENCODE LABELS ====================
print("\n" + "=" * 80)
print("ENCODING LABELS")
print("=" * 80)
le = LabelEncoder()
le.fit(sorted(df['labels'].unique()))
df['label_id'] = le.transform(df['labels'])
NUM_LABELS  = len(le.classes_)
id_to_label = {i: lbl for i, lbl in enumerate(le.classes_)}

os.makedirs(OUTPUT_DIR, exist_ok=True)
pd.DataFrame({'label_id': list(id_to_label.keys()),
              'label_name': list(id_to_label.values())
              }).to_csv(f"{OUTPUT_DIR}/label_mapping.csv", index=False)
print(f"✓ {NUM_LABELS} labels: {', '.join(le.classes_)}")
for idx, lbl in sorted(id_to_label.items()):
    cnt = (df['label_id'] == idx).sum()
    print(f"  [{idx}] {lbl:20s}: {cnt:5d}")

all_texts  = df['comments'].tolist()
all_labels = df['label_id'].tolist()


# ==================== PRE-COMPUTE ALL LASER EMBEDDINGS ====================
print("\n" + "=" * 80)
print("PRE-COMPUTING LASER EMBEDDINGS  (all {len(all_texts)} samples)")
print("=" * 80)
print("This is done once upfront so folds reuse the same embeddings...")
all_embeddings = encode_texts(all_texts)
print(f"✓ Embeddings shape: {all_embeddings.shape}")


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
    fold_train_texts  = [all_texts[i]     for i in train_idx]
    fold_train_embeds = all_embeddings[list(train_idx)]
    fold_train_labels = [all_labels[i]    for i in train_idx]

    fold_val_texts    = [all_texts[i]     for i in val_idx]
    fold_val_embeds   = all_embeddings[list(val_idx)]
    fold_val_labels   = [all_labels[i]    for i in val_idx]

    print(f"Train: {len(fold_train_texts)} samples  |  Val: {len(fold_val_texts)} samples")

    # ── Oversample (train split only) ───────────────────────────────────────
    if OVERSAMPLE_TRAIN:
        # oversample both texts and embeddings together
        stdlib_random.seed(RANDOM_SEED + fold_idx)
        counts    = Counter(fold_train_labels)
        max_count = max(counts.values())
        bal_texts, bal_embeds, bal_labels = (
            list(fold_train_texts),
            list(fold_train_embeds),
            list(fold_train_labels)
        )
        for label, count in counts.items():
            needed  = max_count - count
            if needed == 0:
                continue
            indices = [i for i, l in enumerate(fold_train_labels) if l == label]
            extras  = stdlib_random.choices(indices, k=needed)
            bal_texts  += [fold_train_texts[i]  for i in extras]
            bal_embeds += [fold_train_embeds[i] for i in extras]
            bal_labels += [fold_train_labels[i] for i in extras]
        combined = list(zip(bal_texts, bal_embeds, bal_labels))
        stdlib_random.shuffle(combined)
        fold_train_texts, fold_train_embeds, fold_train_labels = zip(*combined)
        fold_train_texts  = list(fold_train_texts)
        fold_train_embeds = np.stack(fold_train_embeds)
        fold_train_labels = list(fold_train_labels)
        print(f"After oversampling: {len(fold_train_texts)} train samples")

    # ── Class weights (raw pre-oversample distribution) ─────────────────────
    raw_fold_labels = [all_labels[i] for i in train_idx]
    fold_weights    = compute_class_weights(raw_fold_labels, NUM_LABELS)
    if USE_CLASS_WEIGHTS:
        print("Class weights:")
        for i, w in enumerate(fold_weights):
            print(f"  Label {i}: {w.item():.4f}")

    # ── Datasets ────────────────────────────────────────────────────────────
    train_ds = LaserEmbeddingDataset(fold_train_embeds, fold_train_labels)
    val_ds   = LaserEmbeddingDataset(fold_val_embeds,   fold_val_labels)

    # ── Fresh model ─────────────────────────────────────────────────────────
    print(f"Loading fresh model for fold {fold_idx}...")
    model = load_fresh_model(NUM_LABELS)
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
                "model":                  "LASER",
                "laser_lang":             LASER_LANG,
                "fold":                   fold_idx,
                "n_folds":                N_FOLDS,
                "frozen_encoder":         True,
                "embed_dim":              LASER_EMBED_DIM,
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
        fp16=False,
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

    fold_m = {
        'fold':        fold_idx,
        'accuracy':    eval_results['eval_accuracy'],
        'precision':   eval_results['eval_precision'],
        'recall':      eval_results['eval_recall'],
        'f1':          eval_results['eval_f1'],
        'f1_weighted': eval_results['eval_f1_weighted'],
    }
    fold_metrics.append(fold_m)

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
            fig = ff.create_annotated_heatmap(z=cm, colorscale='Blues', showscale=True)
            fig.update_layout(title=f"Confusion Matrix — Fold {fold_idx}")
            wandb.log({"confusion_matrix": fig})
        except ImportError:
            pass

    if fold_m['f1'] > best_fold_f1:
        best_fold_f1  = fold_m['f1']
        best_fold_idx = fold_idx
        os.makedirs(BEST_MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), f"{BEST_MODEL_DIR}/pytorch_model.bin")
        with open(f"{BEST_MODEL_DIR}/model_info.json", "w") as fh:
            json.dump({
                "model_name":  "LASER",
                "laser_lang":  LASER_LANG,
                "task":        "Writing Style Classification",
                "embed_dim":   LASER_EMBED_DIM,
                "num_labels":  NUM_LABELS,
                "frozen_encoder": True,
            }, fh, indent=2)
        print(f"\nNew best model saved (fold {fold_idx}, F1={best_fold_f1:.4f}) → {BEST_MODEL_DIR}")

    print(f"\nFold {fold_idx} done — macro F1: {fold_m['f1']:.4f}")

    if USE_WANDB:
        wandb.finish()
        print(f"W&B fold {fold_idx} run finished")


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

pd.DataFrame(oof_confusion_matrix := confusion_matrix(oof_y_true, oof_y_pred)).to_csv(
    f"{OUTPUT_DIR}/oof_confusion_matrix.csv"
)
pd.DataFrame({
    'text':            all_oof_texts,
    'true_label':      oof_y_true,
    'predicted_label': oof_y_pred,
    'correct':         oof_y_true == oof_y_pred,
}).to_csv(f"{OUTPUT_DIR}/oof_predictions.csv", index=False)


# ==================== W&B CV SUMMARY RUN ====================
if USE_WANDB:
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
    wandb.finish()
    print(f"CV summary logged — group: {WANDB_GROUP}")


# ==================== FINAL SUMMARY ====================
cv_summary = {
    'Model':                'LASER',
    'Frozen Encoder':       'Yes',
    'Embed Dim':            LASER_EMBED_DIM,
    'Task':                 "Writing Style Classification",
    'N Folds':              N_FOLDS,
    'Num Classes':          NUM_LABELS,
    'Total Samples':        len(all_texts),
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
final_summary_lines.append(f"5-FOLD CV COMPLETE — LASER / Writing Style Classification")
final_summary_lines.append("=" * 80)
final_summary_lines.append(f"\nOutputs saved to: {OUTPUT_DIR}/")
if USE_WANDB and wandb_group_url:
    final_summary_lines.append(f"W&B group: {wandb_group_url}")

final_summary_text = "\n".join(final_summary_lines)
print(final_summary_text)

# ==================== SAVE EVAL RESULTS ====================
_eval_results_dir = os.path.join("eval_results", "LASER")
os.makedirs(_eval_results_dir, exist_ok=True)
_eval_results_path = os.path.join(_eval_results_dir, "writing_style.txt")
with open(_eval_results_path, "w", encoding="utf-8") as _f:
    _f.write(final_summary_text + "\n")
print(f"\nEval results saved to: {_eval_results_path}")