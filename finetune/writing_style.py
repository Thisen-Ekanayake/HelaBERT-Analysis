"""
BERT Fine-tuning for Writing Style Classification — 5-Fold Cross Validation
— Balanced training via oversampling + weighted loss —
— Stage 1: Baseline ([CLS] Classifier, no Cross-Attention) —

Architecture:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Comment text → BERT encoder → [CLS] → comment_vec  [B, H]               │
  │                                                                          │
  │  Classifier:                                                             │
  │    comment_vec → LayerNorm → Dropout → Linear → num_labels               │
  └──────────────────────────────────────────────────────────────────────────┘

  Single BERT encoder. No article body chunking, no cross-attention.
  Input = comment only (truncated to COMMENT_MAX_LENGTH tokens).

  Labels are string names — auto-encoded via LabelEncoder.
  The resulting label_map.json is saved alongside model outputs.

  This is the direct baseline for comparing against the Cross-Attention
  (Stage 2) script. All other settings — oversampling, class weights,
  Trainer, W&B, CV strategy — are kept identical so results are comparable.

Cross-validation strategy:
  • StratifiedKFold(n_splits=5) preserves class distribution in every fold
  • Oversampling applied ONLY to each fold's training split (never the val split)
  • Class weights recomputed per fold from that fold's raw training distribution
  • Fresh model loaded at the start of every fold
  • Best model across all folds (highest val macro-F1) is saved as the final model
  • Mean ± std reported across all folds at the end

Expected CSV format:
    body, comments, labels
    "<article body>", "<sinhala text>", "LABEL_NAME"
    ...
  (body column is ignored in this baseline — comment only)
"""

import os
import json
import traceback
from collections import Counter
import random as stdlib_random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import sentencepiece as spm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertModel,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    EarlyStoppingCallback,
)
import random
import wandb

# ==================== CONFIGURATION ====================
print("=" * 80)
print("BERT FINE-TUNING — 5-FOLD CV  [BASELINE: CLS CLASSIFIER]")
print("=" * 80)

BERT_MODEL_PATH  = "HelaBERT"
TOKENIZER_MODEL  = "tokenizer/unigram_32000_0.9995.model"
BERT_CONFIG_FILE = "HelaBERT/config.json"
DATA_PATH        = "data/Writing-style-classification/train/writing_style_train.csv"

# Training hyperparameters  (kept identical to cross-attention script)
NUM_LABELS                   = None   # resolved from data via LabelEncoder
COMMENT_MAX_LENGTH           = 256    # sequence length
TRAIN_BATCH_SIZE             = 4
EVAL_BATCH_SIZE              = 8
LEARNING_RATE                = 2e-5
NUM_EPOCHS                   = 4      # early stopping decides actual epoch count
WARMUP_RATIO                 = 0.05
WEIGHT_DECAY                 = 0.05
GRADIENT_ACCUMULATION_STEPS  = 4      # effective batch = 16
EARLY_STOPPING_PATIENCE      = 3

# Cross-validation
N_FOLDS = 5

# Balancing
OVERSAMPLE_TRAIN  = True
USE_CLASS_WEIGHTS = True

# Output
OUTPUT_DIR      = "HelaBERT_finetuned_writing_style_baseline_cv"
BEST_MODEL_DIR  = f"{OUTPUT_DIR}/best_model"
LABEL_MAP_PATH  = f"{OUTPUT_DIR}/label_map.json"
STAGE_TAG       = "baseline_cls"

# Misc
RANDOM_SEED    = 42
USE_FP16       = True
NUM_WORKERS    = 2
USE_WANDB      = True
WANDB_PROJECT  = "bert-writing-style-finetuning"
WANDB_GROUP    = f"5fold_cv_baseline_lr{LEARNING_RATE}_bs{TRAIN_BATCH_SIZE}"
WANDB_ENTITY   = None

print(f"\n✓ Config loaded — {N_FOLDS}-fold CV, oversampling={'on' if OVERSAMPLE_TRAIN else 'off'}, "
      f"class_weights={'on' if USE_CLASS_WEIGHTS else 'off'}")
print(f"  Architecture   : BERT [CLS] → LayerNorm → Dropout → Linear  (no cross-attention)")
print(f"  Comment max length: {COMMENT_MAX_LENGTH}")
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

assert os.path.exists(BERT_MODEL_PATH),  f"{BERT_MODEL_PATH} not found"
assert os.path.exists(TOKENIZER_MODEL),  f"{TOKENIZER_MODEL} not found"
assert os.path.exists(DATA_PATH),        f"{DATA_PATH} not found"
print("All paths verified")


# ==================== TOKENIZER ====================
print("\n" + "=" * 80)
print("LOADING TOKENIZER")
print("=" * 80)

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)
PAD_ID = sp.pad_id()
print(f"SentencePiece loaded — vocab size: {sp.get_piece_size()}, PAD_ID: {PAD_ID}")


# ==================== DATASET CLASS ====================
class BaselineWritingStyleDataset(Dataset):
    """
    Each sample exposes:
      input_ids      [COMMENT_MAX_LENGTH]  — comment tokens
      attention_mask [COMMENT_MAX_LENGTH]
      labels         scalar

    Body column is intentionally ignored so the model receives
    exactly the same comment-side input as the cross-attention model,
    enabling a fair apples-to-apples comparison.
    """

    def __init__(self, texts, labels, sp_processor, comment_max_length=256):
        self.texts              = texts
        self.labels             = labels
        self.sp                 = sp_processor
        self.comment_max_length = comment_max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids  = self.sp.encode(str(self.texts[idx]))[:self.comment_max_length]
        mask = [1] * len(ids)
        pad  = self.comment_max_length - len(ids)
        ids  += [PAD_ID] * pad
        mask += [0]      * pad

        return {
            'input_ids':      torch.tensor(ids,              dtype=torch.long),
            'attention_mask': torch.tensor(mask,             dtype=torch.long),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ==================== HELPERS ====================

def oversample(texts, labels, seed=42):
    """Oversample minority classes to match majority class count."""
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
    """Inverse-frequency weights normalised so they sum to num_labels."""
    counts  = Counter(labels)
    weights = torch.tensor(
        [1.0 / counts.get(i, 1) for i in range(num_labels)],
        dtype=torch.float,
    )
    weights = weights / weights.sum() * num_labels
    return weights


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    preds  = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    return {
        'accuracy':    accuracy_score(labels, preds),
        'precision':   precision_score(labels, preds, average='macro',    zero_division=0),
        'recall':      recall_score(labels,    preds, average='macro',    zero_division=0),
        'f1':          f1_score(labels,        preds, average='macro',    zero_division=0),
        'f1_weighted': f1_score(labels,        preds, average='weighted', zero_division=0),
    }


def print_metric(key: str, value) -> None:
    if isinstance(value, float):
        print(f"    {key}: {value:.4f}")
    else:
        print(f"    {key}: {value}")


# ==================== MODEL ====================
class BaselineWritingStyleModel(nn.Module):
    """
    Shared BERT [CLS] classifier — baseline (no cross-attention).

    Pipeline:
        comment → BERT → [CLS] → LayerNorm → Dropout → Linear → num_labels

    Returns a dict with 'logits' (and optionally 'loss') so it is
    compatible with HuggingFace Trainer out of the box.
    """

    def __init__(self, bert, hidden_size, num_labels, dropout=0.1):
        super().__init__()
        self.bert        = bert
        self.hidden_size = hidden_size
        self.norm        = nn.LayerNorm(hidden_size)
        self.dropout     = nn.Dropout(dropout)
        self.classifier  = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        out     = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = out.last_hidden_state[:, 0, :]          # [B, H]
        logits  = self.classifier(self.dropout(self.norm(cls_vec)))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {'loss': loss, 'logits': logits}


# ==================== CUSTOM TRAINER ====================
class BaselineTrainer(Trainer):
    """Trainer subclass that optionally applies class-weight loss."""

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


def load_fresh_model(bert_model_path, bert_config_file, num_labels):
    """Load a fresh baseline BERT classifier for each fold."""
    if os.path.exists(bert_config_file):
        bert_config = BertConfig.from_json_file(bert_config_file)
        print(f"Config from {bert_config_file}")
    else:
        try:
            bert_config = BertConfig.from_pretrained(bert_model_path)
            print("Config from model dir")
        except Exception:
            bert_config = None
            print("Config not found — using defaults")

    try:
        bert_backbone = BertModel.from_pretrained(bert_model_path)
        print("BertModel loaded")
    except Exception as e:
        print(f"BertModel failed ({e}), extracting from MLM checkpoint...")
        mlm           = BertForMaskedLM.from_pretrained(bert_model_path)
        bert_backbone = mlm.bert
        if bert_config is None:
            bert_config = mlm.config
        print("BERT encoder extracted from MLM checkpoint")

    hidden_size = (bert_config.hidden_size if bert_config
                   else bert_backbone.config.hidden_size)

    model = BaselineWritingStyleModel(
        bert=bert_backbone,
        hidden_size=hidden_size,
        num_labels=num_labels,
    )
    return model, bert_config, hidden_size


# ==================== LOAD DATA ====================
print("\n" + "=" * 80)
print("LOADING DATASET")
print("=" * 80)

try:
    df = pd.read_csv(DATA_PATH)
except pd.errors.ParserError:
    df = pd.read_csv(DATA_PATH, engine='python', on_bad_lines='skip')

df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

# Detect comment and label columns
possible_comment_cols = [c for c in df.columns if 'comment' in c.lower()]
possible_label_cols   = [c for c in df.columns if 'label'   in c.lower()]

if possible_comment_cols and possible_label_cols:
    rename = {possible_comment_cols[0]: 'comment', possible_label_cols[0]: 'label'}
    df = df.rename(columns=rename)
else:
    df = df.iloc[:, -2:]
    df.columns = ['comment', 'label']

df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c], errors='ignore').dropna(subset=['comment', 'label'])
df['comment'] = df['comment'].astype(str).str.strip()
df['label']   = df['label'].astype(str).str.strip()

# Encode string labels → integers via LabelEncoder
le         = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'].tolist())
NUM_LABELS  = len(le.classes_)
label_map   = {i: name for i, name in enumerate(le.classes_)}

print(f"Loaded {len(df)} samples, {NUM_LABELS} classes")
print("\nFull dataset label distribution:")
for lbl_idx, lbl_name in label_map.items():
    cnt = (df['label_enc'] == lbl_idx).sum()
    print(f"  Label {lbl_idx} ({lbl_name:20s}): {cnt:6d} ({100 * cnt / len(df):.1f}%)")

# Save label map
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(LABEL_MAP_PATH, "w") as f:
    json.dump({str(k): v for k, v in label_map.items()}, f, ensure_ascii=False, indent=2)
print(f"Label map saved → {LABEL_MAP_PATH}")

all_texts  = df['comment'].tolist()
all_labels = df['label_enc'].tolist()


# ==================== CROSS-VALIDATION LOOP ====================
print("\n" + "=" * 80)
print(f"STARTING {N_FOLDS}-FOLD CROSS VALIDATION")
print("=" * 80)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

fold_metrics:  list[dict] = []
all_y_true:    list[int]  = []
all_y_pred:    list[int]  = []
all_oof_texts: list[str]  = []
best_fold_f1  = -1.0
best_fold_idx = -1
wandb_group_url = None

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_texts, all_labels), start=1):
    print("\n" + "=" * 80)
    print(f"FOLD {fold_idx} / {N_FOLDS}")
    print("=" * 80)

    fold_output_dir = f"{OUTPUT_DIR}/fold_{fold_idx}"
    os.makedirs(fold_output_dir, exist_ok=True)

    # ==================== Split ====================
    fold_train_texts  = [all_texts[i]  for i in train_idx]
    fold_train_labels = [all_labels[i] for i in train_idx]
    fold_val_texts    = [all_texts[i]  for i in val_idx]
    fold_val_labels   = [all_labels[i] for i in val_idx]

    print(f"Train: {len(fold_train_texts)} samples  |  Val: {len(fold_val_texts)} samples")
    print("Train label distribution (before oversampling):")
    for lbl, cnt in sorted(Counter(fold_train_labels).items()):
        print(f"  Label {lbl} ({label_map.get(lbl, '?')}): {cnt}")

    # ==================== Oversample (train only) ====================
    if OVERSAMPLE_TRAIN:
        fold_train_texts, fold_train_labels = oversample(
            fold_train_texts, fold_train_labels, seed=RANDOM_SEED + fold_idx
        )
        print(f"After oversampling: {len(fold_train_texts)} train samples")
        print("Train label distribution (after oversampling):")
        for lbl, cnt in sorted(Counter(fold_train_labels).items()):
            print(f"  Label {lbl} ({label_map.get(lbl, '?')}): {cnt}")

    # Class weights (computed from raw pre-oversample distribution)
    raw_fold_labels = [all_labels[i] for i in train_idx]
    fold_weights    = compute_class_weights(raw_fold_labels, NUM_LABELS)
    if USE_CLASS_WEIGHTS:
        print("Class weights:")
        for i, w in enumerate(fold_weights):
            print(f"  Label {i} ({label_map.get(i, '?')}): {w.item():.4f}")

    # ==================== Datasets ====================
    train_ds = BaselineWritingStyleDataset(
        fold_train_texts, fold_train_labels, sp,
        comment_max_length=COMMENT_MAX_LENGTH,
    )
    val_ds = BaselineWritingStyleDataset(
        fold_val_texts, fold_val_labels, sp,
        comment_max_length=COMMENT_MAX_LENGTH,
    )

    # ==================== Fresh model ====================
    print(f"Loading fresh model for fold {fold_idx}...")
    model, bert_config, hidden_size = load_fresh_model(BERT_MODEL_PATH, BERT_CONFIG_FILE, NUM_LABELS)
    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_p:,} total, {trainable_p:,} trainable")

    # ==================== W&B ====================
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
                "fold":                   fold_idx,
                "n_folds":                N_FOLDS,
                "architecture":           "BERT [CLS] baseline (no cross-attention)",
                "stage":                  STAGE_TAG,
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
                "num_labels":             NUM_LABELS,
                "label_map":              label_map,
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

    # ==================== Training args ====================
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
        eval_steps=500,
        save_steps=500,
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

    # ==================== Trainer ====================
    early_stop = EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)

    trainer = BaselineTrainer(
        class_weights=fold_weights if USE_CLASS_WEIGHTS else None,
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics, callbacks=[early_stop],
    )

    # ==================== Train ====================
    try:
        train_result = trainer.train()
        print(f"\n  Fold {fold_idx} training complete.")
        for k, v in train_result.metrics.items():
            print_metric(k, v)

    except KeyboardInterrupt:
        print(f"\nInterrupted at fold {fold_idx}.")
        trainer.save_model(f"{fold_output_dir}/interrupted_model")

    except Exception as e:
        tb = traceback.format_exc()
        print(f"\nFold {fold_idx} failed: {e}\n{tb}")

    # ==================== Evaluate ====================
    print(f"\nEvaluating fold {fold_idx}...")
    eval_results = trainer.evaluate()
    print(f"Fold {fold_idx} eval metrics:")
    for k, v in eval_results.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    # ==================== Predictions & per-class report ====================
    preds_out = trainer.predict(val_ds)
    y_pred    = np.argmax(preds_out.predictions, axis=-1)
    y_true    = np.array(fold_val_labels)

    all_y_true.extend(y_true.tolist())
    all_y_pred.extend(y_pred.tolist())
    all_oof_texts.extend(fold_val_texts)

    target_names = list(le.classes_)
    print(f"\nClassification report — Fold {fold_idx}:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    # ==================== Save per-fold predictions ====================
    fold_conf = [
        torch.softmax(torch.tensor(preds_out.predictions[i]), dim=0)[y_pred[i]].item()
        for i in range(len(y_pred))
    ]
    pd.DataFrame({
        'text':               fold_val_texts,
        'true_label':         le.inverse_transform(y_true),
        'predicted_label':    le.inverse_transform(y_pred),
        'true_label_idx':     y_true,
        'predicted_label_idx': y_pred,
        'correct':            y_true == y_pred,
        'confidence':         fold_conf,
    }).to_csv(f"{fold_output_dir}/predictions.csv", index=False)

    # ==================== Per-class metrics ====================
    prec_pc, rec_pc, f1_pc, sup_pc = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    per_class_df = pd.DataFrame({
        'Class':     range(NUM_LABELS),
        'Style':     [label_map.get(i, f"Style_{i}") for i in range(NUM_LABELS)],
        'Precision': prec_pc,
        'Recall':    rec_pc,
        'F1-Score':  f1_pc,
        'Support':   sup_pc,
    })
    per_class_df.to_csv(f"{fold_output_dir}/per_class_metrics.csv", index=False)

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(
        cm,
        index=[f"True_{label_map.get(i, i)}"   for i in range(NUM_LABELS)],
        columns=[f"Pred_{label_map.get(i, i)}" for i in range(NUM_LABELS)],
    ).to_csv(f"{fold_output_dir}/confusion_matrix.csv")

    # ==================== Store fold-level metrics ====================
    fold_m = {
        'fold':        fold_idx,
        'accuracy':    eval_results['eval_accuracy'],
        'precision':   eval_results['eval_precision'],
        'recall':      eval_results['eval_recall'],
        'f1':          eval_results['eval_f1'],
        'f1_weighted': eval_results['eval_f1_weighted'],
    }
    fold_metrics.append(fold_m)

    # ==================== W&B fold summary ====================
    if USE_WANDB:
        wandb.log({
            "fold_summary/accuracy":    fold_m['accuracy'],
            "fold_summary/f1":          fold_m['f1'],
            "fold_summary/f1_weighted": fold_m['f1_weighted'],
            "fold_summary/precision":   fold_m['precision'],
            "fold_summary/recall":      fold_m['recall'],
        })
        wandb.log({"per_class_metrics": wandb.Table(dataframe=per_class_df)})
        try:
            import plotly.figure_factory as ff
            fig = ff.create_annotated_heatmap(
                z=cm,
                x=[f"Pred_{label_map.get(i, i)}" for i in range(NUM_LABELS)],
                y=[f"True_{label_map.get(i, i)}" for i in range(NUM_LABELS)],
                colorscale='Blues',
                showscale=True,
            )
            fig.update_layout(
                title=f"Confusion Matrix — Fold {fold_idx}",
                xaxis_title="Predicted",
                yaxis_title="True",
            )
            wandb.log({"confusion_matrix": fig})
        except ImportError:
            pass

    # ==================== Save best model ====================
    if fold_m['f1'] > best_fold_f1:
        best_fold_f1  = fold_m['f1']
        best_fold_idx = fold_idx
        os.makedirs(BEST_MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), f"{BEST_MODEL_DIR}/pytorch_model.bin")
        if bert_config:
            bert_config.save_pretrained(BEST_MODEL_DIR)
        pd.DataFrame([{
            'stage':              STAGE_TAG,
            'hidden_size':        hidden_size,
            'num_labels':         NUM_LABELS,
            'comment_max_length': COMMENT_MAX_LENGTH,
        }]).to_csv(f"{BEST_MODEL_DIR}/arch_config.csv", index=False)
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
print(f"Best model saved to: {BEST_MODEL_DIR}")


# ==================== OUT-OF-FOLD (OOF) REPORT ====================
print("\n" + "=" * 80)
print("OUT-OF-FOLD (OOF) REPORT — FULL DATASET")
print("=" * 80)

oof_y_true   = np.array(all_y_true)
oof_y_pred   = np.array(all_y_pred)
target_names = list(le.classes_)
print(classification_report(oof_y_true, oof_y_pred, target_names=target_names, digits=4))

oof_f1   = f1_score(oof_y_true, oof_y_pred, average='macro',    zero_division=0)
oof_f1_w = f1_score(oof_y_true, oof_y_pred, average='weighted', zero_division=0)
oof_acc  = accuracy_score(oof_y_true, oof_y_pred)

print(f"OOF macro F1:    {oof_f1:.4f}")
print(f"OOF weighted F1: {oof_f1_w:.4f}")
print(f"OOF accuracy:    {oof_acc:.4f}")

oof_cm = confusion_matrix(oof_y_true, oof_y_pred)
pd.DataFrame(
    oof_cm,
    index=[f"True_{label_map.get(i, i)}"   for i in range(NUM_LABELS)],
    columns=[f"Pred_{label_map.get(i, i)}" for i in range(NUM_LABELS)],
).to_csv(f"{OUTPUT_DIR}/oof_confusion_matrix.csv")

pd.DataFrame({
    'text':            all_oof_texts,
    'true_label':      le.inverse_transform(oof_y_true),
    'predicted_label': le.inverse_transform(oof_y_pred),
    'correct':         oof_y_true == oof_y_pred,
}).to_csv(f"{OUTPUT_DIR}/oof_predictions.csv", index=False)

print(f"\nOOF confusion matrix → {OUTPUT_DIR}/oof_confusion_matrix.csv")
print(f"OOF predictions      → {OUTPUT_DIR}/oof_predictions.csv")


# ==================== W&B CROSS-VAL SUMMARY RUN ====================
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
        "label_map":       wandb.Table(
            dataframe=pd.DataFrame(list(label_map.items()), columns=["idx", "style"])
        ),
    })
    try:
        import plotly.figure_factory as ff
        fig = ff.create_annotated_heatmap(
            z=oof_cm,
            x=[f"Pred_{label_map.get(i, i)}" for i in range(NUM_LABELS)],
            y=[f"True_{label_map.get(i, i)}" for i in range(NUM_LABELS)],
            colorscale='Blues',
            showscale=True,
        )
        fig.update_layout(
            title="OOF Confusion Matrix — All Folds",
            xaxis_title="Predicted",
            yaxis_title="True",
        )
        wandb.log({"oof_confusion_matrix": fig})
    except ImportError:
        pass
    wandb.finish()
    print(f"CV summary logged — group: {WANDB_GROUP}")


# ==================== FINAL SUMMARY ====================
cv_summary = {
    'Model':                'BERT with SentencePiece',
    'Architecture':         'BERT [CLS] baseline (no cross-attention)',
    'Stage':                STAGE_TAG,
    'Task':                 'Writing Style Classification',
    'Balancing':            'Oversample + Class Weights',
    'N Folds':              N_FOLDS,
    'Num Classes':          NUM_LABELS,
    'Total Samples':        len(all_texts),
    'Comment Max Length':   COMMENT_MAX_LENGTH,
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
    'Label Map Path':       LABEL_MAP_PATH,
}

pd.DataFrame([cv_summary]).to_csv(f'{OUTPUT_DIR}/cv_summary.csv', index=False)

print("\n" + "=" * 80)
print("FINAL CROSS-VALIDATION SUMMARY")
print("=" * 80)
for k, v in cv_summary.items():
    print(f"  {k:<28}: {v}")

print("\n" + "=" * 80)
print("5-FOLD CROSS-VALIDATION (BASELINE CLS) COMPLETE!")
print("=" * 80)
print(f"\nOutputs saved to: {OUTPUT_DIR}/")
print(f"fold_1/ … fold_{N_FOLDS}/       per-fold predictions, metrics, confusion matrix")
print(f"best_model/                    best model weights (fold {best_fold_idx})")
print(f"cv_fold_metrics.csv            per-fold metric table")
print(f"cv_summary.csv                 overall CV summary")
print(f"oof_predictions.csv            out-of-fold predictions (full dataset)")
print(f"oof_confusion_matrix.csv       OOF confusion matrix")
print(f"label_map.json                 string label ↔ integer index mapping")
if USE_WANDB and wandb_group_url:
    print(f"\nW&B group: {wandb_group_url}")