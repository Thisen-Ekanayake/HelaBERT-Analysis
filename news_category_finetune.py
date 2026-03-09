"""
BERT Fine-tuning for News Category Classification — 5-Fold Cross Validation
— Balanced training via oversampling + weighted loss (Option 3) —

Cross-validation strategy:
  • StratifiedKFold(n_splits=5) preserves class distribution in every fold
  • Oversampling applied ONLY to each fold's training split (never the val split)
  • Class weights recomputed per fold from that fold's raw training distribution
  • Fresh model loaded at the start of every fold
  • Best model across all folds (highest val macro-F1) is saved as the final model
  • Mean ± std reported across all folds at the end

Expected CSV format:
    comments, labels
    <sinhala text>, <0-4>
    ...
"""

import os
import smtplib
import traceback
import datetime
from collections import Counter
import random as stdlib_random
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import sentencepiece as spm
from sklearn.model_selection import StratifiedKFold
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
    BertForSequenceClassification,
    BertForMaskedLM,
    BertModel,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    EarlyStoppingCallback,
)
import random
import wandb
from dotenv import load_dotenv

# ==================== LOAD ENVIRONMENT VARIABLES ====================
load_dotenv()


# ==================== EMAIL NOTIFICATION SETUP ====================

EMAIL_NOTIFICATIONS_ENABLED = os.getenv("EMAIL_NOTIFICATIONS_ENABLED", "true").lower() == "true"
EMAIL_SENDER      = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD    = os.getenv("EMAIL_APP_PASSWORD")
EMAIL_RECIPIENT   = os.getenv("EMAIL_RECIPIENT")
EMAIL_SMTP_HOST   = os.getenv("EMAIL_SMTP_HOST", "smtp.gmail.com")
EMAIL_SMTP_PORT   = int(os.getenv("EMAIL_SMTP_PORT", "587"))
TRAINING_JOB_NAME = os.getenv("TRAINING_news_category", "BERT News Category — 5-Fold CV")


def send_email_notification(subject: str, body_html: str, body_text: str = None):
    if not EMAIL_NOTIFICATIONS_ENABLED:
        print("ℹ️  Email notifications disabled")
        return
    if not all([EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENT]):
        print("⚠️  Email skipped — credentials not set in .env")
        return
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = EMAIL_RECIPIENT
        if body_text:
            msg.attach(MIMEText(body_text, "plain"))
        msg.attach(MIMEText(body_html, "html"))
        with smtplib.SMTP(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, msg.as_string())
        print(f"✉️  Email sent to: {EMAIL_RECIPIENT}")
    except Exception as e:
        print(f"⚠️  Failed to send email: {e}")


def _build_cv_success_email(cv_summary: dict, wandb_group_url: str = None) -> tuple:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = "".join(
        f"<tr><td style='padding:4px 12px'>{k}</td>"
        f"<td style='padding:4px 12px'><b>{v if isinstance(v, str) else f'{v:.4f}'}</b></td></tr>"
        for k, v in cv_summary.items()
    )
    wandb_section = f'<p>📊 <a href="{wandb_group_url}">View W&B group</a></p>' if wandb_group_url else ""
    html = f"""
    <html><body style='font-family:Arial,sans-serif;color:#222'>
      <h2 style='color:#2e7d32'>✅ 5-Fold CV Training Complete</h2>
      <p><b>Job:</b> {TRAINING_JOB_NAME}</p>
      <p><b>Finished at:</b> {timestamp}</p>
      <h3>Cross-Validation Summary</h3>
      <table border='1' cellspacing='0' cellpadding='0' style='border-collapse:collapse;font-size:14px'>
        <tr style='background:#e8f5e9'><th style='padding:6px 12px'>Metric</th><th style='padding:6px 12px'>Value</th></tr>
        {rows}
      </table>
      {wandb_section}
      <p style='color:#888;font-size:12px'>Sent by news_category_finetune_cv.py</p>
    </body></html>"""
    plain = (
        f"5-Fold CV Complete\nJob: {TRAINING_JOB_NAME}\nFinished at: {timestamp}\n\n"
        + "\n".join(f"{k}: {v}" for k, v in cv_summary.items())
        + (f"\n\nW&B: {wandb_group_url}" if wandb_group_url else "")
    )
    return f"✅ 5-Fold CV Complete — {TRAINING_JOB_NAME}", html, plain


def _build_failure_email(error: Exception, tb_str: str, fold: int = None) -> tuple:
    timestamp  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fold_label = f" (Fold {fold})" if fold is not None else ""
    tb_escaped = tb_str.replace("<", "&lt;").replace(">", "&gt;")
    html = f"""
    <html><body style='font-family:Arial,sans-serif;color:#222'>
      <h2 style='color:#c62828'>❌ Training Crashed{fold_label}</h2>
      <p><b>Job:</b> {TRAINING_JOB_NAME}</p>
      <p><b>Crashed at:</b> {timestamp}</p>
      <p><b>Error:</b> <span style='color:#c62828'>{type(error).__name__}: {error}</span></p>
      <h3>Traceback</h3>
      <pre style='background:#f5f5f5;padding:12px;font-size:12px'>{tb_escaped}</pre>
    </body></html>"""
    plain = f"Training Crashed{fold_label}\nJob: {TRAINING_JOB_NAME}\n\n{type(error).__name__}: {error}\n\n{tb_str}"
    return f"❌ Training Crashed{fold_label} — {TRAINING_JOB_NAME}", html, plain


# ==================== CONFIGURATION ====================
print("=" * 80)
print("BERT FINE-TUNING — 5-FOLD CROSS VALIDATION  [BALANCED — OPTION 3]")
print("=" * 80)

BERT_MODEL_PATH  = "HelaBERT"
TOKENIZER_MODEL  = "tokenizer/unigram_32000_0.9995.model"
BERT_CONFIG_FILE = "HelaBERT/config.json"
DATA_PATH        = "data/Sinhala-News-Category-classification/sinhala-news-categories.csv"

CATEGORY_NAMES = {0: "Category_0", 1: "Category_1", 2: "Category_2",
                  3: "Category_3", 4: "Category_4"}

# Training hyperparameters
NUM_LABELS                   = 5
MAX_LENGTH                   = 256
TRAIN_BATCH_SIZE             = 16
EVAL_BATCH_SIZE              = 16
LEARNING_RATE                = 3e-5
NUM_EPOCHS                   = 20    # early stopping decides actual epoch count
WARMUP_RATIO                 = 0.06
WEIGHT_DECAY                 = 0.01
GRADIENT_ACCUMULATION_STEPS  = 2     # effective batch = 32
EARLY_STOPPING_PATIENCE      = 3

# Cross-validation
N_FOLDS = 5

# Balancing
OVERSAMPLE_TRAIN  = True
USE_CLASS_WEIGHTS = True

# Output
OUTPUT_DIR     = "HelaBERT_finetuned_news_category_cv"
BEST_MODEL_DIR = f"{OUTPUT_DIR}/best_model"   # saved from the highest-F1 fold

# Misc
RANDOM_SEED    = 42
USE_FP16       = True
NUM_WORKERS    = 2
USE_WANDB      = True
WANDB_PROJECT  = "bert-news-category-finetuning"
WANDB_GROUP    = f"5fold_cv_lr{LEARNING_RATE}_bs{TRAIN_BATCH_SIZE}"  # groups all folds in W&B
WANDB_ENTITY   = None

print(f"\n✓ Config loaded — {N_FOLDS}-fold CV, oversampling={'on' if OVERSAMPLE_TRAIN else 'off'}, "
      f"class_weights={'on' if USE_CLASS_WEIGHTS else 'off'}")


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
    print("⚠️  CPU only — training will be slow")

assert os.path.exists(BERT_MODEL_PATH),  f"❌ {BERT_MODEL_PATH} not found"
assert os.path.exists(TOKENIZER_MODEL),  f"❌ {TOKENIZER_MODEL} not found"
assert os.path.exists(DATA_PATH),        f"❌ {DATA_PATH} not found"
print("✓ All paths verified")


# ==================== TOKENIZER ====================
print("\n" + "=" * 80)
print("LOADING TOKENIZER")
print("=" * 80)

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)
print(f"✓ SentencePiece loaded — vocab size: {sp.get_piece_size()}, PAD_ID: {sp.pad_id()}")


# ==================== DATASET CLASS ====================
class SentencePieceDataset(Dataset):
    def __init__(self, texts, labels, sp_processor, max_length=256):
        self.texts      = texts
        self.labels     = labels
        self.sp         = sp_processor
        self.max_length = max_length
        self.pad_id     = sp_processor.pad_id()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        token_ids = self.sp.encode(self.texts[idx])

        # Truncate to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        # Pad to max_length
        attn_mask = [1] * len(token_ids)
        pad_len   = self.max_length - len(token_ids)
        token_ids += [self.pad_id] * pad_len
        attn_mask += [0] * pad_len

        return {
            'input_ids':      torch.tensor(token_ids,        dtype=torch.long),
            'attention_mask': torch.tensor(attn_mask,        dtype=torch.long),
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
        bal_texts  += [texts[i]  for i in extras]
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


def load_fresh_model(bert_model_path, bert_config_file, num_labels):
    """Load a fresh copy of the model for each fold."""
    if os.path.exists(bert_config_file):
        config = BertConfig.from_json_file(bert_config_file)
    else:
        config = BertConfig.from_pretrained(bert_model_path)
    config.num_labels          = num_labels
    config.hidden_dropout_prob = 0.2

    try:
        # Attempt to load directly as a BertModel
        base     = BertModel.from_pretrained(bert_model_path)
        model    = BertForSequenceClassification(config)
        base_sd  = base.state_dict()
        model_sd = model.state_dict()
        for name, param in base_sd.items():
            if name in model_sd:
                model_sd[name].copy_(param)
        model.load_state_dict(model_sd, strict=False)
    except Exception:
        # Fall back to loading as a masked-LM checkpoint
        mlm      = BertForMaskedLM.from_pretrained(bert_model_path)
        model    = BertForSequenceClassification(config)
        mlm_sd   = mlm.state_dict()
        model_sd = model.state_dict()
        for name, param in mlm_sd.items():
            if name.startswith('bert.') and name in model_sd:
                model_sd[name].copy_(param)
        model.load_state_dict(model_sd, strict=False)

    return model, config


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
    """Print a single training/eval metric, handling both float and non-float values."""
    if isinstance(value, float):
        print(f"    {key}: {value:.4f}")
    else:
        print(f"    {key}: {value}")


class WeightedTrainer(Trainer):
    """Trainer subclass that applies per-class loss weighting."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.get("logits")
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss    = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ==================== LOAD DATA ====================
print("\n" + "=" * 80)
print("LOADING DATASET")
print("=" * 80)

try:
    df = pd.read_csv(DATA_PATH)
except pd.errors.ParserError:
    df = pd.read_csv(DATA_PATH, engine='python', on_bad_lines='skip')

df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

possible_comment_cols = [c for c in df.columns if 'comment' in c.lower()]
possible_label_cols   = [c for c in df.columns if 'label'   in c.lower()]

if possible_comment_cols and possible_label_cols:
    df = df.rename(columns={possible_comment_cols[0]: 'comment', possible_label_cols[0]: 'label'})
else:
    df = df.iloc[:, -2:]
    df.columns = ['comment', 'label']

df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c], errors='ignore').dropna()
df['comment'] = df['comment'].astype(str).str.strip()
df['label']   = df['label'].astype(str).str.strip().astype(int)

# Sync NUM_LABELS and CATEGORY_NAMES with the actual data
actual_num_labels = df['label'].nunique()
if actual_num_labels != NUM_LABELS:
    print(f"⚠️  Updating NUM_LABELS: {NUM_LABELS} → {actual_num_labels}")
    NUM_LABELS     = actual_num_labels
    CATEGORY_NAMES = {i: f"Category_{i}" for i in range(NUM_LABELS)}  # FIX: keep in sync

print(f"✓ Loaded {len(df)} samples, {NUM_LABELS} classes")
print("\nFull dataset label distribution:")
for lbl, cnt in df['label'].value_counts().sort_index().items():
    print(f"  Label {lbl} ({CATEGORY_NAMES.get(lbl, '?'):15s}): {cnt:6d} ({100 * cnt / len(df):.1f}%)")

all_texts  = df['comment'].tolist()
all_labels = df['label'].tolist()


# ==================== CROSS-VALIDATION LOOP ====================
print("\n" + "=" * 80)
print(f"STARTING {N_FOLDS}-FOLD CROSS VALIDATION")
print("=" * 80)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Storage for results across folds
fold_metrics:  list[dict] = []   # one dict per fold
all_y_true:    list[int]  = []   # pooled ground truth  (for OOF report)
all_y_pred:    list[int]  = []   # pooled predictions   (for OOF report)
all_oof_texts: list[str]  = []   # pooled input texts   (for OOF report)
best_fold_f1  = -1.0
best_fold_idx = -1
wandb_group_url = None

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_texts, all_labels), start=1):
    print("\n" + "=" * 80)
    print(f"FOLD {fold_idx} / {N_FOLDS}")
    print("=" * 80)

    fold_output_dir = f"{OUTPUT_DIR}/fold_{fold_idx}"
    os.makedirs(fold_output_dir, exist_ok=True)

    # ── Split ──────────────────────────────────────────────────────────────
    fold_train_texts  = [all_texts[i]  for i in train_idx]
    fold_train_labels = [all_labels[i] for i in train_idx]
    fold_val_texts    = [all_texts[i]  for i in val_idx]
    fold_val_labels   = [all_labels[i] for i in val_idx]

    print(f"  Train: {len(fold_train_texts)} samples  |  Val: {len(fold_val_texts)} samples")
    print("  Train label distribution (before oversampling):")
    for lbl, cnt in sorted(Counter(fold_train_labels).items()):
        print(f"    Label {lbl}: {cnt}")

    # ── Oversample (train only) ────────────────────────────────────────────
    if OVERSAMPLE_TRAIN:
        fold_train_texts, fold_train_labels = oversample(
            fold_train_texts, fold_train_labels, seed=RANDOM_SEED + fold_idx
        )
        print(f"  After oversampling: {len(fold_train_texts)} train samples")
        print("  Train label distribution (after oversampling):")
        for lbl, cnt in sorted(Counter(fold_train_labels).items()):
            print(f"    Label {lbl}: {cnt}")

    # ── Class weights (computed from raw pre-oversample distribution) ──────
    raw_fold_labels = [all_labels[i] for i in train_idx]
    fold_weights    = compute_class_weights(raw_fold_labels, NUM_LABELS)
    if USE_CLASS_WEIGHTS:
        print("  Class weights:")
        for i, w in enumerate(fold_weights):
            print(f"    Label {i}: {w.item():.4f}")

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = SentencePieceDataset(fold_train_texts, fold_train_labels, sp, MAX_LENGTH)
    val_ds   = SentencePieceDataset(fold_val_texts,   fold_val_labels,   sp, MAX_LENGTH)

    # ── Fresh model ────────────────────────────────────────────────────────
    print(f"  Loading fresh model for fold {fold_idx}...")
    model, config = load_fresh_model(BERT_MODEL_PATH, BERT_CONFIG_FILE, NUM_LABELS)
    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_p:,} total, {trainable_p:,} trainable")

    # ── W&B (one run per fold, all in the same group) ─────────────────────
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
                "learning_rate":          LEARNING_RATE,
                "epochs":                 NUM_EPOCHS,
                "train_batch_size":       TRAIN_BATCH_SIZE,
                "effective_batch":        TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
                "warmup_ratio":           WARMUP_RATIO,
                "weight_decay":           WEIGHT_DECAY,
                "max_length":             MAX_LENGTH,
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
        print(f"  W&B run: {run.get_url()}")

    # ── Training args ──────────────────────────────────────────────────────
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
        eval_strategy="epoch",
        save_strategy="epoch",
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

    # ── Trainer ────────────────────────────────────────────────────────────
    early_stop = EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)

    if USE_CLASS_WEIGHTS:
        trainer = WeightedTrainer(
            class_weights=fold_weights,
            model=model, args=training_args,
            train_dataset=train_ds, eval_dataset=val_ds,
            compute_metrics=compute_metrics, callbacks=[early_stop],
        )
    else:
        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=train_ds, eval_dataset=val_ds,
            compute_metrics=compute_metrics, callbacks=[early_stop],
        )

    # ── Train ──────────────────────────────────────────────────────────────
    try:
        train_result = trainer.train()
        print(f"\n  Fold {fold_idx} training complete.")
        for k, v in train_result.metrics.items():
            print_metric(k, v)  # FIX: handles non-float metric values safely

    except KeyboardInterrupt:
        print(f"\n  ⚠️  Interrupted at fold {fold_idx}.")
        trainer.save_model(f"{fold_output_dir}/interrupted_model")
        if USE_WANDB:
            wandb.finish(exit_code=1)
        _s, _h, _p = _build_failure_email(
            KeyboardInterrupt("Manually interrupted"), "User interrupted training.", fold=fold_idx
        )
        send_email_notification(_s, _h, _p)
        raise

    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n  ❌ Fold {fold_idx} failed: {e}")
        if USE_WANDB:
            wandb.finish(exit_code=1)
        _s, _h, _p = _build_failure_email(e, tb, fold=fold_idx)
        send_email_notification(_s, _h, _p)
        raise

    # ── Evaluate ───────────────────────────────────────────────────────────
    print(f"\n  Evaluating fold {fold_idx}...")
    eval_results = trainer.evaluate()
    print(f"  Fold {fold_idx} eval metrics:")
    for k, v in eval_results.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    # ── Predictions & per-class report ────────────────────────────────────
    preds_out = trainer.predict(val_ds)
    y_pred    = np.argmax(preds_out.predictions, axis=-1)
    y_true    = np.array(fold_val_labels)

    # Accumulate for OOF report (text included for traceability)
    all_y_true.extend(y_true.tolist())
    all_y_pred.extend(y_pred.tolist())
    all_oof_texts.extend(fold_val_texts)  # FIX: collect texts for OOF CSV

    target_names = [CATEGORY_NAMES.get(i, f"Category_{i}") for i in range(NUM_LABELS)]
    print(f"\n  Classification report — Fold {fold_idx}:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    # ── Save per-fold predictions ──────────────────────────────────────────
    fold_conf = [
        torch.softmax(torch.tensor(preds_out.predictions[i]), dim=0)[y_pred[i]].item()
        for i in range(len(y_pred))
    ]
    pd.DataFrame({
        'text':               fold_val_texts,
        'true_label':         y_true,
        'true_category':      [CATEGORY_NAMES.get(l, str(l)) for l in y_true],
        'predicted_label':    y_pred,
        'predicted_category': [CATEGORY_NAMES.get(l, str(l)) for l in y_pred],
        'correct':            y_true == y_pred,
        'confidence':         fold_conf,
    }).to_csv(f"{fold_output_dir}/predictions.csv", index=False)

    # ── Per-class metrics ──────────────────────────────────────────────────
    prec_pc, rec_pc, f1_pc, sup_pc = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    per_class_df = pd.DataFrame({
        'Class':     range(NUM_LABELS),
        'Category':  [CATEGORY_NAMES.get(i, f"Category_{i}") for i in range(NUM_LABELS)],
        'Precision': prec_pc,
        'Recall':    rec_pc,
        'F1-Score':  f1_pc,
        'Support':   sup_pc,
    })
    per_class_df.to_csv(f"{fold_output_dir}/per_class_metrics.csv", index=False)

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(
        cm,
        index=[f"True_{CATEGORY_NAMES.get(i, i)}"  for i in range(NUM_LABELS)],
        columns=[f"Pred_{CATEGORY_NAMES.get(i, i)}" for i in range(NUM_LABELS)],
    ).to_csv(f"{fold_output_dir}/confusion_matrix.csv")

    # ── Store fold-level metrics ───────────────────────────────────────────
    fold_m = {
        'fold':        fold_idx,
        'accuracy':    eval_results['eval_accuracy'],
        'precision':   eval_results['eval_precision'],
        'recall':      eval_results['eval_recall'],
        'f1':          eval_results['eval_f1'],
        'f1_weighted': eval_results['eval_f1_weighted'],
    }
    fold_metrics.append(fold_m)

    # ── W&B fold summary ──────────────────────────────────────────────────
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
                x=[f"Pred_{CATEGORY_NAMES.get(i, i)}" for i in range(NUM_LABELS)],
                y=[f"True_{CATEGORY_NAMES.get(i, i)}" for i in range(NUM_LABELS)],
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
        wandb.finish()
        print(f"  ✓ W&B fold {fold_idx} run finished")

    # ── Save best model ────────────────────────────────────────────────────
    if fold_m['f1'] > best_fold_f1:
        best_fold_f1  = fold_m['f1']
        best_fold_idx = fold_idx
        trainer.save_model(BEST_MODEL_DIR)
        if config:
            config.save_pretrained(BEST_MODEL_DIR)
        print(f"\n  ★ New best model saved (fold {fold_idx}, F1={best_fold_f1:.4f}) → {BEST_MODEL_DIR}")

    print(f"\n  Fold {fold_idx} done — macro F1: {fold_m['f1']:.4f}")


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
print(f"\n  Best single fold: Fold {best_fold_idx}  (macro F1 = {best_fold_f1:.4f})")
print(f"  Best model saved to: {BEST_MODEL_DIR}")


# ==================== OUT-OF-FOLD (OOF) REPORT ====================
print("\n" + "=" * 80)
print("OUT-OF-FOLD (OOF) REPORT — FULL DATASET")
print("=" * 80)

oof_y_true   = np.array(all_y_true)
oof_y_pred   = np.array(all_y_pred)
target_names = [CATEGORY_NAMES.get(i, f"Category_{i}") for i in range(NUM_LABELS)]
print(classification_report(oof_y_true, oof_y_pred, target_names=target_names, digits=4))

oof_f1   = f1_score(oof_y_true, oof_y_pred, average='macro',    zero_division=0)
oof_f1_w = f1_score(oof_y_true, oof_y_pred, average='weighted', zero_division=0)
oof_acc  = accuracy_score(oof_y_true, oof_y_pred)

print(f"OOF macro F1:    {oof_f1:.4f}")
print(f"OOF weighted F1: {oof_f1_w:.4f}")
print(f"OOF accuracy:    {oof_acc:.4f}")

# OOF confusion matrix
oof_cm = confusion_matrix(oof_y_true, oof_y_pred)
pd.DataFrame(
    oof_cm,
    index=[f"True_{CATEGORY_NAMES.get(i, i)}"  for i in range(NUM_LABELS)],
    columns=[f"Pred_{CATEGORY_NAMES.get(i, i)}" for i in range(NUM_LABELS)],
).to_csv(f"{OUTPUT_DIR}/oof_confusion_matrix.csv")

# OOF predictions CSV — includes original text for traceability
pd.DataFrame({
    'text':            all_oof_texts,   # FIX: text included for traceability
    'true_label':      oof_y_true,
    'predicted_label': oof_y_pred,
    'correct':         oof_y_true == oof_y_pred,
}).to_csv(f"{OUTPUT_DIR}/oof_predictions.csv", index=False)

print(f"\n✓ OOF confusion matrix → {OUTPUT_DIR}/oof_confusion_matrix.csv")
print(f"✓ OOF predictions      → {OUTPUT_DIR}/oof_predictions.csv")


# ==================== W&B CROSS-VAL SUMMARY RUN ====================
if USE_WANDB:
    print("\nLogging CV summary to W&B...")
    wandb.init(   # FIX: removed unused cv_summary_run variable
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
        fig = ff.create_annotated_heatmap(
            z=oof_cm,
            x=[f"Pred_{CATEGORY_NAMES.get(i, i)}" for i in range(NUM_LABELS)],
            y=[f"True_{CATEGORY_NAMES.get(i, i)}" for i in range(NUM_LABELS)],
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
    print(f"✓ CV summary logged — group: {WANDB_GROUP}")


# ==================== FINAL SUMMARY ====================
cv_summary = {
    'Model':                'BERT with SentencePiece',
    'Task':                 'News Category Classification',
    'Balancing':            'Oversample + Class Weights',
    'N Folds':              N_FOLDS,
    'Num Classes':          NUM_LABELS,
    'Total Samples':        len(all_texts),
    'Learning Rate':        LEARNING_RATE,
    'Effective Batch Size': TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
    'Max Length':           MAX_LENGTH,
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

print("\n" + "=" * 80)
print("FINAL CROSS-VALIDATION SUMMARY")
print("=" * 80)
for k, v in cv_summary.items():
    print(f"  {k:<28}: {v}")

print("\n" + "=" * 80)
print("🎉 5-FOLD CROSS-VALIDATION COMPLETE!")
print("=" * 80)
print(f"\nOutputs saved to: {OUTPUT_DIR}/")
print(f"  fold_1/ … fold_{N_FOLDS}/      per-fold predictions, metrics, confusion matrix")
print(f"  best_model/                    best model weights (fold {best_fold_idx})")
print(f"  cv_fold_metrics.csv            per-fold metric table")
print(f"  cv_summary.csv                 overall CV summary")
print(f"  oof_predictions.csv            out-of-fold predictions (full dataset)")
print(f"  oof_confusion_matrix.csv       OOF confusion matrix")
if USE_WANDB and wandb_group_url:
    print(f"\n📊 W&B group: {wandb_group_url}")

# ==================== SEND COMPLETION EMAIL ====================
_s, _h, _p = _build_cv_success_email(cv_summary, wandb_group_url)
send_email_notification(_s, _h, _p)