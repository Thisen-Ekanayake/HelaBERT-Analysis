"""
BERT Fine-tuning for Sentiment Analysis — Comments Only (Baseline)

Pipeline:
  Stage 1 (this script): Fine-tune using ONLY the comment text → baseline model
  Stage 2 (next script): Fine-tune using comment + article body → context-aware model
  Then compare both models on the same held-out test set.

TSV columns used:
  comment_phrase    → input text
  comment_sentiment → label  (e.g. POSITIVE, NEGATIVE, NEUTRAL, ...)

Separate train/test TSV files are expected (no internal split needed,
but a small validation split is carved from train for early stopping).
"""

import os
import smtplib
import traceback
import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import sentencepiece as spm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertForMaskedLM,
    BertModel,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
import random
import wandb
from dotenv import load_dotenv



# ==================== LOAD ENVIRONMENT VARIABLES ====================
load_dotenv()  # Loads variables from .env file into os.environ


# ==================== EMAIL NOTIFICATION SETUP ====================

EMAIL_NOTIFICATIONS_ENABLED = os.getenv("EMAIL_NOTIFICATIONS_ENABLED", "true").lower() == "true"
EMAIL_SENDER      = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD    = os.getenv("EMAIL_APP_PASSWORD")
EMAIL_RECIPIENT   = os.getenv("EMAIL_RECIPIENT")
EMAIL_SMTP_HOST   = os.getenv("EMAIL_SMTP_HOST", "smtp.gmail.com")
EMAIL_SMTP_PORT   = int(os.getenv("EMAIL_SMTP_PORT", "587"))
TRAINING_JOB_NAME = os.getenv("TRAINING_sentiment_comments", "BERT Fine-tuning")


def send_email_notification(subject: str, body_html: str, body_text: str = None):
    """Send an email notification. Silently skips if email is not configured."""
    if not EMAIL_NOTIFICATIONS_ENABLED:
        print("ℹ️  Email notifications disabled (EMAIL_NOTIFICATIONS_ENABLED=false)")
        return
    if not all([EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENT]):
        print("⚠️  Email notification skipped — EMAIL_SENDER, EMAIL_APP_PASSWORD, "
              "or EMAIL_RECIPIENT not set in .env")
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
        print(f"✉️  Notification email sent to: {EMAIL_RECIPIENT}")
    except Exception as e:
        print(f"⚠️  Failed to send email notification: {e}")


def _build_success_email(summary: dict, wandb_url: str = None) -> tuple:
    timestamp    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wandb_section = (
        f'<p>📊 <a href="{wandb_url}">View full W&amp;B dashboard</a></p>'
        if wandb_url else ""
    )
    metric_rows = "".join(
        f"<tr><td style='padding:4px 12px'>{k}</td>"
        f"<td style='padding:4px 12px'><b>"
        f"{(f'{v:.4f}' if isinstance(v, float) else str(v))}"
        f"</b></td></tr>"
        for k, v in summary.items()
    )
    html = f"""
    <html><body style='font-family:Arial,sans-serif;color:#222'>
      <h2 style='color:#2e7d32'>✅ Training Completed Successfully</h2>
      <p><b>Job:</b> {TRAINING_JOB_NAME}</p>
      <p><b>Finished at:</b> {timestamp}</p>
      <h3>Final Metrics</h3>
      <table border='1' cellspacing='0' cellpadding='0'
             style='border-collapse:collapse;font-size:14px'>
        <tr style='background:#e8f5e9'>
          <th style='padding:6px 12px'>Metric</th>
          <th style='padding:6px 12px'>Value</th>
        </tr>
        {metric_rows}
      </table>
      {wandb_section}
      <p style='color:#888;font-size:12px'>Sent automatically by training script</p>
    </body></html>
    """
    plain = (
        f"Training Completed Successfully\n"
        f"Job: {TRAINING_JOB_NAME}\n"
        f"Finished at: {timestamp}\n\n"
        + "\n".join(
            f"{k}: {(f'{v:.4f}' if isinstance(v, float) else str(v))}"
            for k, v in summary.items()
        )
        + (f"\n\nW&B: {wandb_url}" if wandb_url else "")
    )
    subject = f"✅ Training Complete — {TRAINING_JOB_NAME}"
    return subject, html, plain


def _build_failure_email(error: Exception, tb_str: str) -> tuple:
    timestamp  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_type = type(error).__name__
    error_msg  = str(error)
    tb_escaped = tb_str.replace("<", "&lt;").replace(">", "&gt;")
    html = f"""
    <html><body style='font-family:Arial,sans-serif;color:#222'>
      <h2 style='color:#c62828'>❌ Training Crashed</h2>
      <p><b>Job:</b> {TRAINING_JOB_NAME}</p>
      <p><b>Crashed at:</b> {timestamp}</p>
      <p><b>Error:</b> <span style='color:#c62828'>{error_type}: {error_msg}</span></p>
      <h3>Traceback</h3>
      <pre style='background:#f5f5f5;padding:12px;font-size:12px;overflow:auto'>{tb_escaped}</pre>
      <p style='color:#888;font-size:12px'>Sent automatically by training script</p>
    </body></html>
    """
    plain = (
        f"Training Crashed\n"
        f"Job: {TRAINING_JOB_NAME}\n"
        f"Crashed at: {timestamp}\n\n"
        f"Error: {error_type}: {error_msg}\n\n"
        f"Traceback:\n{tb_str}"
    )
    subject = f"❌ Training Crashed — {TRAINING_JOB_NAME}"
    return subject, html, plain

# ==================== CONFIGURATION ====================
print("=" * 80)
print("BERT SENTIMENT ANALYSIS — STAGE 1: COMMENTS ONLY (BASELINE)")
print("=" * 80)

# ── Paths ──────────────────────────────────────────────────────────────────────
BERT_MODEL_PATH  = "HelaBERT"                               # ← pretrained BERT
TOKENIZER_MODEL  = "tokenizer/unigram_32000_0.9995.model"  # SentencePiece
BERT_CONFIG_FILE = "HelaBERT/config.json"

TRAIN_DATA_PATH  = "data/sinhala-sentiment-analysis/train.tsv"              # ← CHANGE: train TSV
TEST_DATA_PATH   = "data/sinhala-sentiment-analysis/test.tsv"               # ← CHANGE: test  TSV

# ── TSV column names (change if your file uses different headers) ──────────────
COMMENT_COL   = "comment_phrase"
LABEL_COL     = "comment_sentiment"

# ── Training hyperparameters ───────────────────────────────────────────────────
MAX_LENGTH                  = 256   # max tokens per comment
TRAIN_BATCH_SIZE            = 16
EVAL_BATCH_SIZE             = 64
LEARNING_RATE               = 5e-6
NUM_EPOCHS                  = 1
WARMUP_RATIO                = 0.1
WEIGHT_DECAY                = 0.05
GRADIENT_ACCUMULATION_STEPS = 1
VAL_SPLIT                   = 0.1   # fraction of train set used for validation

# ── Output ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR        = "HelaBERT_sentiment_comments_only"
STAGE_TAG         = "comments_only"   # used in W&B run name & saved files

# ── Misc ───────────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
USE_FP16     = True
NUM_WORKERS  = 2

# ── Weights & Biases ───────────────────────────────────────────────────────────
USE_WANDB         = True
WANDB_PROJECT     = "bert-sentiment-analysis"
WANDB_RUN_NAME    = f"bert_{STAGE_TAG}_lr{LEARNING_RATE}_bs{TRAIN_BATCH_SIZE}_ep{NUM_EPOCHS}"
WANDB_ENTITY      = None
WANDB_RUN_ID_FILE = f"wandb_run_id_{STAGE_TAG}.txt"

print("\n✓ Configuration loaded")
print(f"  - BERT model:       {BERT_MODEL_PATH}")
print(f"  - Tokenizer:        {TOKENIZER_MODEL}")
print(f"  - Train data:       {TRAIN_DATA_PATH}")
print(f"  - Test data:        {TEST_DATA_PATH}")
print(f"  - Output directory: {OUTPUT_DIR}")
print(f"  - Stage:            {STAGE_TAG}")
print(f"  - W&B logging:      {'Enabled' if USE_WANDB else 'Disabled'}")


# ==================== RANDOM SEEDS ====================
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
print("\n✓ Random seeds set")


# ==================== ENVIRONMENT CHECK ====================
print("\n" + "=" * 80)
print("ENVIRONMENT CHECK")
print("=" * 80)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device:     {torch.cuda.get_device_name(0)}")
    print(f"CUDA version:    {torch.version.cuda}")
else:
    print("⚠️  Running on CPU — training will be slower")


# ==================== VERIFY PATHS ====================
print("\n" + "=" * 80)
print("VERIFYING PATHS")
print("=" * 80)
assert os.path.exists(BERT_MODEL_PATH),  f"❌ Model not found: {BERT_MODEL_PATH}"
assert os.path.exists(TOKENIZER_MODEL),  f"❌ Tokenizer not found: {TOKENIZER_MODEL}"
assert os.path.exists(TRAIN_DATA_PATH),  f"❌ Train file not found: {TRAIN_DATA_PATH}"
assert os.path.exists(TEST_DATA_PATH),   f"❌ Test file not found: {TEST_DATA_PATH}"
print("✓ All paths verified")


# ==================== LOAD TOKENIZER ====================
print("\n" + "=" * 80)
print("LOADING SENTENCEPIECE TOKENIZER")
print("=" * 80)
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)
PAD_ID = sp.pad_id()
print(f"✓ Tokenizer loaded  |  vocab size: {sp.get_piece_size()}  |  PAD_ID: {PAD_ID}")


# ==================== HELPER: LOAD TSV ====================
def load_tsv(path: str, comment_col: str, label_col: str) -> pd.DataFrame:
    """Load a TSV, clean column names, extract comment + label columns."""
    try:
        df = pd.read_csv(path, sep='\t')
    except pd.errors.ParserError:
        print(f"  ⚠️  Parsing issues in {path}, retrying with python engine...")
        df = pd.read_csv(path, sep='\t', engine='python', on_bad_lines='skip')

    df.columns = df.columns.str.strip()

    # Flexible column detection — accept exact name or fuzzy match
    def find_col(df, preferred):
        if preferred in df.columns:
            return preferred
        matches = [c for c in df.columns if preferred.lower() in c.lower()]
        if matches:
            return matches[0]
        raise KeyError(f"Could not find column '{preferred}' in {list(df.columns)}")

    c_col = find_col(df, comment_col)
    l_col = find_col(df, label_col)

    df = df[[c_col, l_col]].rename(columns={c_col: 'comment', l_col: 'label'})
    df = df.dropna(subset=['comment', 'label'])
    df['comment'] = df['comment'].astype(str).str.strip()
    df['label']   = df['label'].astype(str).str.strip().str.upper()
    df = df[df['comment'].str.len() > 0]
    return df


# ==================== LOAD DATA ====================
print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

train_df = load_tsv(TRAIN_DATA_PATH, COMMENT_COL, LABEL_COL)
test_df  = load_tsv(TEST_DATA_PATH,  COMMENT_COL, LABEL_COL)

print(f"✓ Train TSV loaded  →  {len(train_df):,} rows")
print(f"✓ Test  TSV loaded  →  {len(test_df):,}  rows")


# ==================== ENCODE LABELS ====================
print("\n" + "=" * 80)
print("ENCODING LABELS")
print("=" * 80)

# Fit encoder on ALL labels (train + test) so IDs are consistent
all_labels = pd.concat([train_df['label'], test_df['label']]).unique()
le = LabelEncoder()
le.fit(sorted(all_labels))   # sorted for determinism

train_df['label_id'] = le.transform(train_df['label'])
test_df['label_id']  = le.transform(test_df['label'])

NUM_LABELS  = len(le.classes_)
id_to_label = {i: lbl for i, lbl in enumerate(le.classes_)}
label_to_id = {lbl: i for i, lbl in id_to_label.items()}

# Save mapping
os.makedirs(OUTPUT_DIR, exist_ok=True)
mapping_df = pd.DataFrame({'label_id': list(id_to_label.keys()),
                            'label_name': list(id_to_label.values())})
mapping_df.to_csv(f"{OUTPUT_DIR}/label_mapping.csv", index=False)

print(f"✓ {NUM_LABELS} unique sentiment labels detected:")
for idx, lbl in sorted(id_to_label.items()):
    tr  = (train_df['label_id'] == idx).sum()
    te  = (test_df['label_id']  == idx).sum()
    print(f"  [{idx}] {lbl:20s}  train: {tr:5d}  test: {te:5d}")
print(f"\n✓ Label mapping saved to {OUTPUT_DIR}/label_mapping.csv")


# ==================== TRAIN / VALIDATION SPLIT ====================
print("\n" + "=" * 80)
print("TRAIN / VALIDATION SPLIT")
print("=" * 80)

train_comments = train_df['comment'].tolist()
train_labels   = train_df['label_id'].tolist()

tr_texts, val_texts, tr_labels, val_labels = train_test_split(
    train_comments, train_labels,
    test_size=VAL_SPLIT,
    random_state=RANDOM_SEED,
    stratify=train_labels
)

print(f"  Train    : {len(tr_texts):,}")
print(f"  Val      : {len(val_texts):,}")
print(f"  Test     : {len(test_df):,}  (held-out, not used during training)")

test_comments = test_df['comment'].tolist()
test_labels   = test_df['label_id'].tolist()


# ==================== DATASET CLASS ====================
print("\n" + "=" * 80)
print("CREATING PYTORCH DATASETS")
print("=" * 80)


class CommentDataset(Dataset):
    """Tokenise a comment and return input_ids + attention_mask + label."""

    def __init__(self, texts, labels, sp_processor, max_length):
        self.texts      = texts
        self.labels     = labels
        self.sp         = sp_processor
        self.max_length = max_length
        self.pad_id     = sp_processor.pad_id()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids  = self.sp.encode(self.texts[idx])[:self.max_length]
        mask = [1] * len(ids)
        pad  = self.max_length - len(ids)
        ids  += [self.pad_id] * pad
        mask += [0] * pad

        return {
            'input_ids':      torch.tensor(ids,             dtype=torch.long),
            'attention_mask': torch.tensor(mask,            dtype=torch.long),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long)
        }


train_dataset = CommentDataset(tr_texts,       tr_labels,   sp, MAX_LENGTH)
val_dataset   = CommentDataset(val_texts,       val_labels,  sp, MAX_LENGTH)
test_dataset  = CommentDataset(test_comments,   test_labels, sp, MAX_LENGTH)

print(f"✓ train_dataset : {len(train_dataset):,} samples")
print(f"✓ val_dataset   : {len(val_dataset):,}   samples")
print(f"✓ test_dataset  : {len(test_dataset):,}  samples")

sample = train_dataset[0]
print(f"\nSample check:")
print(f"  input_ids shape     : {sample['input_ids'].shape}")
print(f"  attention_mask shape: {sample['attention_mask'].shape}")
print(f"  label               : {sample['labels'].item()} → {id_to_label[sample['labels'].item()]}")
print(f"  non-pad tokens      : {sample['attention_mask'].sum().item()}")


# ==================== LOAD MODEL ====================
print("\n" + "=" * 80)
print("LOADING MODEL")
print("=" * 80)

# Load config
if os.path.exists(BERT_CONFIG_FILE):
    config = BertConfig.from_json_file(BERT_CONFIG_FILE)
    print(f"✓ Config loaded from: {BERT_CONFIG_FILE}")
else:
    try:
        config = BertConfig.from_pretrained(BERT_MODEL_PATH)
        print("✓ Config loaded from model directory")
    except Exception:
        config = None
        print("⚠️  Config not found — will infer from model")

if config:
    config.num_labels = NUM_LABELS
    print(f"  hidden_size: {config.hidden_size}  |  layers: {config.num_hidden_layers}"
          f"  |  heads: {config.num_attention_heads}  |  num_labels: {config.num_labels}")

print(f"\nLoading weights from: {BERT_MODEL_PATH}")

def load_bert_for_classification(model_path, config):
    """Try BertModel → BertForMaskedLM → raise."""
    try:
        base   = BertModel.from_pretrained(model_path)
        model  = BertForSequenceClassification(config)
        base_s = base.state_dict()
        mod_s  = model.state_dict()
        for name, param in base_s.items():
            if name in mod_s:
                mod_s[name].copy_(param)
        model.load_state_dict(mod_s, strict=False)
        print("✓ Weights loaded via BertModel")
        return model
    except Exception as e:
        print(f"  BertModel load failed ({e}), trying MLM checkpoint...")

    mlm   = BertForMaskedLM.from_pretrained(model_path)
    if config is None:
        config = mlm.config
        config.num_labels = NUM_LABELS
    model = BertForSequenceClassification(config)
    mlm_s = mlm.state_dict()
    mod_s = model.state_dict()
    for name, param in mlm_s.items():
        if name.startswith('bert.') and name in mod_s:
            mod_s[name].copy_(param)
    model.load_state_dict(mod_s, strict=False)
    print("✓ Weights loaded via BertForMaskedLM")
    return model

model = load_bert_for_classification(BERT_MODEL_PATH, config)

total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params:     {total:,}")
print(f"  Trainable params: {trainable:,}  ({100*trainable/total:.1f}%)")


# ==================== METRICS ====================
print("\n" + "=" * 80)
print("SETTING UP METRICS")
print("=" * 80)


def compute_metrics(eval_pred: EvalPrediction):
    preds  = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    return {
        'accuracy':    accuracy_score(labels, preds),
        'precision':   precision_score(labels, preds, average='macro',    zero_division=0),
        'recall':      recall_score(labels, preds,    average='macro',    zero_division=0),
        'f1':          f1_score(labels, preds,        average='macro',    zero_division=0),
        'f1_weighted': f1_score(labels, preds,        average='weighted', zero_division=0),
    }


print("✓ Metrics: accuracy, precision, recall, macro-F1, weighted-F1")


# ==================== WEIGHTS & BIASES ====================
if USE_WANDB:
    print("\n" + "=" * 80)
    print("INITIALIZING W&B")
    print("=" * 80)

    wandb_run_id = None
    if os.path.exists(WANDB_RUN_ID_FILE):
        with open(WANDB_RUN_ID_FILE) as f:
            wandb_run_id = f.read().strip()
        print(f"  Resuming W&B run: {wandb_run_id}")

    if wandb.run is None:
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=WANDB_RUN_NAME,
            id=wandb_run_id,
            resume="allow",
            config={
                "stage":               STAGE_TAG,
                "model":               BERT_MODEL_PATH,
                "tokenizer":           "SentencePiece",
                "vocab_size":          sp.get_piece_size(),
                "hidden_size":         config.hidden_size         if config else "?",
                "num_layers":          config.num_hidden_layers   if config else "?",
                "num_attention_heads": config.num_attention_heads if config else "?",
                "num_labels":          NUM_LABELS,
                "label_names":         list(le.classes_),
                "learning_rate":       LEARNING_RATE,
                "epochs":              NUM_EPOCHS,
                "train_batch_size":    TRAIN_BATCH_SIZE,
                "effective_batch":     TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
                "warmup_ratio":        WARMUP_RATIO,
                "weight_decay":        WEIGHT_DECAY,
                "max_length":          MAX_LENGTH,
                "fp16":                USE_FP16 and torch.cuda.is_available(),
                "train_samples":       len(tr_texts),
                "val_samples":         len(val_texts),
                "test_samples":        len(test_comments),
            }
        )
        with open(WANDB_RUN_ID_FILE, 'w') as f:
            f.write(run.id)
        print(f"✓ W&B run ID: {run.id}")
        print(f"✓ Dashboard : {run.get_url()}")
        wandb.log({
            "train_label_dist": wandb.Histogram(tr_labels),
            "val_label_dist":   wandb.Histogram(val_labels),
        })


# ==================== TRAINING ARGS ====================
print("\n" + "=" * 80)
print("CONFIGURING TRAINER")
print("=" * 80)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

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
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=3,

    fp16=USE_FP16 and torch.cuda.is_available(),
    dataloader_num_workers=NUM_WORKERS,
    seed=RANDOM_SEED,

    report_to="wandb" if USE_WANDB else "none",
    run_name=WANDB_RUN_NAME if USE_WANDB else None,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("✓ Trainer ready")
print(f"  Approx optimiser steps: "
      f"~{len(train_dataset) * NUM_EPOCHS // (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)}")


# ==================== TRAINING ====================
print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)
print(f"Stage : {STAGE_TAG}")
print(f"Input : comment text only")
print(f"Labels: {', '.join(le.classes_)}")
if USE_WANDB:
    print(f"W&B   : {wandb.run.get_url()}")
print()

try:
    train_result = trainer.train()
    print("\n✓ Training complete")
    for k, v in train_result.metrics.items():
        print(f"  {k}: {v:.4f}")

except KeyboardInterrupt:
    print("\nInterrupted — saving model...")
    trainer.save_model(f"{OUTPUT_DIR}/interrupted_model")
    if USE_WANDB:
        wandb.finish(exit_code=1)
    _subj, _html, _plain = _build_failure_email(
        KeyboardInterrupt("Training was manually interrupted (KeyboardInterrupt)"),
        "Training was manually interrupted by the user."
    )
    send_email_notification(_subj, _html, _plain)
    raise

except Exception as e:
    tb_str = traceback.format_exc()
    print(f"\n❌ Training failed: {e}")
    if USE_WANDB:
        wandb.finish(exit_code=1)
    _subj, _html, _plain = _build_failure_email(e, tb_str)
    send_email_notification(_subj, _html, _plain)
    raise


# ==================== SAVE FINAL MODEL ====================
print("\n" + "=" * 80)
print("SAVING MODEL")
print("=" * 80)

final_model_path = f"{OUTPUT_DIR}/final_model"
trainer.save_model(final_model_path)
if config:
    config.save_pretrained(final_model_path)
mapping_df.to_csv(f"{final_model_path}/label_mapping.csv", index=False)
print(f"✓ Model saved to: {final_model_path}")



# ==================== EVALUATE ON VALIDATION SET ====================
print("\n" + "=" * 80)
print("VALIDATION SET EVALUATION")
print("=" * 80)

val_results = trainer.evaluate()
print("\nValidation metrics:")
for k, v in val_results.items():
    fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
    print(f"  {k:30s}: {fmt}")


# ==================== EVALUATE ON TEST SET ====================
print("\n" + "=" * 80)
print("TEST SET EVALUATION  (held-out)")
print("=" * 80)

test_output = trainer.predict(test_dataset)
y_pred      = np.argmax(test_output.predictions, axis=-1)
y_true      = np.array(test_labels)

test_metrics = {
    'accuracy':    accuracy_score(y_true, y_pred),
    'precision':   precision_score(y_true, y_pred, average='macro',    zero_division=0),
    'recall':      recall_score(y_true, y_pred,    average='macro',    zero_division=0),
    'f1':          f1_score(y_true, y_pred,        average='macro',    zero_division=0),
    'f1_weighted': f1_score(y_true, y_pred,        average='weighted', zero_division=0),
}

print("\nTest metrics:")
for k, v in test_metrics.items():
    print(f"  {k:20s}: {v:.4f}")

if USE_WANDB:
    wandb.log({f"test/{k}": v for k, v in test_metrics.items()})


# ==================== DETAILED REPORT ====================
print("\n" + "=" * 80)
print("CLASSIFICATION REPORT (test set)")
print("=" * 80)

target_names = [id_to_label[i] for i in range(NUM_LABELS)]
report_text  = classification_report(y_true, y_pred, target_names=target_names, digits=4)
print(report_text)

if USE_WANDB:
    wandb.log({"test/classification_report": wandb.Table(
        data=[[report_text]], columns=["report"]
    )})


# ==================== CONFUSION MATRIX ====================
print("\n" + "=" * 80)
print("CONFUSION MATRIX (test set)")
print("=" * 80)

cm    = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=[f"True_{id_to_label[i]}"  for i in range(NUM_LABELS)],
    columns=[f"Pred_{id_to_label[i]}" for i in range(NUM_LABELS)]
)
print(cm_df.to_string())
cm_df.to_csv(f"{OUTPUT_DIR}/confusion_matrix_test.csv")
print(f"\n✓ Saved to {OUTPUT_DIR}/confusion_matrix_test.csv")

if USE_WANDB:
    try:
        import plotly.figure_factory as ff
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=[f"Pred_{id_to_label[i]}"  for i in range(NUM_LABELS)],
            y=[f"True_{id_to_label[i]}"  for i in range(NUM_LABELS)],
            colorscale='Blues', showscale=True
        )
        fig.update_layout(title=f"Confusion Matrix — {STAGE_TAG}",
                          xaxis_title="Predicted", yaxis_title="True")
        wandb.log({"test/confusion_matrix": fig})
    except ImportError:
        print("⚠️  plotly not installed — skipping heatmap")


# ==================== PER-CLASS METRICS ====================
print("\n" + "=" * 80)
print("PER-CLASS METRICS (test set)")
print("=" * 80)

prec_pc, rec_pc, f1_pc, sup_pc = precision_recall_fscore_support(
    y_true, y_pred, average=None, zero_division=0
)

per_class_df = pd.DataFrame({
    'Label ID':   range(NUM_LABELS),
    'Sentiment':  target_names,
    'Precision':  prec_pc,
    'Recall':     rec_pc,
    'F1-Score':   f1_pc,
    'Support':    sup_pc
})
print(per_class_df.to_string(index=False))
per_class_df.to_csv(f"{OUTPUT_DIR}/per_class_metrics_test.csv", index=False)
print(f"\n✓ Saved to {OUTPUT_DIR}/per_class_metrics_test.csv")

if USE_WANDB:
    wandb.log({"test/per_class_metrics": wandb.Table(dataframe=per_class_df)})
    for metric in ['Precision', 'Recall', 'F1-Score']:
        data  = [[target_names[i], per_class_df.loc[i, metric]] for i in range(NUM_LABELS)]
        tbl   = wandb.Table(data=data, columns=["Sentiment", metric])
        wandb.log({f"test/per_class_{metric.lower().replace('-','_')}":
                   wandb.plot.bar(tbl, "Sentiment", metric, title=f"Per-Class {metric}")})


# ==================== SAMPLE PREDICTIONS ====================
print("\n" + "=" * 80)
print("SAMPLE PREDICTIONS (test set)")
print("=" * 80)

correct_mask      = (y_pred == y_true)
correct_indices   = np.where(correct_mask)[0]
incorrect_indices = np.where(~correct_mask)[0]

print(f"\n  Correct  : {correct_mask.sum()} / {len(y_true)} "
      f"({100*correct_mask.sum()/len(y_true):.2f}%)")
print(f"  Incorrect: {(~correct_mask).sum()} / {len(y_true)} "
      f"({100*(~correct_mask).sum()/len(y_true):.2f}%)")

sample_rows = []

print("\n── Correct (showing 5) ──────────────────────────────────────────────")
for i, idx in enumerate(correct_indices[:5]):
    probs      = torch.softmax(torch.tensor(test_output.predictions[idx]), dim=0)
    confidence = probs[y_pred[idx]].item()
    preview    = test_comments[idx][:100] + "..." if len(test_comments[idx]) > 100 else test_comments[idx]
    print(f"\n{i+1}. {preview}")
    print(f"   True: {id_to_label[y_true[idx]]:15s}  Pred: {id_to_label[y_pred[idx]]:15s}  Conf: {confidence:.4f}")
    if USE_WANDB and i < 10:
        sample_rows.append([test_comments[idx], id_to_label[y_true[idx]],
                            id_to_label[y_pred[idx]], confidence, "Correct"])

if len(incorrect_indices) > 0:
    print("\n── Incorrect (showing 5) ────────────────────────────────────────────")
    for i, idx in enumerate(incorrect_indices[:5]):
        probs       = torch.softmax(torch.tensor(test_output.predictions[idx]), dim=0)
        confidence  = probs[y_pred[idx]].item()
        true_conf   = probs[y_true[idx]].item()
        preview     = test_comments[idx][:100] + "..." if len(test_comments[idx]) > 100 else test_comments[idx]
        print(f"\n{i+1}. {preview}")
        print(f"   True: {id_to_label[y_true[idx]]:15s} [{true_conf:.4f}]  "
              f"Pred: {id_to_label[y_pred[idx]]:15s} [{confidence:.4f}]")
        if USE_WANDB and i < 10:
            sample_rows.append([test_comments[idx], id_to_label[y_true[idx]],
                                id_to_label[y_pred[idx]], confidence, "Incorrect"])

if USE_WANDB and sample_rows:
    wandb.log({"test/sample_predictions": wandb.Table(
        data=sample_rows,
        columns=["Comment", "True Sentiment", "Predicted Sentiment", "Confidence", "Result"]
    )})


# ==================== SAVE PREDICTIONS ====================
print("\n" + "=" * 80)
print("SAVING PREDICTIONS")
print("=" * 80)

confidences = [
    torch.softmax(torch.tensor(test_output.predictions[i]), dim=0)[y_pred[i]].item()
    for i in range(len(y_pred))
]

results_df = pd.DataFrame({
    'comment':            test_comments,
    'true_label_id':      y_true,
    'true_sentiment':     [id_to_label[l] for l in y_true],
    'predicted_label_id': y_pred,
    'predicted_sentiment':[id_to_label[l] for l in y_pred],
    'correct':            y_true == y_pred,
    'confidence':         confidences
})
results_df.to_csv(f"{OUTPUT_DIR}/predictions_test.csv", index=False)
print(f"✓ Predictions saved to {OUTPUT_DIR}/predictions_test.csv")


# ==================== SUMMARY ====================
summary = {
    'Stage':             STAGE_TAG,
    'Model':             'BERT with SentencePiece',
    'Input':             'comment_phrase only',
    'Num Classes':       NUM_LABELS,
    'Classes':           ', '.join(le.classes_),
    'Train Samples':     len(tr_texts),
    'Val Samples':       len(val_texts),
    'Test Samples':      len(test_comments),
    'Epochs':            NUM_EPOCHS,
    'Learning Rate':     LEARNING_RATE,
    'Batch Size':        TRAIN_BATCH_SIZE,
    'Max Length':        MAX_LENGTH,
    # val metrics
    'Val Accuracy':      val_results.get('eval_accuracy',    float('nan')),
    'Val F1 (macro)':    val_results.get('eval_f1',          float('nan')),
    'Val F1 (weighted)': val_results.get('eval_f1_weighted', float('nan')),
    # test metrics
    'Test Accuracy':     test_metrics['accuracy'],
    'Test Precision':    test_metrics['precision'],
    'Test Recall':       test_metrics['recall'],
    'Test F1 (macro)':   test_metrics['f1'],
    'Test F1 (weighted)':test_metrics['f1_weighted'],
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(f"{OUTPUT_DIR}/training_summary.csv", index=False)

if USE_WANDB:
    wandb.log({"summary": wandb.Table(dataframe=summary_df)})

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
for k, v in summary.items():
    fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
    print(f"  {k:25s}: {fmt}")

print("\n" + "=" * 80)
print("🎉 STAGE 1 (COMMENTS ONLY) COMPLETE!")
print("=" * 80)
print(f"\nOutputs in: {OUTPUT_DIR}/")
print(f"  final_model/                  — trained model + label_mapping.csv")
print(f"  label_mapping.csv             — id ↔ sentiment name")
print(f"  training_summary.csv          — all metrics (val + test)")
print(f"  per_class_metrics_test.csv    — per-class breakdown")
print(f"  predictions_test.csv          — all test predictions")
print(f"  confusion_matrix_test.csv     — confusion matrix")
print(f"\n→ Next: run sentiment_finetune_context.py (Stage 2) to include article body")
print(f"        and compare against these baseline results.")

if USE_WANDB:
    print(f"\n📊 Full results: {wandb.run.get_url()}")
    _wandb_url = wandb.run.get_url()
    wandb.finish()
    print("✓ W&B run finished")
else:
    _wandb_url = None

# ==================== SEND COMPLETION EMAIL ====================
print("\n" + "=" * 80)
print("SENDING COMPLETION NOTIFICATION")
print("=" * 80)
_subj, _html, _plain = _build_success_email(summary, _wandb_url)
send_email_notification(_subj, _html, _plain)

print("\n" + "=" * 80)