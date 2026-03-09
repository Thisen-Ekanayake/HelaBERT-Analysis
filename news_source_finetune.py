"""
BERT Fine-tuning for News Source Classification with W&B Logging
— Balanced training via oversampling + weighted loss (Option 3) —

This script fine-tunes a pre-trained BERT model (trained with SentencePiece tokenizer)
on a news source classification task with Weights & Biases logging.
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
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
TRAINING_JOB_NAME = os.getenv("TRAINING_news_source", "BERT News Source Fine-tuning (Balanced)")


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
    timestamp     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
      <p style='color:#888;font-size:12px'>Sent by news_source_finetune_balanced.py</p>
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
      <p style='color:#888;font-size:12px'>Sent by news_source_finetune_balanced.py</p>
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
print("BERT FINE-TUNING FOR NEWS SOURCE CLASSIFICATION  [BALANCED — OPTION 3]")
print("=" * 80)

# Model and Tokenizer Paths
BERT_MODEL_PATH  = "HelaBERT"
TOKENIZER_MODEL  = "tokenizer/unigram_32000_0.9995.model"
BERT_CONFIG_FILE = "HelaBERT/config.json"

# Dataset Path
DATA_PATH = "data/Sinhala-News-Source-classification/sinhala-news-sources.csv"

# Training Parameters
NUM_LABELS                   = 9      # auto-detected from data below
MAX_LENGTH                   = 64     # source classification needs less context than category
TRAIN_BATCH_SIZE             = 16     # reduced from 64; cleaner gradients with oversampling
EVAL_BATCH_SIZE              = 16
LEARNING_RATE                = 6e-5   # increased from 2e-5
NUM_EPOCHS                   = 5     # early stopping decides actual stop point
WARMUP_RATIO                 = 0.06   # reduced from 0.1
WEIGHT_DECAY                 = 0.01   # reduced from 0.05
GRADIENT_ACCUMULATION_STEPS  = 2      # effective batch = 32
EARLY_STOPPING_PATIENCE      = 3

# Balancing
OVERSAMPLE_TRAIN  = True
USE_CLASS_WEIGHTS = True

# Output Directory
OUTPUT_DIR = "HelaBERT_finetuned_news_source_balanced"

# Random Seed
RANDOM_SEED = 42

# Test Split Ratio
TEST_SIZE = 0.2

# Hardware Settings
USE_FP16    = True
NUM_WORKERS = 2

# Weights & Biases Configuration
USE_WANDB         = True
WANDB_PROJECT     = "bert-news-source-finetuning"
WANDB_RUN_NAME    = f"bert_balanced_lr{LEARNING_RATE}_bs{TRAIN_BATCH_SIZE}_ep{NUM_EPOCHS}_source"
WANDB_ENTITY      = None
WANDB_RUN_ID_FILE = "wandb_run_id_source_balanced.txt"

print("\n✓ Configuration loaded")
print(f"  - Model path:       {BERT_MODEL_PATH}")
print(f"  - Tokenizer:        {TOKENIZER_MODEL}")
print(f"  - Dataset:          {DATA_PATH}")
print(f"  - Output directory: {OUTPUT_DIR}")
print(f"  - Oversampling:     {'Enabled' if OVERSAMPLE_TRAIN else 'Disabled'}")
print(f"  - Class weights:    {'Enabled' if USE_CLASS_WEIGHTS else 'Disabled'}")
print(f"  - W&B logging:      {'Enabled' if USE_WANDB else 'Disabled'}")


# ==================== SET RANDOM SEEDS ====================
random.seed(RANDOM_SEED)
stdlib_random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print("\n✓ Random seeds set for reproducibility")


# ==================== CHECK ENVIRONMENT ====================
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

assert os.path.exists(BERT_MODEL_PATH),  f"❌ Model path not found: {BERT_MODEL_PATH}"
assert os.path.exists(TOKENIZER_MODEL),  f"❌ Tokenizer not found: {TOKENIZER_MODEL}"
assert os.path.exists(DATA_PATH),        f"❌ Data file not found: {DATA_PATH}"

print("✓ All required paths verified")


# ==================== LOAD TOKENIZER ====================
print("\n" + "=" * 80)
print("LOADING SENTENCEPIECE TOKENIZER")
print("=" * 80)

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)

PAD_ID = sp.pad_id()
UNK_ID = sp.unk_id()
BOS_ID = sp.bos_id()
EOS_ID = sp.eos_id()

print("✓ SentencePiece tokenizer loaded")
print(f"  - Vocab size: {sp.get_piece_size()}")
print(f"  - PAD_ID:     {PAD_ID}")
print(f"  - UNK_ID:     {UNK_ID}")
print(f"  - BOS_ID:     {BOS_ID}")
print(f"  - EOS_ID:     {EOS_ID}")

test_text   = "ත්‍රිකුණාමලයේ දී නැගෙනහිර ආරක්ෂක සේනා මූලස්ථානය"
test_tokens = sp.encode(test_text)
print(f"\nTest tokenization:")
print(f"  - Input:  {test_text}")
print(f"  - Tokens: {test_tokens[:10]}... (showing first 10)")
print(f"  - Length: {len(test_tokens)}")


# ==================== LOAD DATASET ====================
print("\n" + "=" * 80)
print("LOADING DATASET")
print("=" * 80)

try:
    df = pd.read_csv(DATA_PATH)
except pd.errors.ParserError:
    print("⚠️  CSV has formatting issues, using error-tolerant parsing...")
    df = pd.read_csv(DATA_PATH, engine='python', on_bad_lines='skip')
    print("✓ Loaded with some lines skipped")

df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)

print(f"\nCleaned columns: {df.columns.tolist()}")

possible_comment_cols = [col for col in df.columns if 'comment' in col.lower()]
possible_label_cols   = [col for col in df.columns if 'label'   in col.lower()]

if possible_comment_cols and possible_label_cols:
    comment_col = possible_comment_cols[0]
    label_col   = possible_label_cols[0]
    print(f"✓ Identified comment column: '{comment_col}'")
    print(f"✓ Identified label column:   '{label_col}'")
    df = df.rename(columns={comment_col: 'comment', label_col: 'label'})
else:
    print("⚠️  Could not automatically identify columns — assuming col1=comment, col2=label")
    df = df.iloc[:, -2:]
    df.columns = ['comment', 'label']

df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore')
df = df.dropna()
df['label']   = df['label'].astype(int)
df['comment'] = df['comment'].astype(str).str.strip()

# Second pass to catch any nulls introduced by coercion
df = df.dropna()

print("✓ Dataset loaded")
print(f"  - Total samples: {len(df)}")
print(f"  - Columns:       {df.columns.tolist()}")
print(f"  - Shape:         {df.shape}")

missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print(f"\n⚠️  Missing values found — dropping affected rows")
    df = df.dropna()
    print(f"  - Remaining samples: {len(df)}")
else:
    print("✓ No missing values")

print("\n" + "-" * 80)
print("LABEL DISTRIBUTION")
print("-" * 80)
label_counts = df['label'].value_counts().sort_index()
for lbl, count in label_counts.items():
    pct = 100 * count / len(df)
    print(f"  Label {lbl:2d}: {count:6d} samples ({pct:.1f}%)")

actual_num_labels = df['label'].nunique()
print(f"\nUnique labels found: {actual_num_labels}")
print(f"Label range: {df['label'].min()} to {df['label'].max()}")

if actual_num_labels != NUM_LABELS:
    print(f"⚠️  Updating NUM_LABELS from {NUM_LABELS} to {actual_num_labels}")
    NUM_LABELS = actual_num_labels

print("\n" + "-" * 80)
print("SAMPLE DATA")
print("-" * 80)
print(df.head())


# ==================== PREPARE DATA ====================
print("\n" + "=" * 80)
print("PREPARING DATA")
print("=" * 80)

comments = df['comment'].tolist()
labels   = df['label'].tolist()

print(f"✓ Extracted {len(comments)} comments and {len(labels)} labels")

print("\nSample examples:")
for i in range(min(3, len(comments))):
    preview = comments[i][:80] + "..." if len(comments[i]) > 80 else comments[i]
    print(f"  {i+1}. [{labels[i]}] {preview}")

print(f"\nSplitting data (train: {1-TEST_SIZE:.0%}, val: {TEST_SIZE:.0%})...")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    comments, labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=labels
)

print(f"✓ Data split completed")
print(f"  - Training samples (before balancing): {len(train_texts)}")
print(f"  - Validation samples:                  {len(val_texts)}")

print("\nTrain label distribution (before balancing):")
print(pd.Series(train_labels).value_counts().sort_index().to_string())
print("\nValidation label distribution:")
print(pd.Series(val_labels).value_counts().sort_index().to_string())


# ==================== OVERSAMPLING ====================
print("\n" + "=" * 80)
print("BALANCING TRAINING DATA (OVERSAMPLING)")
print("=" * 80)


def oversample(texts, labels, seed=42):
    """
    Oversample minority classes so every class matches the majority class count.
    Returns shuffled balanced lists.
    """
    stdlib_random.seed(seed)
    counts    = Counter(labels)
    max_count = max(counts.values())

    balanced_texts  = list(texts)
    balanced_labels = list(labels)

    for label, count in counts.items():
        needed  = max_count - count
        if needed == 0:
            continue
        indices = [i for i, l in enumerate(labels) if l == label]
        extras  = stdlib_random.choices(indices, k=needed)
        balanced_texts  += [texts[i]  for i in extras]
        balanced_labels += [labels[i] for i in extras]

    combined = list(zip(balanced_texts, balanced_labels))
    stdlib_random.shuffle(combined)
    balanced_texts, balanced_labels = zip(*combined)
    return list(balanced_texts), list(balanced_labels)


if OVERSAMPLE_TRAIN:
    original_train_size = len(train_texts)
    train_texts, train_labels = oversample(train_texts, train_labels, seed=RANDOM_SEED)

    print(f"✓ Oversampling complete")
    print(f"  - Before: {original_train_size} samples")
    print(f"  - After:  {len(train_texts)} samples")
    print(f"\nTrain label distribution (after oversampling):")
    print(pd.Series(train_labels).value_counts().sort_index().to_string())
else:
    print("⚠️  Oversampling disabled — using original distribution")


# ==================== CLASS WEIGHTS ====================
print("\n" + "=" * 80)
print("COMPUTING CLASS WEIGHTS")
print("=" * 80)

# Compute from original (pre-oversample) distribution so minority classes
# still receive a signal boost even after balancing
original_counts     = Counter(labels)
class_weights_list  = [1.0 / original_counts.get(i, 1) for i in range(NUM_LABELS)]
weights_tensor      = torch.tensor(class_weights_list, dtype=torch.float)
weights_tensor      = weights_tensor / weights_tensor.sum() * NUM_LABELS

print("Class weights (based on original distribution):")
for i, w in enumerate(weights_tensor):
    print(f"  Label {i:2d}: {w.item():.4f}")


# ==================== DEFINE DATASET CLASS ====================
print("\n" + "=" * 80)
print("CREATING DATASET")
print("=" * 80)


class SentencePieceDataset(Dataset):
    """Custom Dataset for BERT fine-tuning with SentencePiece tokenizer."""

    def __init__(self, texts, labels, sp_processor, max_length=64):
        self.texts      = texts
        self.labels     = labels
        self.sp         = sp_processor
        self.max_length = max_length
        self.pad_id     = sp_processor.pad_id()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        token_ids = self.sp.encode(self.texts[idx])

        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        attention_mask = [1] * len(token_ids)
        padding_length = self.max_length - len(token_ids)
        token_ids      = token_ids      + [self.pad_id] * padding_length
        attention_mask = attention_mask + [0]           * padding_length

        return {
            'input_ids':      torch.tensor(token_ids,           dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask,      dtype=torch.long),
            'labels':         torch.tensor(self.labels[idx],    dtype=torch.long),
        }


train_dataset = SentencePieceDataset(train_texts, train_labels, sp, max_length=MAX_LENGTH)
val_dataset   = SentencePieceDataset(val_texts,   val_labels,   sp, max_length=MAX_LENGTH)

print(f"✓ Datasets created")
print(f"  - Train dataset size:      {len(train_dataset)}")
print(f"  - Validation dataset size: {len(val_dataset)}")

sample = train_dataset[0]
print(f"\nSample from dataset:")
print(f"  - input_ids shape:      {sample['input_ids'].shape}")
print(f"  - attention_mask shape: {sample['attention_mask'].shape}")
print(f"  - label:                {sample['labels'].item()}")
print(f"  - Non-padding tokens:   {sample['attention_mask'].sum().item()}")


# ==================== LOAD MODEL ====================
print("\n" + "=" * 80)
print("LOADING MODEL")
print("=" * 80)

if os.path.exists(BERT_CONFIG_FILE):
    config = BertConfig.from_json_file(BERT_CONFIG_FILE)
    print(f"✓ Loaded BERT config from: {BERT_CONFIG_FILE}")
else:
    try:
        config = BertConfig.from_pretrained(BERT_MODEL_PATH)
        print("✓ Loaded BERT config from model directory")
    except Exception:
        print("⚠️  Could not load config, will try to infer from model")
        config = None

if config:
    config.num_labels          = NUM_LABELS
    config.hidden_dropout_prob = 0.2   # increased from default 0.1 to counter duplicate memorisation
    print(f"\nBERT Configuration:")
    print(f"  - Hidden size:             {config.hidden_size}")
    print(f"  - Num layers:              {config.num_hidden_layers}")
    print(f"  - Num attention heads:     {config.num_attention_heads}")
    print(f"  - Vocab size:              {config.vocab_size}")
    print(f"  - Max position embeddings: {config.max_position_embeddings}")
    print(f"  - Num labels:              {config.num_labels}")
    print(f"  - Hidden dropout:          {config.hidden_dropout_prob}")

print(f"\nLoading model from: {BERT_MODEL_PATH}")

try:
    base_model = BertModel.from_pretrained(BERT_MODEL_PATH)
    print("✓ Loaded base BERT model from pretrained weights")

    if config is None:
        config                     = BertConfig.from_pretrained(BERT_MODEL_PATH)
        config.num_labels          = NUM_LABELS
        config.hidden_dropout_prob = 0.2

    model       = BertForSequenceClassification(config)
    base_state  = base_model.state_dict()
    model_state = model.state_dict()

    for name, param in base_state.items():
        if name in model_state:
            model_state[name].copy_(param)

    model.load_state_dict(model_state, strict=False)
    print("✓ Model loaded with classification head")

except Exception as e:
    print(f"⚠️  Initial load attempt failed: {e}")
    print("   Attempting to load MLM model and transfer weights...")

    try:
        mlm_model = BertForMaskedLM.from_pretrained(BERT_MODEL_PATH)

        if config is None:
            config                     = mlm_model.config
            config.num_labels          = NUM_LABELS
            config.hidden_dropout_prob = 0.2

        model       = BertForSequenceClassification(config)
        mlm_state   = mlm_model.state_dict()
        model_state = model.state_dict()

        for name, param in mlm_state.items():
            if name.startswith('bert.') and name in model_state:
                model_state[name].copy_(param)

        model.load_state_dict(model_state, strict=False)
        print("✓ Model weights loaded from MLM checkpoint with classification head initialized")

    except Exception as e2:
        print(f"❌ Failed to load model: {e2}")
        raise

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params     = sum(p.numel() for p in model.parameters())
print(f"\nModel statistics:")
print(f"  - Total parameters:      {total_params:,}")
print(f"  - Trainable parameters:  {trainable_params:,}")
print(f"  - Percentage trainable:  {100 * trainable_params / total_params:.2f}%")


# ==================== DEFINE METRICS ====================
print("\n" + "=" * 80)
print("SETTING UP METRICS")
print("=" * 80)


def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    accuracy    = accuracy_score(labels, preds)
    precision   = precision_score(labels, preds, average='macro',    zero_division=0)
    recall      = recall_score(labels,    preds, average='macro',    zero_division=0)
    f1          = f1_score(labels,        preds, average='macro',    zero_division=0)
    f1_weighted = f1_score(labels,        preds, average='weighted', zero_division=0)

    return {
        'accuracy':    accuracy,
        'precision':   precision,
        'recall':      recall,
        'f1':          f1,
        'f1_weighted': f1_weighted,
    }


print("✓ Metrics function defined (accuracy, precision, recall, F1, weighted F1)")


# ==================== WEIGHTED TRAINER ====================
print("\n" + "=" * 80)
print("SETTING UP WEIGHTED TRAINER")
print("=" * 80)


class WeightedTrainer(Trainer):
    """
    Subclass of HuggingFace Trainer that replaces the default CrossEntropyLoss
    with a class-weighted version. Acts as a safety net on top of oversampling.
    """

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.get("logits")
        loss    = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )(logits, labels)
        return (loss, outputs) if return_outputs else loss


if USE_CLASS_WEIGHTS:
    print("✓ WeightedTrainer will be used with class weights:")
    for i, w in enumerate(weights_tensor):
        print(f"  Label {i:2d}: {w.item():.4f}")
else:
    print("⚠️  Class weights disabled — falling back to standard Trainer")


# ==================== INITIALIZE WEIGHTS & BIASES ====================
if USE_WANDB:
    print("\n" + "=" * 80)
    print("INITIALIZING WEIGHTS & BIASES")
    print("=" * 80)

    wandb_run_id = None
    if os.path.exists(WANDB_RUN_ID_FILE):
        with open(WANDB_RUN_ID_FILE, 'r') as f:
            wandb_run_id = f.read().strip()
        print(f"✓ Found existing W&B run ID: {wandb_run_id}")
    else:
        print("✓ Starting new W&B run")

    if wandb.run is None:
        wandb_config = {
            # Model
            "model_architecture":   "BERT",
            "tokenizer":            "SentencePiece",
            "vocab_size":           sp.get_piece_size(),
            "hidden_size":          config.hidden_size         if config else "unknown",
            "num_layers":           config.num_hidden_layers   if config else "unknown",
            "num_attention_heads":  config.num_attention_heads if config else "unknown",
            "hidden_dropout_prob":  config.hidden_dropout_prob if config else 0.2,
            # Training
            "task":                    "news_source_classification",
            "balancing_strategy":      "oversample+class_weights",
            "num_labels":              NUM_LABELS,
            "learning_rate":           LEARNING_RATE,
            "epochs":                  NUM_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "train_batch_size":        TRAIN_BATCH_SIZE,
            "eval_batch_size":         EVAL_BATCH_SIZE,
            "gradient_accumulation":   GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size":    TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "warmup_ratio":            WARMUP_RATIO,
            "weight_decay":            WEIGHT_DECAY,
            "max_length":              MAX_LENGTH,
            "fp16":                    USE_FP16 and torch.cuda.is_available(),
            # Data
            "original_train_samples": len(comments) - len(val_texts),
            "balanced_train_samples": len(train_texts),
            "val_samples":            len(val_texts),
            "test_size":              TEST_SIZE,
            "random_seed":            RANDOM_SEED,
            # Class weights
            **{f"class_weight_{i}": weights_tensor[i].item() for i in range(NUM_LABELS)},
        }

        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=WANDB_RUN_NAME,
            id=wandb_run_id,
            resume="allow",
            config=wandb_config
        )

        with open(WANDB_RUN_ID_FILE, 'w') as f:
            f.write(run.id)
        print(f"✓ W&B run ID saved: {run.id}")
        print(f"✓ W&B dashboard:    {run.get_url()}")

        wandb.log({
            "train_label_distribution_balanced": wandb.Histogram(train_labels),
            "val_label_distribution":            wandb.Histogram(val_labels),
        })
    else:
        print("✓ W&B already initialized")


# ==================== SETUP TRAINING ====================
print("\n" + "=" * 80)
print("CONFIGURING TRAINING")
print("=" * 80)

os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Hyperparameters
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,

    # Evaluation and checkpointing
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    logging_first_step=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",   # fixed: must match the logged key
    greater_is_better=True,
    save_total_limit=3,

    # Hardware
    fp16=USE_FP16 and torch.cuda.is_available(),
    dataloader_num_workers=NUM_WORKERS,

    # Reproducibility
    seed=RANDOM_SEED,

    # W&B
    report_to="wandb" if USE_WANDB else "none",
    run_name=WANDB_RUN_NAME if USE_WANDB else None,

    push_to_hub=False,
)

print("✓ Training arguments configured")
print(f"\nTraining configuration:")
print(f"  - Epochs (max):         {NUM_EPOCHS}  (early stopping patience={EARLY_STOPPING_PATIENCE})")
print(f"  - Train batch size:     {TRAIN_BATCH_SIZE}")
print(f"  - Eval batch size:      {EVAL_BATCH_SIZE}")
print(f"  - Grad accumulation:    {GRADIENT_ACCUMULATION_STEPS}")
print(f"  - Effective batch size: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"  - Learning rate:        {LEARNING_RATE}")
print(f"  - Warmup ratio:         {WARMUP_RATIO}")
print(f"  - Weight decay:         {WEIGHT_DECAY}")
print(f"  - Max sequence length:  {MAX_LENGTH}")
print(f"  - FP16:                 {training_args.fp16}")
print(f"  - Approx steps/epoch:   ~{len(train_dataset) // (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)}")


# ==================== INITIALIZE TRAINER ====================
print("\n" + "=" * 80)
print("INITIALIZING TRAINER")
print("=" * 80)

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=EARLY_STOPPING_PATIENCE
)

if USE_CLASS_WEIGHTS:
    trainer = WeightedTrainer(
        class_weights=weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )
    print("✓ WeightedTrainer initialized (oversampling + class weights active)")
else:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )
    print("✓ Standard Trainer initialized (oversampling only)")


# ==================== TRAIN MODEL ====================
print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)
print(f"Training for up to {NUM_EPOCHS} epochs on {NUM_LABELS}-class source classification...")
print(f"  - Early stopping will halt if eval F1 doesn't improve for {EARLY_STOPPING_PATIENCE} epochs")
if USE_WANDB:
    print(f"Monitor progress at: {wandb.run.get_url()}")
print()

try:
    train_result = trainer.train()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED!")
    print("=" * 80)
    print("\nTraining Metrics:")
    for key, value in train_result.metrics.items():
        print(f"  - {key}: {value:.4f}")

except KeyboardInterrupt:
    print("\nTraining interrupted. Saving model...")
    trainer.save_model(f"{OUTPUT_DIR}/interrupted_model")
    print(f"✓ Model saved to {OUTPUT_DIR}/interrupted_model")
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
print(f"✓ Final model saved to: {final_model_path}")


# ==================== EVALUATE MODEL ====================
print("\n" + "=" * 80)
print("EVALUATING ON VALIDATION SET")
print("=" * 80)

eval_results = trainer.evaluate()

print("\nValidation Metrics:")
for key, value in eval_results.items():
    if isinstance(value, float):
        print(f"  - {key}: {value:.4f}")
    else:
        print(f"  - {key}: {value}")


# ==================== GENERATE PREDICTIONS ====================
print("\n" + "=" * 80)
print("GENERATING PREDICTIONS")
print("=" * 80)

predictions_output = trainer.predict(val_dataset)
y_pred = np.argmax(predictions_output.predictions, axis=-1)
y_true = np.array(val_labels)

print(f"✓ Generated {len(y_pred)} predictions")


# ==================== DETAILED EVALUATION ====================
print("\n" + "=" * 80)
print("DETAILED CLASSIFICATION REPORT")
print("=" * 80)

report_text = classification_report(y_true, y_pred, digits=4)
print(report_text)

if USE_WANDB:
    wandb.log({"classification_report": wandb.Table(
        data=[[report_text]],
        columns=["report"]
    )})


# ==================== CONFUSION MATRIX ====================
print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

cm = confusion_matrix(y_true, y_pred)
print(cm)

cm_df = pd.DataFrame(
    cm,
    index=[f"True_{i}"  for i in range(NUM_LABELS)],
    columns=[f"Pred_{i}" for i in range(NUM_LABELS)]
)
cm_df.to_csv(f"{OUTPUT_DIR}/confusion_matrix.csv")
print(f"\n✓ Confusion matrix saved to {OUTPUT_DIR}/confusion_matrix.csv")

if USE_WANDB:
    try:
        import plotly.figure_factory as ff
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=[f"Pred_{i}" for i in range(NUM_LABELS)],
            y=[f"True_{i}" for i in range(NUM_LABELS)],
            colorscale='Blues',
            showscale=True
        )
        fig.update_layout(
            title="Confusion Matrix — News Source (Balanced)",
            xaxis_title="Predicted Label",
            yaxis_title="True Label"
        )
        wandb.log({"confusion_matrix": fig})
        print("✓ Confusion matrix logged to W&B")
    except ImportError:
        print("⚠️  plotly not installed — skipping confusion matrix heatmap")


# ==================== PER-CLASS METRICS ====================
print("\n" + "=" * 80)
print("PER-CLASS PERFORMANCE")
print("=" * 80)

precision_per_class, recall_per_class, f1_per_class, support_per_class = \
    precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

per_class_report = pd.DataFrame({
    'Class':     range(NUM_LABELS),
    'Precision': precision_per_class,
    'Recall':    recall_per_class,
    'F1-Score':  f1_per_class,
    'Support':   support_per_class
})

print(per_class_report.to_string(index=False))

per_class_report.to_csv(f'{OUTPUT_DIR}/per_class_metrics.csv', index=False)
print(f"\n✓ Per-class metrics saved to {OUTPUT_DIR}/per_class_metrics.csv")

if USE_WANDB:
    wandb.log({"per_class_metrics": wandb.Table(dataframe=per_class_report)})
    for metric in ['Precision', 'Recall', 'F1-Score']:
        data  = [[i, per_class_report.loc[i, metric]] for i in range(NUM_LABELS)]
        table = wandb.Table(data=data, columns=["Class", metric])
        wandb.log({f"per_class_{metric.lower().replace('-', '_')}":
                   wandb.plot.bar(table, "Class", metric, title=f"Per-Class {metric}")})


# ==================== SAMPLE PREDICTIONS ====================
print("\n" + "=" * 80)
print("SAMPLE PREDICTIONS")
print("=" * 80)

correct_mask      = (y_pred == y_true)
correct_indices   = np.where(correct_mask)[0]
incorrect_indices = np.where(~correct_mask)[0]

print(f"\nCorrect predictions:   {correct_mask.sum()} / {len(y_true)} "
      f"({100*correct_mask.sum()/len(y_true):.2f}%)")
print(f"Incorrect predictions: {(~correct_mask).sum()} / {len(y_true)} "
      f"({100*(~correct_mask).sum()/len(y_true):.2f}%)")

sample_predictions = []

print("\n" + "-" * 80)
print("CORRECT PREDICTIONS (showing 5)")
print("-" * 80)
for i, idx in enumerate(correct_indices[:5]):
    text         = val_texts[idx]
    text_preview = text[:100] + "..." if len(text) > 100 else text
    true_label   = y_true[idx]
    pred_label   = y_pred[idx]
    probs        = torch.softmax(torch.tensor(predictions_output.predictions[idx]), dim=0)
    confidence   = probs[pred_label].item()

    print(f"\n{i+1}. Text: {text_preview}")
    print(f"   True: {true_label} | Predicted: {pred_label} | Confidence: {confidence:.4f}")

    if USE_WANDB and i < 10:
        sample_predictions.append([text, true_label, pred_label, confidence, "Correct"])

if len(incorrect_indices) > 0:
    print("\n" + "-" * 80)
    print("INCORRECT PREDICTIONS (showing 5)")
    print("-" * 80)
    for i, idx in enumerate(incorrect_indices[:5]):
        text            = val_texts[idx]
        text_preview    = text[:100] + "..." if len(text) > 100 else text
        true_label      = y_true[idx]
        pred_label      = y_pred[idx]
        probs           = torch.softmax(torch.tensor(predictions_output.predictions[idx]), dim=0)
        confidence      = probs[pred_label].item()
        true_confidence = probs[true_label].item()

        print(f"\n{i+1}. Text: {text_preview}")
        print(f"   True: {true_label} (conf: {true_confidence:.4f}) | "
              f"Predicted: {pred_label} (conf: {confidence:.4f})")

        if USE_WANDB and i < 10:
            sample_predictions.append([text, true_label, pred_label, confidence, "Incorrect"])
else:
    print("\n🎉 No incorrect predictions! Perfect accuracy!")

if USE_WANDB and sample_predictions:
    wandb.log({"sample_predictions": wandb.Table(
        data=sample_predictions,
        columns=["Text", "True Label", "Predicted Label", "Confidence", "Result"]
    )})


# ==================== SAVE ALL RESULTS ====================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

confidences = []
for i in range(len(predictions_output.predictions)):
    probs      = torch.softmax(torch.tensor(predictions_output.predictions[i]), dim=0)
    pred_label = y_pred[i]
    confidences.append(probs[pred_label].item())

results_df = pd.DataFrame({
    'text':            val_texts,
    'true_label':      y_true,
    'predicted_label': y_pred,
    'correct':         y_true == y_pred,
    'confidence':      confidences,
})
results_df.to_csv(f'{OUTPUT_DIR}/predictions.csv', index=False)
print(f"✓ All predictions saved to {OUTPUT_DIR}/predictions.csv")

summary = {
    'Model':                  'BERT with SentencePiece',
    'Task':                   'News Source Classification',
    'Balancing':              'Oversample + Class Weights',
    'Num Classes':            NUM_LABELS,
    'Original Train Samples': len(comments) - len(val_texts),
    'Balanced Train Samples': len(train_texts),
    'Val Samples':            len(val_texts),
    'Epochs':                 NUM_EPOCHS,
    'Learning Rate':          LEARNING_RATE,
    'Batch Size':             TRAIN_BATCH_SIZE,
    'Effective Batch Size':   TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
    'Max Length':             MAX_LENGTH,
    'Accuracy':               eval_results['eval_accuracy'],
    'Precision (macro)':      eval_results['eval_precision'],
    'Recall (macro)':         eval_results['eval_recall'],
    'F1 (macro)':             eval_results['eval_f1'],
    'F1 (weighted)':          eval_results['eval_f1_weighted'],
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(f'{OUTPUT_DIR}/training_summary.csv', index=False)
print(f"✓ Training summary saved to {OUTPUT_DIR}/training_summary.csv")

if USE_WANDB:
    wandb.log({
        "final/accuracy":    eval_results['eval_accuracy'],
        "final/precision":   eval_results['eval_precision'],
        "final/recall":      eval_results['eval_recall'],
        "final/f1":          eval_results['eval_f1'],
        "final/f1_weighted": eval_results['eval_f1_weighted'],
    })
    wandb.log({"training_summary": wandb.Table(dataframe=summary_df)})


# ==================== FINAL SUMMARY ====================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

for key, value in summary.items():
    if isinstance(value, float):
        print(f"{key:28s}: {value:.4f}")
    else:
        print(f"{key:28s}: {value}")

print("\n" + "=" * 80)
print("🎉 FINE-TUNING COMPLETE!")
print("=" * 80)
print(f"\nAll results saved to: {OUTPUT_DIR}/")
print(f"\nFiles created:")
print(f"  - final_model/            (trained model)")
print(f"  - training_summary.csv    (overall metrics)")
print(f"  - per_class_metrics.csv   (per-class performance)")
print(f"  - predictions.csv         (all validation predictions)")
print(f"  - confusion_matrix.csv    (confusion matrix)")
print(f"\nBest model metrics:")
print(f"  - F1 Score (macro):    {eval_results['eval_f1']:.4f}")
print(f"  - F1 Score (weighted): {eval_results['eval_f1_weighted']:.4f}")
print(f"  - Accuracy:            {eval_results['eval_accuracy']:.4f}")
print(f"  - Precision:           {eval_results['eval_precision']:.4f}")
print(f"  - Recall:              {eval_results['eval_recall']:.4f}")

if USE_WANDB:
    print(f"\n📊 View full results at: {wandb.run.get_url()}")
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