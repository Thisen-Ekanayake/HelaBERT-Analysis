"""
BERT Fine-tuning for News Category Classification with W&B Logging

This script fine-tunes a pre-trained BERT model (trained with SentencePiece tokenizer)
on a 5-class news category classification task (labels 0–4) with Weights & Biases logging.

Expected CSV format:
    comments, labels
    <sinhala text>, <0-4>
    ...
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
    EvalPrediction
)
import random
import wandb
from dotenv import load_dotenv

# ==================== LOAD ENVIRONMENT VARIABLES ====================
load_dotenv()  # Loads variables from .env file into os.environ


# ==================== EMAIL NOTIFICATION SETUP ====================

# Email settings — all loaded from .env
EMAIL_NOTIFICATIONS_ENABLED = os.getenv("EMAIL_NOTIFICATIONS_ENABLED", "true").lower() == "true"
EMAIL_SENDER        = os.getenv("EMAIL_SENDER")           # e.g. yourname@gmail.com
EMAIL_PASSWORD      = os.getenv("EMAIL_APP_PASSWORD")     # App password (not your login password)
EMAIL_RECIPIENT     = os.getenv("EMAIL_RECIPIENT")        # Where to send notifications
EMAIL_SMTP_HOST     = os.getenv("EMAIL_SMTP_HOST", "smtp.gmail.com")
EMAIL_SMTP_PORT     = int(os.getenv("EMAIL_SMTP_PORT", "587"))
TRAINING_JOB_NAME   = os.getenv("TRAINING_JOB_NAME", "BERT News Category Fine-tuning")


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


def _build_success_email(summary: dict, eval_results: dict, wandb_url: str = None) -> tuple:
    """Build success email content. Returns (subject, html, plain_text)."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
      <p style='color:#888;font-size:12px'>Sent by news_category_finetune.py</p>
    </body></html>
    """
    plain = (
        f"Training Completed Successfully\n"
        f"Job: {TRAINING_JOB_NAME}\n"
        f"Finished at: {timestamp}\n\n"
        + "\n".join(
            f"{k}: {f'{v:.4f}' if isinstance(v, float) else str(v)}"
            for k, v in summary.items()
        )
        + (f"\n\nW&B: {wandb_url}" if wandb_url else "")
    )
    subject = f"✅ Training Complete — {TRAINING_JOB_NAME}"
    return subject, html, plain


def _build_failure_email(error: Exception, tb_str: str) -> tuple:
    """Build crash email content. Returns (subject, html, plain_text)."""
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
      <p style='color:#888;font-size:12px'>Sent by news_category_finetune.py</p>
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
print("BERT FINE-TUNING FOR NEWS CATEGORY CLASSIFICATION")
print("=" * 80)

# Model and Tokenizer Paths
BERT_MODEL_PATH = "HelaBERT"                              # ← your pretrained BERT path
TOKENIZER_MODEL = "tokenizer/unigram_32000_0.9995.model"  # SentencePiece tokenizer
BERT_CONFIG_FILE = "HelaBERT/config.json"

# Dataset Path
DATA_PATH = "data/Sinhala-News-Category-classification/sinhala-news-categories.csv" # ← CHANGE THIS to your CSV path

# Category labels (0–4) — update names to match your actual categories
CATEGORY_NAMES = {
    0: "Category_0",
    1: "Category_1",
    2: "Category_2",
    3: "Category_3",
    4: "Category_4",
}

# Training Parameters
NUM_LABELS = 5          # 5 news categories (0–4)
MAX_LENGTH = 256        # Longer than source task — category may need more context
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.05
GRADIENT_ACCUMULATION_STEPS = 1

# Output Directory
OUTPUT_DIR = "HelaBERT_finetuned_news_category"

# Random Seed
RANDOM_SEED = 42

# Test Split Ratio
TEST_SIZE = 0.2

# Hardware Settings
USE_FP16 = True
NUM_WORKERS = 2

# Weights & Biases Configuration
USE_WANDB = True
WANDB_PROJECT = "bert-news-category-finetuning"
WANDB_RUN_NAME = f"bert_lr{LEARNING_RATE}_bs{TRAIN_BATCH_SIZE}_ep{NUM_EPOCHS}_category"
WANDB_ENTITY = None
WANDB_RUN_ID_FILE = "wandb_run_id_category.txt"

print("\n✓ Configuration loaded")
print(f"  - Model path:       {BERT_MODEL_PATH}")
print(f"  - Tokenizer:        {TOKENIZER_MODEL}")
print(f"  - Dataset:          {DATA_PATH}")
print(f"  - Output directory: {OUTPUT_DIR}")
print(f"  - Num categories:   {NUM_LABELS}")
print(f"  - W&B logging:      {'Enabled' if USE_WANDB else 'Disabled'}")


# ==================== SET RANDOM SEEDS ====================
random.seed(RANDOM_SEED)
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

assert os.path.exists(BERT_MODEL_PATH), f"❌ Model path not found: {BERT_MODEL_PATH}"
assert os.path.exists(TOKENIZER_MODEL), f"❌ Tokenizer not found: {TOKENIZER_MODEL}"
assert os.path.exists(DATA_PATH),       f"❌ Data file not found: {DATA_PATH}"

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

test_text = "ලෝක පාපන්දු සම්මේලනයේ සභාපති සෙප් බ්ලැටර්"
test_tokens = sp.encode(test_text)
print(f"\nTest tokenization:")
print(f"  - Input:  {test_text}")
print(f"  - Tokens: {test_tokens[:10]}... (first 10 shown)")
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

# Clean up column names
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)

print(f"\nCleaned columns: {df.columns.tolist()}")

# Auto-detect comment and label columns
possible_comment_cols = [col for col in df.columns if 'comment' in col.lower()]
possible_label_cols   = [col for col in df.columns if 'label' in col.lower()]

if possible_comment_cols and possible_label_cols:
    comment_col = possible_comment_cols[0]
    label_col   = possible_label_cols[0]
    print(f"✓ Identified comment column: '{comment_col}'")
    print(f"✓ Identified label column:   '{label_col}'")
    df = df.rename(columns={comment_col: 'comment', label_col: 'label'})
else:
    print("⚠️  Could not auto-detect columns — assuming last two columns are [comment, label]")
    df = df.iloc[:, -2:]
    df.columns = ['comment', 'label']

# Drop unnamed columns and nulls
df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore')
df = df.dropna()
df['comment'] = df['comment'].astype(str).str.strip()

# Coerce label to int (strip whitespace)
df['label'] = df['label'].astype(str).str.strip().astype(int)

print("\n✓ Dataset loaded")
print(f"  - Total samples: {len(df)}")
print(f"  - Columns:       {df.columns.tolist()}")
print(f"  - Shape:         {df.shape}")

# Validate label range
assert df['label'].min() >= 0, "❌ Labels must be >= 0"
assert df['label'].max() <= NUM_LABELS - 1, \
    f"❌ Labels must be <= {NUM_LABELS - 1}, found {df['label'].max()}"

# Label distribution
print("\n" + "-" * 80)
print("LABEL DISTRIBUTION")
print("-" * 80)
label_counts = df['label'].value_counts().sort_index()
for lbl, count in label_counts.items():
    category_name = CATEGORY_NAMES.get(lbl, f"Category_{lbl}")
    pct = 100 * count / len(df)
    print(f"  Label {lbl} ({category_name:15s}): {count:6d} samples ({pct:.1f}%)")

actual_num_labels = df['label'].nunique()
print(f"\nUnique labels found: {actual_num_labels}")
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
    print(f"  {i+1}. [{labels[i]} - {CATEGORY_NAMES.get(labels[i], '?')}] {preview}")

# Stratified train/val split
print(f"\nSplitting data (train: {1-TEST_SIZE:.0%}, val: {TEST_SIZE:.0%})...")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    comments,
    labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=labels
)

print(f"✓ Data split completed")
print(f"  - Training samples:   {len(train_texts)}")
print(f"  - Validation samples: {len(val_texts)}")

print("\nTrain label distribution:")
print(pd.Series(train_labels).value_counts().sort_index().to_string())
print("\nValidation label distribution:")
print(pd.Series(val_labels).value_counts().sort_index().to_string())


# ==================== DEFINE DATASET CLASS ====================
print("\n" + "=" * 80)
print("CREATING DATASET")
print("=" * 80)


class SentencePieceDataset(Dataset):
    """Custom Dataset for BERT fine-tuning with SentencePiece tokenizer."""

    def __init__(self, texts, labels, sp_processor, max_length=128):
        self.texts      = texts
        self.labels     = labels
        self.sp         = sp_processor
        self.max_length = max_length
        self.pad_id     = sp_processor.pad_id()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = self.texts[idx]
        label = self.labels[idx]

        token_ids = self.sp.encode(text)

        # Truncate
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        attention_mask  = [1] * len(token_ids)
        padding_length  = self.max_length - len(token_ids)
        token_ids       = token_ids      + [self.pad_id] * padding_length
        attention_mask  = attention_mask + [0]           * padding_length

        return {
            'input_ids':      torch.tensor(token_ids,      dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels':         torch.tensor(label,          dtype=torch.long)
        }


train_dataset = SentencePieceDataset(train_texts, train_labels, sp, max_length=MAX_LENGTH)
val_dataset   = SentencePieceDataset(val_texts,   val_labels,   sp, max_length=MAX_LENGTH)

print(f"✓ Datasets created")
print(f"  - Train dataset size:      {len(train_dataset)}")
print(f"  - Validation dataset size: {len(val_dataset)}")

sample = train_dataset[0]
print(f"\nSample from dataset:")
print(f"  - input_ids shape:     {sample['input_ids'].shape}")
print(f"  - attention_mask shape:{sample['attention_mask'].shape}")
print(f"  - label:               {sample['labels'].item()} "
      f"({CATEGORY_NAMES.get(sample['labels'].item(), '?')})")
print(f"  - Non-padding tokens:  {sample['attention_mask'].sum().item()}")


# ==================== LOAD MODEL ====================
print("\n" + "=" * 80)
print("LOADING MODEL")
print("=" * 80)

# Load BERT config
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
    config.num_labels = NUM_LABELS
    print(f"\nBERT Configuration:")
    print(f"  - Hidden size:            {config.hidden_size}")
    print(f"  - Num layers:             {config.num_hidden_layers}")
    print(f"  - Num attention heads:    {config.num_attention_heads}")
    print(f"  - Vocab size:             {config.vocab_size}")
    print(f"  - Max position embeddings:{config.max_position_embeddings}")
    print(f"  - Num labels:             {config.num_labels}")

print(f"\nLoading model from: {BERT_MODEL_PATH}")

try:
    base_model = BertModel.from_pretrained(BERT_MODEL_PATH)
    print("✓ Loaded base BERT model")

    if config is None:
        config = BertConfig.from_pretrained(BERT_MODEL_PATH)
        config.num_labels = NUM_LABELS

    model = BertForSequenceClassification(config)

    base_state  = base_model.state_dict()
    model_state = model.state_dict()
    for name, param in base_state.items():
        if name in model_state:
            model_state[name].copy_(param)
    model.load_state_dict(model_state, strict=False)
    print("✓ Model loaded with classification head")

except Exception as e:
    print(f"⚠️  Initial load failed: {e}")
    print("   Attempting to load from MLM checkpoint...")

    try:
        mlm_model = BertForMaskedLM.from_pretrained(BERT_MODEL_PATH)

        if config is None:
            config = mlm_model.config
            config.num_labels = NUM_LABELS

        model = BertForSequenceClassification(config)
        mlm_state   = mlm_model.state_dict()
        model_state = model.state_dict()

        for name, param in mlm_state.items():
            if name.startswith('bert.') and name in model_state:
                model_state[name].copy_(param)

        model.load_state_dict(model_state, strict=False)
        print("✓ Model weights loaded from MLM checkpoint")

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

    accuracy  = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall    = recall_score(labels, preds,    average='macro', zero_division=0)
    f1        = f1_score(labels, preds,        average='macro', zero_division=0)

    # Also compute weighted F1 (better for imbalanced datasets)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)

    return {
        'accuracy':    accuracy,
        'precision':   precision,
        'recall':      recall,
        'f1':          f1,
        'f1_weighted': f1_weighted,
    }


print("✓ Metrics function defined (accuracy, precision, recall, F1, weighted F1)")


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
            "hidden_size":          config.hidden_size          if config else "unknown",
            "num_layers":           config.num_hidden_layers    if config else "unknown",
            "num_attention_heads":  config.num_attention_heads  if config else "unknown",
            # Training
            "task":                    "news_category_classification",
            "num_labels":              NUM_LABELS,
            "learning_rate":           LEARNING_RATE,
            "epochs":                  NUM_EPOCHS,
            "train_batch_size":        TRAIN_BATCH_SIZE,
            "eval_batch_size":         EVAL_BATCH_SIZE,
            "gradient_accumulation":   GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size":    TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "warmup_ratio":            WARMUP_RATIO,
            "weight_decay":            WEIGHT_DECAY,
            "max_length":              MAX_LENGTH,
            "fp16":                    USE_FP16 and torch.cuda.is_available(),
            # Data
            "train_samples":  len(train_texts),
            "val_samples":    len(val_texts),
            "test_size":      TEST_SIZE,
            "random_seed":    RANDOM_SEED,
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
            "train_label_distribution": wandb.Histogram(train_labels),
            "val_label_distribution":   wandb.Histogram(val_labels)
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
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=3,

    # Hardware
    fp16=USE_FP16 and torch.cuda.is_available(),
    dataloader_num_workers=NUM_WORKERS,

    # Reproducibility
    seed=RANDOM_SEED,

    # W&B / reporting
    report_to="wandb" if USE_WANDB else "none",
    run_name=WANDB_RUN_NAME if USE_WANDB else None,

    push_to_hub=False,
)

print("✓ Training arguments configured")
print(f"  - Epochs:              {NUM_EPOCHS}")
print(f"  - Train batch size:    {TRAIN_BATCH_SIZE}")
print(f"  - Effective batch:     {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"  - Learning rate:       {LEARNING_RATE}")
print(f"  - Warmup ratio:        {WARMUP_RATIO}")
print(f"  - Max sequence length: {MAX_LENGTH}")
print(f"  - FP16:                {training_args.fp16}")


# ==================== INITIALIZE TRAINER ====================
print("\n" + "=" * 80)
print("INITIALIZING TRAINER")
print("=" * 80)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("✓ Trainer initialized and ready")


# ==================== TRAIN MODEL ====================
print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)
print(f"Training for {NUM_EPOCHS} epochs on {NUM_LABELS}-class category classification...")
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

target_names = [CATEGORY_NAMES.get(i, f"Category_{i}") for i in range(NUM_LABELS)]
report_text  = classification_report(y_true, y_pred, target_names=target_names, digits=4)
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
    index=[f"True_{CATEGORY_NAMES.get(i, i)}"  for i in range(NUM_LABELS)],
    columns=[f"Pred_{CATEGORY_NAMES.get(i, i)}" for i in range(NUM_LABELS)]
)
cm_df.to_csv(f"{OUTPUT_DIR}/confusion_matrix.csv")
print(f"\n✓ Confusion matrix saved to {OUTPUT_DIR}/confusion_matrix.csv")

if USE_WANDB:
    try:
        import plotly.figure_factory as ff
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=[f"Pred_{CATEGORY_NAMES.get(i, i)}"  for i in range(NUM_LABELS)],
            y=[f"True_{CATEGORY_NAMES.get(i, i)}"  for i in range(NUM_LABELS)],
            colorscale='Blues',
            showscale=True
        )
        fig.update_layout(
            title="Confusion Matrix — News Category",
            xaxis_title="Predicted Label",
            yaxis_title="True Label"
        )
        wandb.log({"confusion_matrix": fig})
        print("✓ Confusion matrix heatmap logged to W&B")
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
    'Category':  [CATEGORY_NAMES.get(i, f"Category_{i}") for i in range(NUM_LABELS)],
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
        data  = [[CATEGORY_NAMES.get(i, str(i)), per_class_report.loc[i, metric]]
                 for i in range(NUM_LABELS)]
        table = wandb.Table(data=data, columns=["Category", metric])
        wandb.log({f"per_class_{metric.lower().replace('-', '_')}":
                   wandb.plot.bar(table, "Category", metric, title=f"Per-Class {metric}")})


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
    print(f"   True: {true_label} ({CATEGORY_NAMES.get(true_label, '?')}) | "
          f"Predicted: {pred_label} ({CATEGORY_NAMES.get(pred_label, '?')}) | "
          f"Confidence: {confidence:.4f}")

    if USE_WANDB and i < 10:
        sample_predictions.append([
            text, true_label, CATEGORY_NAMES.get(true_label, str(true_label)),
            pred_label, CATEGORY_NAMES.get(pred_label, str(pred_label)),
            confidence, "Correct"
        ])

if len(incorrect_indices) > 0:
    print("\n" + "-" * 80)
    print("INCORRECT PREDICTIONS (showing 5)")
    print("-" * 80)
    for i, idx in enumerate(incorrect_indices[:5]):
        text             = val_texts[idx]
        text_preview     = text[:100] + "..." if len(text) > 100 else text
        true_label       = y_true[idx]
        pred_label       = y_pred[idx]
        probs            = torch.softmax(torch.tensor(predictions_output.predictions[idx]), dim=0)
        confidence       = probs[pred_label].item()
        true_confidence  = probs[true_label].item()

        print(f"\n{i+1}. Text: {text_preview}")
        print(f"   True: {true_label} ({CATEGORY_NAMES.get(true_label, '?')}) "
              f"[conf: {true_confidence:.4f}] | "
              f"Predicted: {pred_label} ({CATEGORY_NAMES.get(pred_label, '?')}) "
              f"[conf: {confidence:.4f}]")

        if USE_WANDB and i < 10:
            sample_predictions.append([
                text, true_label, CATEGORY_NAMES.get(true_label, str(true_label)),
                pred_label, CATEGORY_NAMES.get(pred_label, str(pred_label)),
                confidence, "Incorrect"
            ])
else:
    print("\n🎉 No incorrect predictions! Perfect accuracy!")

if USE_WANDB and sample_predictions:
    wandb.log({"sample_predictions": wandb.Table(
        data=sample_predictions,
        columns=["Text", "True Label", "True Category",
                 "Predicted Label", "Predicted Category",
                 "Confidence", "Result"]
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
    'text':              val_texts,
    'true_label':        y_true,
    'true_category':     [CATEGORY_NAMES.get(l, str(l)) for l in y_true],
    'predicted_label':   y_pred,
    'predicted_category':[CATEGORY_NAMES.get(l, str(l)) for l in y_pred],
    'correct':           y_true == y_pred,
    'confidence':        confidences
})
results_df.to_csv(f'{OUTPUT_DIR}/predictions.csv', index=False)
print(f"✓ All predictions saved to {OUTPUT_DIR}/predictions.csv")

summary = {
    'Model':              'BERT with SentencePiece',
    'Task':               'News Category Classification',
    'Num Classes':        NUM_LABELS,
    'Train Samples':      len(train_texts),
    'Val Samples':        len(val_texts),
    'Epochs':             NUM_EPOCHS,
    'Learning Rate':      LEARNING_RATE,
    'Batch Size':         TRAIN_BATCH_SIZE,
    'Max Length':         MAX_LENGTH,
    'Accuracy':           eval_results['eval_accuracy'],
    'Precision (macro)':  eval_results['eval_precision'],
    'Recall (macro)':     eval_results['eval_recall'],
    'F1 (macro)':         eval_results['eval_f1'],
    'F1 (weighted)':      eval_results['eval_f1_weighted'],
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
        print(f"{key:22s}: {value:.4f}")
    else:
        print(f"{key:22s}: {value}")

print("\n" + "=" * 80)
print("🎉 FINE-TUNING COMPLETE!")
print("=" * 80)
print(f"\nAll results saved to: {OUTPUT_DIR}/")
print(f"\nFiles created:")
print(f"  - final_model/            (trained model)")
print(f"  - training_summary.csv    (overall metrics)")
print(f"  - per_class_metrics.csv   (per-class performance)")
print(f"  - predictions.csv         (all validation predictions with category names)")
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

_subj, _html, _plain = _build_success_email(summary, eval_results, _wandb_url)
send_email_notification(_subj, _html, _plain)

print("\n" + "=" * 80)