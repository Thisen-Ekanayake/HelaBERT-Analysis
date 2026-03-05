"""
BERT Fine-tuning for Sentiment Analysis — Stage 2: Context-Aware (Cross-Attention)

Architecture:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Article body → sliding window (512 tok, stride 256, 50% overlap)        │
  │               → BERT encoder per chunk → [CLS] vectors                   │
  │               → chunk_matrix  [B, MAX_CHUNKS, H]                         │
  │                                                                          │
  │  Comment text → BERT encoder → [CLS] → comment_vec  [B, H]               │
  │                                                                          │
  │  Cross-Attention:                                                        │
  │    Query  = comment_vec  [B, 1, H]                                       │
  │    Key    = chunk_matrix [B, MAX_CHUNKS, H]                              │
  │    Value  = chunk_matrix [B, MAX_CHUNKS, H]                              │
  │    → attended_context    [B, H]                                          │
  │                                                                          │
  │  Fusion:                                                                 │
  │    [comment_vec ; attended_ctx ; comment_vec ⊙ attended_ctx]            │
  │    → LayerNorm → Dropout → Linear → num_labels                           │
  └──────────────────────────────────────────────────────────────────────────┘

  Both encoders share the same BERT weights (weight-tied).
  The element-wise product captures interaction between what the commenter said
  and which part of the article they are responding to.

  At the end, cross-attention weights are inspected on a few test samples so
  you can see which article chunks the model focused on per comment.
  A full Stage 1 vs Stage 2 comparison table is printed and saved.
"""

import os
import smtplib
import traceback
import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
from transformers import (
    BertConfig, BertModel, BertForMaskedLM,
    Trainer, TrainingArguments, EvalPrediction
)
from transformers.modeling_outputs import SequenceClassifierOutput
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
TRAINING_JOB_NAME = os.getenv("TRAINING_sentiment_context", "BERT Fine-tuning")


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
print("BERT SENTIMENT — STAGE 2: CROSS-ATTENTION CONTEXT-AWARE MODEL")
print("=" * 80)

# ── File paths ─────────────────────────────────────────────────────────────────
BERT_MODEL_PATH  = "HelaBERT"
TOKENIZER_MODEL  = "tokenizer/unigram_32000_0.9995.model"
BERT_CONFIG_FILE = "HelaBERT/config.json"

TRAIN_DATA_PATH  = "data/sinhala-sentiment-analysis/train.tsv"              # ← CHANGE: train TSV
TEST_DATA_PATH   = "data/sinhala-sentiment-analysis/test.tsv"               # ← CHANGE: test  TSV

# Stage 1 predictions CSV for end-of-run comparison (set None to skip)
STAGE1_PREDICTIONS_CSV = "HelaBERT_sentiment_comments_only/predictions_test.csv"

# ── TSV column names ───────────────────────────────────────────────────────────
BODY_COL    = "body"
COMMENT_COL = "comment_phrase"
LABEL_COL   = "comment_sentiment"

# ── Sliding window ─────────────────────────────────────────────────────────────
CHUNK_SIZE   = 512    # tokens per chunk
CHUNK_STRIDE = 256    # 50% overlap
MAX_CHUNKS   = 16     # cap per article — reduce to 8 if OOM

# ── Sequence lengths ───────────────────────────────────────────────────────────
COMMENT_MAX_LENGTH = 256

# ── Cross-attention ────────────────────────────────────────────────────────────
CROSS_ATTN_HEADS   = 8     # must divide hidden_size (768/8 = 96 per head)
CROSS_ATTN_DROPOUT = 0.1

# ── Training ───────────────────────────────────────────────────────────────────
TRAIN_BATCH_SIZE            = 8     # lower: MAX_CHUNKS+1 BERT passes per sample
EVAL_BATCH_SIZE             = 8
LEARNING_RATE               = 1e-5
NUM_EPOCHS                  = 10
WARMUP_RATIO                = 0.1
WEIGHT_DECAY                = 0.05
GRADIENT_ACCUMULATION_STEPS = 8     # effective batch = 64
VAL_SPLIT                   = 0.1

# ── Output ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "HelaBERT_sentiment_crossattn"
STAGE_TAG  = "cross_attention"

# ── Misc ───────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
USE_FP16    = True
NUM_WORKERS = 2

# ── W&B ────────────────────────────────────────────────────────────────────────
USE_WANDB         = True
WANDB_PROJECT     = "bert-sentiment-analysis"
WANDB_RUN_NAME    = f"bert_{STAGE_TAG}_lr{LEARNING_RATE}_bs{TRAIN_BATCH_SIZE}_ep{NUM_EPOCHS}"
WANDB_ENTITY      = None
WANDB_RUN_ID_FILE = f"wandb_run_id_{STAGE_TAG}.txt"

print("\n✓ Configuration loaded")
print(f"  BERT model       : {BERT_MODEL_PATH}")
print(f"  Train data       : {TRAIN_DATA_PATH}")
print(f"  Test data        : {TEST_DATA_PATH}")
print(f"  Output dir       : {OUTPUT_DIR}")
print(f"  Chunk size/stride: {CHUNK_SIZE} / {CHUNK_STRIDE}  max chunks: {MAX_CHUNKS}")
print(f"  Cross-attn heads : {CROSS_ATTN_HEADS}")
print(f"  Effective batch  : {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")


# ==================== SEEDS ====================
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ==================== ENVIRONMENT ====================
print("\n" + "=" * 80)
print("ENVIRONMENT CHECK")
print("=" * 80)
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM    : {props.total_memory / 1e9:.1f} GB")
else:
    print("⚠️  CPU only — sliding window will be slow")


# ==================== VERIFY PATHS ====================
print("\n" + "=" * 80)
print("VERIFYING PATHS")
print("=" * 80)
assert os.path.exists(BERT_MODEL_PATH), f"❌ {BERT_MODEL_PATH}"
assert os.path.exists(TOKENIZER_MODEL), f"❌ {TOKENIZER_MODEL}"
assert os.path.exists(TRAIN_DATA_PATH), f"❌ {TRAIN_DATA_PATH}"
assert os.path.exists(TEST_DATA_PATH),  f"❌ {TEST_DATA_PATH}"
print("✓ All paths verified")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== TOKENIZER ====================
print("\n" + "=" * 80)
print("LOADING TOKENIZER")
print("=" * 80)
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)
PAD_ID = sp.pad_id()
print(f"✓ SentencePiece  vocab: {sp.get_piece_size()}  PAD_ID: {PAD_ID}")


# ==================== HELPERS ====================
def find_col(df, name):
    if name in df.columns:
        return name
    hits = [c for c in df.columns if name.lower() in c.lower()]
    if hits:
        return hits[0]
    raise KeyError(f"Column '{name}' not found in {list(df.columns)}")


def load_tsv(path):
    try:
        df = pd.read_csv(path, sep='\t')
    except pd.errors.ParserError:
        df = pd.read_csv(path, sep='\t', engine='python', on_bad_lines='skip')
    df.columns = df.columns.str.strip()
    df = df[[find_col(df, BODY_COL),
             find_col(df, COMMENT_COL),
             find_col(df, LABEL_COL)]].copy()
    df.columns = ['body', 'comment', 'label']
    df = df.dropna(subset=['comment', 'label'])
    df['body']    = df['body'].fillna('').astype(str).str.strip()
    df['comment'] = df['comment'].astype(str).str.strip()
    df['label']   = df['label'].astype(str).str.strip().str.upper()
    return df[df['comment'].str.len() > 0].reset_index(drop=True)


def tokenize_chunks(text, chunk_size, stride, max_chunks):
    """
    Tokenize text and split into overlapping chunks.
    Returns (chunks_list, num_real_chunks).
    Each entry in chunks_list is (ids_tensor, mask_tensor) of length chunk_size.
    The list is padded with dummy (all-PAD) entries up to max_chunks.
    """
    ids    = sp.encode(str(text))
    chunks = []
    start  = 0
    while start < len(ids) and len(chunks) < max_chunks:
        end  = min(start + chunk_size, len(ids))
        seg  = ids[start:end]
        mask = [1] * len(seg)
        pad  = chunk_size - len(seg)
        seg  += [PAD_ID] * pad
        mask += [0]      * pad
        chunks.append((torch.tensor(seg, dtype=torch.long),
                       torch.tensor(mask, dtype=torch.long)))
        if end == len(ids):
            break
        start += stride

    num_real = max(len(chunks), 1)

    # Pad up to max_chunks
    dummy_ids  = torch.full((chunk_size,), PAD_ID, dtype=torch.long)
    dummy_mask = torch.zeros(chunk_size,           dtype=torch.long)
    while len(chunks) < max_chunks:
        chunks.append((dummy_ids.clone(), dummy_mask.clone()))

    return chunks, num_real


# ==================== LOAD DATA ====================
print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)
train_df = load_tsv(TRAIN_DATA_PATH)
test_df  = load_tsv(TEST_DATA_PATH)
print(f"✓ Train: {len(train_df):,}  Test: {len(test_df):,}")


# ==================== ENCODE LABELS ====================
print("\n" + "=" * 80)
print("ENCODING LABELS")
print("=" * 80)
all_labels = pd.concat([train_df['label'], test_df['label']]).unique()
le = LabelEncoder()
le.fit(sorted(all_labels))
train_df['label_id'] = le.transform(train_df['label'])
test_df['label_id']  = le.transform(test_df['label'])
NUM_LABELS  = len(le.classes_)
id_to_label = {i: lbl for i, lbl in enumerate(le.classes_)}

mapping_df = pd.DataFrame({'label_id': list(id_to_label.keys()),
                            'label_name': list(id_to_label.values())})
mapping_df.to_csv(f"{OUTPUT_DIR}/label_mapping.csv", index=False)
print(f"✓ {NUM_LABELS} labels: {', '.join(le.classes_)}")
for idx, lbl in sorted(id_to_label.items()):
    tr = (train_df['label_id'] == idx).sum()
    te = (test_df['label_id']  == idx).sum()
    print(f"  [{idx}] {lbl:20s}  train: {tr:5d}  test: {te:5d}")


# ==================== BODY LENGTH ANALYSIS ====================
print("\n" + "=" * 80)
print("ARTICLE BODY LENGTH ANALYSIS")
print("=" * 80)
lengths = train_df['body'].apply(lambda x: len(sp.encode(x)))
print(f"  min    : {lengths.min():,}")
print(f"  mean   : {lengths.mean():.0f}")
print(f"  median : {lengths.median():.0f}")
print(f"  90th % : {lengths.quantile(0.90):.0f}")
print(f"  max    : {lengths.max():,}")
avg_c = lengths.apply(
    lambda l: min(math.ceil(max(l - CHUNK_SIZE, 0) / CHUNK_STRIDE) + 1, MAX_CHUNKS)
).mean()
print(f"\n  Avg chunks/article (capped {MAX_CHUNKS}): {avg_c:.1f}")
print(f"  Overlap per boundary: {CHUNK_SIZE - CHUNK_STRIDE} tokens")


# ==================== TRAIN / VAL SPLIT ====================
print("\n" + "=" * 80)
print("TRAIN / VALIDATION SPLIT")
print("=" * 80)
tr_idx, val_idx = train_test_split(
    range(len(train_df)), test_size=VAL_SPLIT,
    random_state=RANDOM_SEED, stratify=train_df['label_id'].tolist()
)
tr_df  = train_df.iloc[tr_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)
print(f"  Train: {len(tr_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")


# ==================== DATASET ====================
print("\n" + "=" * 80)
print("CREATING DATASETS")
print("=" * 80)


class CrossAttnDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Article chunks
        chunks, num_real = tokenize_chunks(
            row['body'], CHUNK_SIZE, CHUNK_STRIDE, MAX_CHUNKS
        )
        chunk_ids  = torch.stack([c[0] for c in chunks])   # [MAX_CHUNKS, CHUNK_SIZE]
        chunk_mask = torch.stack([c[1] for c in chunks])   # [MAX_CHUNKS, CHUNK_SIZE]

        # Comment
        c_ids  = sp.encode(str(row['comment']))[:COMMENT_MAX_LENGTH]
        c_mask = [1] * len(c_ids)
        pad    = COMMENT_MAX_LENGTH - len(c_ids)
        c_ids  += [PAD_ID] * pad
        c_mask += [0]      * pad

        return {
            'chunk_ids':    chunk_ids,
            'chunk_mask':   chunk_mask,
            'num_chunks':   torch.tensor(num_real,           dtype=torch.long),
            'comment_ids':  torch.tensor(c_ids,             dtype=torch.long),
            'comment_mask': torch.tensor(c_mask,            dtype=torch.long),
            'labels':       torch.tensor(int(row['label_id']), dtype=torch.long),
        }


train_dataset = CrossAttnDataset(tr_df)
val_dataset   = CrossAttnDataset(val_df)
test_dataset  = CrossAttnDataset(test_df)

print(f"✓ train: {len(train_dataset):,}  val: {len(val_dataset):,}  test: {len(test_dataset):,}")
s = train_dataset[0]
print(f"\n  chunk_ids  : {s['chunk_ids'].shape}   [MAX_CHUNKS × CHUNK_SIZE]")
print(f"  num_chunks : {s['num_chunks'].item()}  real chunks for this article")
print(f"  comment_ids: {s['comment_ids'].shape}")
print(f"  label      : {s['labels'].item()} → {id_to_label[s['labels'].item()]}")


# ==================== CROSS-ATTENTION MODULE ====================
class MultiHeadCrossAttention(nn.Module):
    """
    Comment queries article chunks.
      Query  : comment_vec    [B, 1, H]
      Key/Val: chunk_vecs     [B, C, H]
      Output : attended_ctx   [B, H]

    Dummy-chunk mask prevents the model from attending to PAD-only chunks.
    """

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads   = num_heads
        self.head_dim    = hidden_size // num_heads
        self.scale       = math.sqrt(self.head_dim)

        self.q_proj   = nn.Linear(hidden_size, hidden_size)
        self.k_proj   = nn.Linear(hidden_size, hidden_size)
        self.v_proj   = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, query, context, key_padding_mask=None):
        """
        query           : [B, 1, H]
        context         : [B, C, H]
        key_padding_mask: [B, C] — True = ignore (dummy chunk)
        Returns         : [B, H]
        """
        B = query.shape[0]
        C = context.shape[1]
        h = self.num_heads
        d = self.head_dim

        def proj_and_split(linear, x):
            # [B, S, H] → [B, h, S, d]
            return linear(x).view(B, -1, h, d).transpose(1, 2)

        Q = proj_and_split(self.q_proj, query)    # [B, h, 1, d]
        K = proj_and_split(self.k_proj, context)  # [B, h, C, d]
        V = proj_and_split(self.v_proj, context)  # [B, h, C, d]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, h, 1, C]

        if key_padding_mask is not None:
            # [B, C] → [B, 1, 1, C]
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        weights  = self.dropout(F.softmax(scores, dim=-1))          # [B, h, 1, C]
        attended = torch.matmul(weights, V).squeeze(2)              # [B, h, d]
        attended = attended.transpose(1, 2).contiguous().view(B, -1) # [B, H]
        return self.out_proj(attended)


# ==================== FULL MODEL ====================
class CrossAttnSentimentModel(nn.Module):
    """
    Shared BERT + multi-head cross-attention + interaction fusion.

    Fusion vector = [comment_vec ; attended_ctx ; comment_vec ⊙ attended_ctx]
    The ⊙ (element-wise product) captures whether the comment and the relevant
    article passage carry similar or opposing signals.
    """

    def __init__(self, bert, hidden_size, num_labels, num_heads, attn_dropout):
        super().__init__()
        self.bert        = bert
        self.hidden_size = hidden_size
        self.cross_attn  = MultiHeadCrossAttention(hidden_size, num_heads, attn_dropout)
        # Fusion dim = 3 * H
        self.fusion_norm = nn.LayerNorm(hidden_size * 3)
        self.dropout     = nn.Dropout(0.1)
        self.classifier  = nn.Linear(hidden_size * 3, num_labels)

    def encode_chunks(self, chunk_ids, chunk_mask, num_chunks):
        """
        Returns:
          cls_vecs       [B, C, H]
          chunk_pad_mask [B, C]  True = dummy chunk
        """
        B, C, L = chunk_ids.shape
        out      = self.bert(
            input_ids=chunk_ids.view(B * C, L),
            attention_mask=chunk_mask.view(B * C, L)
        )
        cls_vecs = out.last_hidden_state[:, 0, :].view(B, C, -1)   # [B, C, H]

        idx_range      = torch.arange(C, device=chunk_ids.device).unsqueeze(0)
        chunk_pad_mask = idx_range >= num_chunks.unsqueeze(1)       # [B, C]
        return cls_vecs, chunk_pad_mask

    def encode_comment(self, comment_ids, comment_mask):
        out = self.bert(input_ids=comment_ids, attention_mask=comment_mask)
        return out.last_hidden_state[:, 0, :]   # [B, H]

    def forward(self, chunk_ids, chunk_mask, num_chunks,
                comment_ids, comment_mask, labels=None):

        cls_vecs, pad_mask = self.encode_chunks(chunk_ids, chunk_mask, num_chunks)
        comment_vec        = self.encode_comment(comment_ids, comment_mask)

        # comment attends to relevant article chunks
        attended_ctx = self.cross_attn(
            query=comment_vec.unsqueeze(1),   # [B, 1, H]
            context=cls_vecs,                 # [B, C, H]
            key_padding_mask=pad_mask         # [B, C]
        )                                     # → [B, H]

        # Three-way fusion
        fusion = torch.cat(
            [comment_vec, attended_ctx, comment_vec * attended_ctx], dim=-1
        )                                     # [B, 3H]

        logits = self.classifier(self.dropout(self.fusion_norm(fusion)))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


# ==================== LOAD BERT BACKBONE ====================
print("\n" + "=" * 80)
print("LOADING MODEL")
print("=" * 80)

if os.path.exists(BERT_CONFIG_FILE):
    bert_config = BertConfig.from_json_file(BERT_CONFIG_FILE)
    print(f"✓ Config from {BERT_CONFIG_FILE}")
else:
    try:
        bert_config = BertConfig.from_pretrained(BERT_MODEL_PATH)
        print("✓ Config from model dir")
    except Exception:
        bert_config = None
        print("⚠️  Config not found")

print(f"\nLoading weights: {BERT_MODEL_PATH}")
try:
    bert_backbone = BertModel.from_pretrained(BERT_MODEL_PATH)
    print("✓ BertModel loaded")
except Exception as e:
    print(f"  BertModel failed ({e}), extracting from MLM checkpoint...")
    mlm           = BertForMaskedLM.from_pretrained(BERT_MODEL_PATH)
    bert_backbone = mlm.bert
    if bert_config is None:
        bert_config = mlm.config
    print("✓ BERT encoder extracted from MLM checkpoint")

hidden_size = bert_config.hidden_size if bert_config else bert_backbone.config.hidden_size

assert hidden_size % CROSS_ATTN_HEADS == 0, (
    f"CROSS_ATTN_HEADS ({CROSS_ATTN_HEADS}) must divide hidden_size ({hidden_size}). "
    f"Valid choices: {[h for h in [1,2,4,8,12,16] if hidden_size % h == 0]}"
)

model = CrossAttnSentimentModel(
    bert=bert_backbone,
    hidden_size=hidden_size,
    num_labels=NUM_LABELS,
    num_heads=CROSS_ATTN_HEADS,
    attn_dropout=CROSS_ATTN_DROPOUT
)

total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
extra     = sum(p.numel() for p in
                list(model.cross_attn.parameters()) +
                list(model.fusion_norm.parameters()) +
                list(model.classifier.parameters()))

print(f"\nModel statistics:")
print(f"  Total params      : {total:,}")
print(f"  Trainable params  : {trainable:,}  ({100*trainable/total:.1f}%)")
print(f"  Extra params      : {extra:,}  (cross-attn + fusion norm + head)")
print(f"  Cross-attn heads  : {CROSS_ATTN_HEADS}  head dim: {hidden_size // CROSS_ATTN_HEADS}")
print(f"  Fusion input dim  : {hidden_size * 3}  (comment + ctx + comment⊙ctx)")


# ==================== COLLATOR + CUSTOM TRAINER ====================
def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


class CrossAttnTrainer(Trainer):
    def _make_loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=(self.args.per_device_train_batch_size if shuffle
                        else self.args.per_device_eval_batch_size),
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available()
        )

    def get_train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True)

    def get_eval_dataloader(self, eval_dataset=None):
        return self._make_loader(eval_dataset or self.eval_dataset, shuffle=False)

    def get_test_dataloader(self, test_dataset):
        return self._make_loader(test_dataset, shuffle=False)


# ==================== METRICS ====================
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


# ==================== W&B ====================
if USE_WANDB:
    print("\n" + "=" * 80)
    print("INITIALIZING W&B")
    print("=" * 80)
    wandb_run_id = None
    if os.path.exists(WANDB_RUN_ID_FILE):
        with open(WANDB_RUN_ID_FILE) as f:
            wandb_run_id = f.read().strip()
    if wandb.run is None:
        run = wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY,
            name=WANDB_RUN_NAME, id=wandb_run_id, resume="allow",
            config={
                "stage":               STAGE_TAG,
                "architecture":        "shared-BERT + multi-head cross-attention + interaction fusion",
                "cross_attn_heads":    CROSS_ATTN_HEADS,
                "cross_attn_dropout":  CROSS_ATTN_DROPOUT,
                "fusion":              "[comment ; ctx ; comment*ctx] → LayerNorm → Linear",
                "chunk_size":          CHUNK_SIZE,
                "chunk_stride":        CHUNK_STRIDE,
                "max_chunks":          MAX_CHUNKS,
                "comment_max_length":  COMMENT_MAX_LENGTH,
                "num_labels":          NUM_LABELS,
                "label_names":         list(le.classes_),
                "hidden_size":         hidden_size,
                "extra_params":        extra,
                "learning_rate":       LEARNING_RATE,
                "epochs":              NUM_EPOCHS,
                "train_batch_size":    TRAIN_BATCH_SIZE,
                "effective_batch":     TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
                "warmup_ratio":        WARMUP_RATIO,
                "weight_decay":        WEIGHT_DECAY,
                "fp16":                USE_FP16 and torch.cuda.is_available(),
                "train_samples":       len(tr_df),
                "val_samples":         len(val_df),
                "test_samples":        len(test_df),
            }
        )
        with open(WANDB_RUN_ID_FILE, 'w') as f:
            f.write(run.id)
        print(f"✓ W&B run  : {run.get_url()}")
        wandb.log({
            "train_label_dist": wandb.Histogram(tr_df['label_id'].tolist()),
            "val_label_dist":   wandb.Histogram(val_df['label_id'].tolist()),
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
    dataloader_num_workers=0,   # handled inside CrossAttnTrainer
    seed=RANDOM_SEED,
    report_to="wandb" if USE_WANDB else "none",
    run_name=WANDB_RUN_NAME if USE_WANDB else None,
    push_to_hub=False,
)

trainer = CrossAttnTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

approx_steps = (len(tr_df) * NUM_EPOCHS
                // (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
print(f"✓ Trainer ready  |  approx optimiser steps: ~{approx_steps:,}")
print(f"  Memory note: each step runs {MAX_CHUNKS}+1 BERT forward passes per sample in the batch")


# ==================== TRAINING ====================
print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)
print(f"  Architecture: shared BERT + {CROSS_ATTN_HEADS}-head cross-attention")
print(f"  Fusion      : [comment ; ctx ; comment⊙ctx] → LayerNorm → Linear")
print(f"  Labels      : {', '.join(le.classes_)}")
if USE_WANDB:
    print(f"  W&B         : {wandb.run.get_url()}")
print()

try:
    train_result = trainer.train()
    print("\n✓ Training complete")
    for k, v in train_result.metrics.items():
        print(f"  {k}: {v:.4f}")

except KeyboardInterrupt:
    print("\nInterrupted — saving model...")
    os.makedirs(f"{OUTPUT_DIR}/interrupted_model", exist_ok=True)
    torch.save(model.state_dict(),
               f"{OUTPUT_DIR}/interrupted_model/pytorch_model.bin")
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


# ==================== SAVE MODEL ====================
print("\n" + "=" * 80)
print("SAVING MODEL")
print("=" * 80)
final_model_path = f"{OUTPUT_DIR}/final_model"
os.makedirs(final_model_path, exist_ok=True)
torch.save(model.state_dict(), f"{final_model_path}/pytorch_model.bin")
if bert_config:
    bert_config.save_pretrained(final_model_path)
mapping_df.to_csv(f"{final_model_path}/label_mapping.csv", index=False)
pd.DataFrame([{
    'stage': STAGE_TAG, 'hidden_size': hidden_size,
    'num_labels': NUM_LABELS, 'cross_attn_heads': CROSS_ATTN_HEADS,
    'chunk_size': CHUNK_SIZE, 'chunk_stride': CHUNK_STRIDE,
    'max_chunks': MAX_CHUNKS, 'comment_max_length': COMMENT_MAX_LENGTH,
}]).to_csv(f"{final_model_path}/arch_config.csv", index=False)
print(f"✓ Model saved to {final_model_path}/")



# ==================== EVALUATE ====================
print("\n" + "=" * 80)
print("VALIDATION SET EVALUATION")
print("=" * 80)
val_results = trainer.evaluate()
for k, v in val_results.items():
    print(f"  {k:30s}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

print("\n" + "=" * 80)
print("TEST SET EVALUATION  (held-out)")
print("=" * 80)
test_output  = trainer.predict(test_dataset)
y_pred       = np.argmax(test_output.predictions, axis=-1)
y_true       = test_df['label_id'].values

test_metrics = {
    'accuracy':    accuracy_score(y_true, y_pred),
    'precision':   precision_score(y_true, y_pred, average='macro',    zero_division=0),
    'recall':      recall_score(y_true, y_pred,    average='macro',    zero_division=0),
    'f1':          f1_score(y_true, y_pred,        average='macro',    zero_division=0),
    'f1_weighted': f1_score(y_true, y_pred,        average='weighted', zero_division=0),
}
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
        data=[[report_text]], columns=["report"])})


# ==================== CONFUSION MATRIX ====================
cm    = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=[f"True_{id_to_label[i]}"  for i in range(NUM_LABELS)],
    columns=[f"Pred_{id_to_label[i]}" for i in range(NUM_LABELS)]
)
cm_df.to_csv(f"{OUTPUT_DIR}/confusion_matrix_test.csv")
if USE_WANDB:
    try:
        import plotly.figure_factory as ff
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=[f"Pred_{id_to_label[i]}" for i in range(NUM_LABELS)],
            y=[f"True_{id_to_label[i]}" for i in range(NUM_LABELS)],
            colorscale='Blues', showscale=True
        )
        fig.update_layout(title=f"Confusion Matrix — {STAGE_TAG}",
                          xaxis_title="Predicted", yaxis_title="True")
        wandb.log({"test/confusion_matrix": fig})
    except ImportError:
        pass


# ==================== PER-CLASS METRICS ====================
prec_pc, rec_pc, f1_pc, sup_pc = precision_recall_fscore_support(
    y_true, y_pred, average=None, zero_division=0
)
per_class_df = pd.DataFrame({
    'Label ID':  range(NUM_LABELS),
    'Sentiment': target_names,
    'Precision': prec_pc,
    'Recall':    rec_pc,
    'F1-Score':  f1_pc,
    'Support':   sup_pc
})
print("\n" + "=" * 80)
print("PER-CLASS METRICS (test set)")
print("=" * 80)
print(per_class_df.to_string(index=False))
per_class_df.to_csv(f"{OUTPUT_DIR}/per_class_metrics_test.csv", index=False)
if USE_WANDB:
    wandb.log({"test/per_class_metrics": wandb.Table(dataframe=per_class_df)})
    for metric in ['Precision', 'Recall', 'F1-Score']:
        data  = [[target_names[i], per_class_df.loc[i, metric]] for i in range(NUM_LABELS)]
        tbl   = wandb.Table(data=data, columns=["Sentiment", metric])
        wandb.log({f"test/per_class_{metric.lower().replace('-','_')}":
                   wandb.plot.bar(tbl, "Sentiment", metric, title=f"Per-Sentiment {metric}")})


# ==================== SAVE PREDICTIONS ====================
confidences = [
    torch.softmax(torch.tensor(test_output.predictions[i]), dim=0)[y_pred[i]].item()
    for i in range(len(y_pred))
]
results_df = pd.DataFrame({
    'comment':             test_df['comment'].tolist(),
    'body_snippet':        test_df['body'].str[:120].tolist(),
    'true_label_id':       y_true,
    'true_sentiment':      [id_to_label[l] for l in y_true],
    'predicted_label_id':  y_pred,
    'predicted_sentiment': [id_to_label[l] for l in y_pred],
    'correct':             y_true == y_pred,
    'confidence':          confidences,
})
results_df.to_csv(f"{OUTPUT_DIR}/predictions_test.csv", index=False)
print(f"\n✓ Predictions saved to {OUTPUT_DIR}/predictions_test.csv")


# ==================== CROSS-ATTENTION WEIGHT INSPECTION ====================
print("\n" + "=" * 80)
print("CROSS-ATTENTION WEIGHT INSPECTION (5 test samples)")
print("=" * 80)
print("Shows which article chunks each comment attended to most.")
print()

model.eval()
device    = next(model.parameters()).device
attn_rows = []

with torch.no_grad():
    for si in range(min(5, len(test_dataset))):
        sample = test_dataset[si]
        batch  = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}

        B          = 1
        n_real     = batch['num_chunks'].item()
        cls_vecs, pad_mask = model.encode_chunks(
            batch['chunk_ids'], batch['chunk_mask'], batch['num_chunks']
        )
        comment_vec = model.encode_comment(
            batch['comment_ids'], batch['comment_mask']
        )

        # Recompute attention weights without dropout for inspection
        ca = model.cross_attn
        h, d = ca.num_heads, ca.head_dim
        C    = cls_vecs.shape[1]

        Q = ca.q_proj(comment_vec.unsqueeze(1)).view(B, 1, h, d).transpose(1, 2)
        K = ca.k_proj(cls_vecs).view(B, C, h, d).transpose(1, 2)

        raw = torch.matmul(Q, K.transpose(-2, -1)) / ca.scale   # [1, h, 1, C]
        raw = raw.masked_fill(
            pad_mask.unsqueeze(1).unsqueeze(2), float('-inf')
        )
        # Average attention across heads → per-chunk importance
        weights = F.softmax(raw, dim=-1).squeeze()               # [h, C] or [C]
        if weights.dim() == 2:
            chunk_importance = weights.mean(dim=0)[:n_real].cpu().numpy()
        else:
            chunk_importance = weights[:n_real].cpu().numpy()

        comment_text = test_df.iloc[si]['comment'][:80]
        true_lbl     = id_to_label[y_true[si]]
        pred_lbl     = id_to_label[y_pred[si]]
        correct_sym  = "✓" if y_true[si] == y_pred[si] else "✗"

        print(f"Sample {si+1} [{correct_sym}]")
        print(f"  Comment : {comment_text}...")
        print(f"  True    : {true_lbl}   Predicted: {pred_lbl}")
        print(f"  Chunks  : {n_real} real  |  chunk attention weights (avg across {h} heads):")
        for ci, w in enumerate(chunk_importance):
            bar = "█" * int(w * 40)
            print(f"    chunk {ci:2d}: {w:.4f}  {bar}")
        print()

        for ci, w in enumerate(chunk_importance):
            attn_rows.append({
                'sample_idx':     si,
                'comment':        comment_text,
                'true_sentiment': true_lbl,
                'pred_sentiment': pred_lbl,
                'chunk_idx':      ci,
                'attn_weight':    float(w),
            })

if USE_WANDB and attn_rows:
    wandb.log({"cross_attention_weights": wandb.Table(
        dataframe=pd.DataFrame(attn_rows))})
    print("✓ Attention weights logged to W&B")

model.train()


# ==================== STAGE COMPARISON ====================
print("\n" + "=" * 80)
print("STAGE COMPARISON:  COMMENTS ONLY  vs  CROSS-ATTENTION CONTEXT")
print("=" * 80)

metrics_order  = ['accuracy', 'precision', 'recall', 'f1', 'f1_weighted']
stage1_metrics = None

if STAGE1_PREDICTIONS_CSV and os.path.exists(STAGE1_PREDICTIONS_CSV):
    try:
        s1 = pd.read_csv(STAGE1_PREDICTIONS_CSV)
        s1_true = s1['true_label_id'].values
        s1_pred = s1['predicted_label_id'].values
        stage1_metrics = {
            'accuracy':    accuracy_score(s1_true, s1_pred),
            'precision':   precision_score(s1_true, s1_pred, average='macro',    zero_division=0),
            'recall':      recall_score(s1_true, s1_pred,    average='macro',    zero_division=0),
            'f1':          f1_score(s1_true, s1_pred,        average='macro',    zero_division=0),
            'f1_weighted': f1_score(s1_true, s1_pred,        average='weighted', zero_division=0),
        }
        print(f"✓ Stage 1 results loaded from {STAGE1_PREDICTIONS_CSV}\n")
    except Exception as e:
        print(f"⚠️  Could not load Stage 1 results: {e}\n")
else:
    print(f"⚠️  STAGE1_PREDICTIONS_CSV not found — set it to enable comparison\n")

W = 22
header = f"  {'Metric':18s}  {'Stage 1 (comments)':>{W}}  {'Stage 2 (cross-attn)':>{W}}"
if stage1_metrics:
    header += f"  {'Δ (S2 - S1)':>{W}}"
print(header)
print("  " + "-" * (len(header) - 2))

for m in metrics_order:
    s2 = test_metrics[m]
    if stage1_metrics:
        s1    = stage1_metrics[m]
        delta = s2 - s1
        sign  = "+" if delta >= 0 else ""
        print(f"  {m:18s}  {s1:>{W}.4f}  {s2:>{W}.4f}  {sign}{delta:>{W-1}.4f}")
    else:
        print(f"  {m:18s}  {'N/A':>{W}}  {s2:>{W}.4f}")

print("  " + "-" * (len(header) - 2))

if stage1_metrics:
    delta_f1 = test_metrics['f1'] - stage1_metrics['f1']
    if delta_f1 > 0.01:
        print(f"\n  ✅ Cross-attention context improves macro-F1 by +{delta_f1:.4f}")
    elif delta_f1 < -0.01:
        print(f"\n  ⚠️  Baseline is better by {abs(delta_f1):.4f} — "
              f"try more epochs or reduce MAX_CHUNKS")
    else:
        print(f"\n  ↔️  Roughly equivalent (Δ macro-F1 = {delta_f1:+.4f})")

    comp_rows = [{'metric': m,
                  'stage1_comments_only':    stage1_metrics[m],
                  'stage2_cross_attention':  test_metrics[m],
                  'delta': test_metrics[m] - stage1_metrics[m]}
                 for m in metrics_order]
    pd.DataFrame(comp_rows).to_csv(f"{OUTPUT_DIR}/stage_comparison.csv", index=False)
    if USE_WANDB:
        wandb.log({"stage_comparison": wandb.Table(
            dataframe=pd.DataFrame(comp_rows))})
    print(f"\n  ✓ Comparison saved to {OUTPUT_DIR}/stage_comparison.csv")


# ==================== FINAL SUMMARY ====================
print("\n" + "=" * 80)
print("🎉 STAGE 2 (CROSS-ATTENTION CONTEXT-AWARE) COMPLETE!")
print("=" * 80)
print(f"\nOutputs in: {OUTPUT_DIR}/")
print(f"  final_model/                — weights, bert config, label map, arch config")
print(f"  label_mapping.csv           — id ↔ sentiment name")
print(f"  per_class_metrics_test.csv  — per-class breakdown")
print(f"  predictions_test.csv        — test predictions + body snippet")
print(f"  confusion_matrix_test.csv   — confusion matrix")
print(f"  stage_comparison.csv        — Stage 1 vs Stage 2 delta table")

if USE_WANDB:
    print(f"\n📊 Full results: {wandb.run.get_url()}")
    _wandb_url = wandb.run.get_url()
    wandb.finish()
else:
    _wandb_url = None

# ==================== SEND COMPLETION EMAIL ====================
print("\n" + "=" * 80)
print("SENDING COMPLETION NOTIFICATION")
print("=" * 80)
_subj, _html, _plain = _build_success_email(test_metrics, _wandb_url)
send_email_notification(_subj, _html, _plain)

print("\n" + "=" * 80)