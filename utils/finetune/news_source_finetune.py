"""
BERT Fine-tuning for News Source Classification — 5-Fold Cross Validation
— Balanced training via oversampling + weighted loss —
— Stage 2: Context-Aware (Cross-Attention) —

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
  The element-wise product captures interaction between the comment and the
  most relevant article chunk for that prediction.

  Cross-attention weights are inspected on a few validation samples per fold
  to show which article chunks the model focused on.

Cross-validation strategy:
  • StratifiedKFold(n_splits=5) preserves class distribution in every fold
  • Oversampling applied ONLY to each fold's training split (never the val split)
  • Class weights recomputed per fold from that fold's raw training distribution
  • Fresh model loaded at the start of every fold
  • Best model across all folds (highest val macro-F1) is saved as the final model
  • Mean ± std reported across all folds at the end

Expected CSV format:
    body, comments, labels
    <article body text>, <sinhala comment text>, <0-8>
    ...
  (If 'body' column is absent, the model falls back to comment-only encoding)
"""

import os
import traceback
from collections import Counter
import random as stdlib_random
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from sklearn.model_selection import StratifiedKFold
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
    BertForMaskedLM,
    BertModel,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    EarlyStoppingCallback,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import random
import wandb

# ==================== CONFIGURATION ====================
print("=" * 80)
print("BERT FINE-TUNING — 5-FOLD CV  [CONTEXT-AWARE CROSS-ATTENTION]")
print("=" * 80)

# Model and Tokenizer Paths
BERT_MODEL_PATH  = "HelaBERT"
TOKENIZER_MODEL  = "tokenizer/unigram_32000_0.9995.model"
BERT_CONFIG_FILE = "HelaBERT/config.json"

# Dataset Path
DATA_PATH = "data/Sinhala-News-Source-classification/train/news_source_train.csv"

# ==================== Sliding window (article body → chunks) ====================
CHUNK_SIZE   = 512    # tokens per chunk
CHUNK_STRIDE = 256    # 50% overlap
MAX_CHUNKS   = 16     # cap per article — reduce to 8 if OOM

# ==================== Cross-attention ====================
CROSS_ATTN_HEADS   = 8     # must divide hidden_size (768/8 = 96 per head)
CROSS_ATTN_DROPOUT = 0.2

# Training Parameters
NUM_LABELS                   = 9      # auto-detected from data below
COMMENT_MAX_LENGTH           = 32     # source classification needs less context than category
TRAIN_BATCH_SIZE             = 4      # lower: MAX_CHUNKS+1 BERT passes per sample
EVAL_BATCH_SIZE              = 8
LEARNING_RATE                = 3e-5
NUM_EPOCHS                   = 2     # early stopping decides actual stop point
WARMUP_RATIO                 = 0.06
WEIGHT_DECAY                 = 0.01
GRADIENT_ACCUMULATION_STEPS  = 4      # effective batch = 16
EARLY_STOPPING_PATIENCE      = 3

# Cross-validation
N_FOLDS = 5

# Balancing
OVERSAMPLE_TRAIN  = True
USE_CLASS_WEIGHTS = True

# Output
OUTPUT_DIR     = "HelaBERT_finetuned_news_source_cv"
BEST_MODEL_DIR = f"{OUTPUT_DIR}/best_model"   # saved from the highest-F1 fold
STAGE_TAG      = "cross_attention"

# Misc
RANDOM_SEED = 42
USE_FP16    = True
NUM_WORKERS = 2

# Weights & Biases Configuration
USE_WANDB      = True
WANDB_PROJECT  = "bert-news-source-finetuning"
WANDB_GROUP    = f"5fold_cv_crossattn_lr{LEARNING_RATE}_bs{TRAIN_BATCH_SIZE}"
WANDB_ENTITY   = None

print(f"\n✓ Config loaded — {N_FOLDS}-fold CV, oversampling={'on' if OVERSAMPLE_TRAIN else 'off'}, "
      f"class_weights={'on' if USE_CLASS_WEIGHTS else 'off'}")
print(f"  Architecture   : shared BERT + {CROSS_ATTN_HEADS}-head cross-attention + interaction fusion")
print(f"  Chunk size/stride: {CHUNK_SIZE}/{CHUNK_STRIDE}  max chunks: {MAX_CHUNKS}")
print(f"  Effective batch: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")


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
    print("Running on CPU — training will be slower")


# ==================== VERIFY PATHS ====================
print("\n" + "=" * 80)
print("VERIFYING PATHS")
print("=" * 80)

assert os.path.exists(BERT_MODEL_PATH),  f"Model path not found: {BERT_MODEL_PATH}"
assert os.path.exists(TOKENIZER_MODEL),  f"Tokenizer not found: {TOKENIZER_MODEL}"
assert os.path.exists(DATA_PATH),        f"Data file not found: {DATA_PATH}"

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
    print("CSV has formatting issues, using error-tolerant parsing...")
    df = pd.read_csv(DATA_PATH, engine='python', on_bad_lines='skip')
    print("✓ Loaded with some lines skipped")

df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)

print(f"\nCleaned columns: {df.columns.tolist()}")

possible_body_cols    = [col for col in df.columns if 'body'    in col.lower()]
possible_comment_cols = [col for col in df.columns if 'comment' in col.lower()]
possible_label_cols   = [col for col in df.columns if 'label'   in col.lower()]

if possible_comment_cols and possible_label_cols:
    comment_col = possible_comment_cols[0]
    label_col   = possible_label_cols[0]
    print(f"✓ Identified comment column: '{comment_col}'")
    print(f"✓ Identified label column:   '{label_col}'")
    rename = {comment_col: 'comment', label_col: 'label'}
    if possible_body_cols:
        rename[possible_body_cols[0]] = 'body'
        print(f"✓ Identified body column:    '{possible_body_cols[0]}'")
    df = df.rename(columns=rename)
else:
    print("Could not automatically identify columns — assuming col1=comment, col2=label")
    df = df.iloc[:, -2:]
    df.columns = ['comment', 'label']

df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore')
df = df.dropna(subset=['comment', 'label'])
df['label']   = df['label'].astype(int)
df['comment'] = df['comment'].astype(str).str.strip()

# Fill body column — fall back to empty string (comment-only mode)
if 'body' not in df.columns:
    print("No 'body' column found — using comment-only mode (single dummy chunk)")
    df['body'] = ''
else:
    df['body'] = df['body'].fillna('').astype(str).str.strip()

# Second pass to catch any nulls introduced by coercion
df = df.dropna(subset=['comment', 'label'])

print("✓ Dataset loaded")
print(f"  - Total samples: {len(df)}")
print(f"  - Columns:       {df.columns.tolist()}")
print(f"  - Shape:         {df.shape}")

missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print(f"\nMissing values found — dropping affected rows")
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
    print(f"Updating NUM_LABELS from {NUM_LABELS} to {actual_num_labels}")
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
bodies   = df['body'].tolist()
labels   = df['label'].tolist()

print(f"✓ Extracted {len(comments)} comments and {len(labels)} labels")

print("\nSample examples:")
for i in range(min(3, len(comments))):
    preview = comments[i][:80] + "..." if len(comments[i]) > 80 else comments[i]
    print(f"  {i+1}. [{labels[i]}] {preview}")

all_texts  = comments
all_bodies = bodies
all_labels = labels


# ==================== BODY LENGTH ANALYSIS ====================
print("\n" + "=" * 80)
print("ARTICLE BODY LENGTH ANALYSIS")
print("=" * 80)
body_lengths = df['body'].apply(lambda x: len(sp.encode(x)) if x else 0)
has_bodies   = (body_lengths > 0).sum()
print(f"  Samples with body text: {has_bodies:,} / {len(df):,}")
if has_bodies > 0:
    bl = body_lengths[body_lengths > 0]
    print(f"  min    : {bl.min():,}")
    print(f"  mean   : {bl.mean():.0f}")
    print(f"  median : {bl.median():.0f}")
    print(f"  90th % : {bl.quantile(0.90):.0f}")
    print(f"  max    : {bl.max():,}")
    avg_chunks = bl.apply(
        lambda l: min(math.ceil(max(l - CHUNK_SIZE, 0) / CHUNK_STRIDE) + 1, MAX_CHUNKS)
    ).mean()
    print(f"\nAvg chunks/article (capped {MAX_CHUNKS}): {avg_chunks:.1f}")
    print(f"Overlap per boundary: {CHUNK_SIZE - CHUNK_STRIDE} tokens")


# ==================== SLIDING WINDOW CHUNKER ====================
def tokenize_chunks(text, chunk_size, stride, max_chunks):
    """
    Tokenize text and split into overlapping chunks.
    Returns (chunks_list, num_real_chunks).
    Each entry in chunks_list is (ids_tensor, mask_tensor) of length chunk_size.
    The list is padded with dummy (all-PAD) entries up to max_chunks.
    """
    ids    = sp.encode(str(text)) if text else []
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

    dummy_ids  = torch.full((chunk_size,), PAD_ID, dtype=torch.long)
    dummy_mask = torch.zeros(chunk_size,           dtype=torch.long)
    while len(chunks) < max_chunks:
        chunks.append((dummy_ids.clone(), dummy_mask.clone()))

    return chunks, num_real


# ==================== DATASET CLASS ====================
class CrossAttnNewsSourceDataset(Dataset):
    """
    Each sample exposes:
      chunk_ids    [MAX_CHUNKS, CHUNK_SIZE]  — article body chunks
      chunk_mask   [MAX_CHUNKS, CHUNK_SIZE]
      num_chunks   scalar                    — real (non-padding) chunk count
      comment_ids  [COMMENT_MAX_LENGTH]      — comment / headline as query
      comment_mask [COMMENT_MAX_LENGTH]
      labels       scalar
    If no 'body' column is present, chunk_ids will be a single dummy chunk and
    num_chunks = 1, so the cross-attention degrades gracefully to comment-only.
    """

    def __init__(self, texts, bodies, labels, sp_processor,
                 comment_max_length=32,
                 chunk_size=512, chunk_stride=256, max_chunks=16):
        self.texts              = texts
        self.bodies             = bodies
        self.labels             = labels
        self.sp                 = sp_processor
        self.comment_max_length = comment_max_length
        self.chunk_size         = chunk_size
        self.chunk_stride       = chunk_stride
        self.max_chunks         = max_chunks

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # ==================== Article body → chunks ====================
        chunks, num_real = tokenize_chunks(
            self.bodies[idx], self.chunk_size, self.chunk_stride, self.max_chunks
        )
        chunk_ids  = torch.stack([c[0] for c in chunks])   # [MAX_CHUNKS, CHUNK_SIZE]
        chunk_mask = torch.stack([c[1] for c in chunks])   # [MAX_CHUNKS, CHUNK_SIZE]

        # ==================== Comment → query ====================
        c_ids  = self.sp.encode(self.texts[idx])[:self.comment_max_length]
        c_mask = [1] * len(c_ids)
        pad    = self.comment_max_length - len(c_ids)
        c_ids  += [PAD_ID] * pad
        c_mask += [0]      * pad

        return {
            'chunk_ids':    chunk_ids,
            'chunk_mask':   chunk_mask,
            'num_chunks':   torch.tensor(num_real,               dtype=torch.long),
            'comment_ids':  torch.tensor(c_ids,                  dtype=torch.long),
            'comment_mask': torch.tensor(c_mask,                 dtype=torch.long),
            'labels':       torch.tensor(self.labels[idx],       dtype=torch.long),
        }


# ==================== HELPERS ====================

def oversample(texts, bodies, labels, seed=42):
    """Oversample minority classes to match majority class count."""
    stdlib_random.seed(seed)
    counts    = Counter(labels)
    max_count = max(counts.values())

    bal_texts, bal_bodies, bal_labels = list(texts), list(bodies), list(labels)
    for label, count in counts.items():
        needed = max_count - count
        if needed == 0:
            continue
        indices = [i for i, l in enumerate(labels) if l == label]
        extras  = stdlib_random.choices(indices, k=needed)
        bal_texts  += [texts[i]  for i in extras]
        bal_bodies += [bodies[i] for i in extras]
        bal_labels += [labels[i] for i in extras]

    combined = list(zip(bal_texts, bal_bodies, bal_labels))
    stdlib_random.shuffle(combined)
    bal_texts, bal_bodies, bal_labels = zip(*combined)
    return list(bal_texts), list(bal_bodies), list(bal_labels)


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


# ==================== COLLATOR ====================
def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ==================== CROSS-ATTENTION MODULE ====================
class MultiHeadCrossAttention(nn.Module):
    """
    Comment (query) attends over article chunks (key/value).
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
            return linear(x).view(B, -1, h, d).transpose(1, 2)

        Q = proj_and_split(self.q_proj, query)    # [B, h, 1, d]
        K = proj_and_split(self.k_proj, context)  # [B, h, C, d]
        V = proj_and_split(self.v_proj, context)  # [B, h, C, d]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, h, 1, C]

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        weights  = self.dropout(F.softmax(scores, dim=-1))           # [B, h, 1, C]
        attended = torch.matmul(weights, V).squeeze(2)               # [B, h, d]
        attended = attended.transpose(1, 2).contiguous().view(B, -1) # [B, H]
        return self.out_proj(attended)


# ==================== FULL CONTEXT-AWARE MODEL ====================
class CrossAttnNewsSourceModel(nn.Module):
    """
    Shared BERT + multi-head cross-attention + interaction fusion for
    news source classification.

    Fusion vector = [comment_vec ; attended_ctx ; comment_vec ⊙ attended_ctx]
    The ⊙ (element-wise product) captures interaction between the comment
    and the most attended article passage.
    """

    def __init__(self, bert, hidden_size, num_labels, num_heads, attn_dropout):
        super().__init__()
        self.bert        = bert
        self.hidden_size = hidden_size
        self.cross_attn  = MultiHeadCrossAttention(hidden_size, num_heads, attn_dropout)
        self.fusion_norm = nn.LayerNorm(hidden_size * 3)
        self.dropout     = nn.Dropout(0.1)
        self.classifier  = nn.Linear(hidden_size * 3, num_labels)

    def encode_chunks(self, chunk_ids, chunk_mask, num_chunks):
        """Returns cls_vecs [B, C, H] and chunk_pad_mask [B, C]."""
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

        attended_ctx = self.cross_attn(
            query=comment_vec.unsqueeze(1),
            context=cls_vecs,
            key_padding_mask=pad_mask
        )

        fusion = torch.cat(
            [comment_vec, attended_ctx, comment_vec * attended_ctx], dim=-1
        )
        logits = self.classifier(self.dropout(self.fusion_norm(fusion)))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


# ==================== CUSTOM TRAINER (handles cross-attn batches) ====================
class CrossAttnTrainer(Trainer):
    """Trainer subclass that uses collate_fn and optionally applies class-weight loss."""

    def __init__(self, class_weights: torch.Tensor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def _make_loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=(self.args.per_device_train_batch_size if shuffle
                        else self.args.per_device_eval_batch_size),
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

    def get_train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True)

    def get_eval_dataloader(self, eval_dataset=None):
        return self._make_loader(eval_dataset or self.eval_dataset, shuffle=False)

    def get_test_dataloader(self, test_dataset):
        return self._make_loader(test_dataset, shuffle=False)

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
    """Load a fresh shared-BERT cross-attention model for each fold."""
    if os.path.exists(bert_config_file):
        bert_config = BertConfig.from_json_file(bert_config_file)
        print(f"✓ Config from {bert_config_file}")
    else:
        try:
            bert_config = BertConfig.from_pretrained(bert_model_path)
            print("✓ Config from model dir")
        except Exception:
            bert_config = None
            print("Config not found — using defaults")

    try:
        bert_backbone = BertModel.from_pretrained(bert_model_path)
        print("✓ BertModel loaded")
    except Exception as e:
        print(f"BertModel failed ({e}), extracting from MLM checkpoint...")
        mlm           = BertForMaskedLM.from_pretrained(bert_model_path)
        bert_backbone = mlm.bert
        if bert_config is None:
            bert_config = mlm.config
        print("✓ BERT encoder extracted from MLM checkpoint")

    hidden_size = (bert_config.hidden_size if bert_config
                   else bert_backbone.config.hidden_size)

    assert hidden_size % CROSS_ATTN_HEADS == 0, (
        f"CROSS_ATTN_HEADS ({CROSS_ATTN_HEADS}) must divide hidden_size ({hidden_size}). "
        f"Valid choices: {[h for h in [1,2,4,8,12,16] if hidden_size % h == 0]}"
    )

    model = CrossAttnNewsSourceModel(
        bert=bert_backbone,
        hidden_size=hidden_size,
        num_labels=num_labels,
        num_heads=CROSS_ATTN_HEADS,
        attn_dropout=CROSS_ATTN_DROPOUT,
    )
    return model, bert_config, hidden_size


def print_metric(key: str, value) -> None:
    if isinstance(value, float):
        print(f"    {key}: {value:.4f}")
    else:
        print(f"    {key}: {value}")


# ==================== CROSS-VALIDATION LOOP ====================
print("\n" + "=" * 80)
print(f"STARTING {N_FOLDS}-FOLD CROSS VALIDATION")
print("=" * 80)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Storage for results across folds
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
    fold_train_bodies = [all_bodies[i] for i in train_idx]
    fold_train_labels = [all_labels[i] for i in train_idx]
    fold_val_texts    = [all_texts[i]  for i in val_idx]
    fold_val_bodies   = [all_bodies[i] for i in val_idx]
    fold_val_labels   = [all_labels[i] for i in val_idx]

    print(f"Train: {len(fold_train_texts)} samples  |  Val: {len(fold_val_texts)} samples")

    print("Train label distribution (before oversampling):")
    for lbl, cnt in sorted(Counter(fold_train_labels).items()):
        print(f"Label {lbl}: {cnt}")

    # ==================== Oversample (train only) ====================
    if OVERSAMPLE_TRAIN:
        fold_train_texts, fold_train_bodies, fold_train_labels = oversample(
            fold_train_texts, fold_train_bodies, fold_train_labels, seed=RANDOM_SEED + fold_idx
        )
        print(f"  After oversampling: {len(fold_train_texts)} train samples")
        print("  Train label distribution (after oversampling):")
        for lbl, cnt in sorted(Counter(fold_train_labels).items()):
            print(f"    Label {lbl}: {cnt}")

    # Class weights (computed from raw pre-oversample distribution)
    raw_fold_labels = [all_labels[i] for i in train_idx]
    fold_weights    = compute_class_weights(raw_fold_labels, NUM_LABELS)
    if USE_CLASS_WEIGHTS:
        print("Class weights:")
        for i, w in enumerate(fold_weights):
            print(f"Label {i}: {w.item():.4f}")

    # ==================== Datasets ====================
    train_ds = CrossAttnNewsSourceDataset(
        fold_train_texts, fold_train_bodies, fold_train_labels, sp,
        comment_max_length=COMMENT_MAX_LENGTH,
        chunk_size=CHUNK_SIZE, chunk_stride=CHUNK_STRIDE, max_chunks=MAX_CHUNKS,
    )
    val_ds = CrossAttnNewsSourceDataset(
        fold_val_texts, fold_val_bodies, fold_val_labels, sp,
        comment_max_length=COMMENT_MAX_LENGTH,
        chunk_size=CHUNK_SIZE, chunk_stride=CHUNK_STRIDE, max_chunks=MAX_CHUNKS,
    )

    # ==================== Fresh model ====================
    print(f"Loading fresh model for fold {fold_idx}...")
    model, bert_config, hidden_size = load_fresh_model(BERT_MODEL_PATH, BERT_CONFIG_FILE, NUM_LABELS)
    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_p:,} total, {trainable_p:,} trainable")

    # W&B (one run per fold, all in the same group)
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
                "architecture":           "shared-BERT + multi-head cross-attention + interaction fusion",
                "stage":                  STAGE_TAG,
                "cross_attn_heads":       CROSS_ATTN_HEADS,
                "cross_attn_dropout":     CROSS_ATTN_DROPOUT,
                "chunk_size":             CHUNK_SIZE,
                "chunk_stride":           CHUNK_STRIDE,
                "max_chunks":             MAX_CHUNKS,
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
        print(f"  W&B run: {run.get_url()}")

    # Training args
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
        eval_steps=150,
        save_steps=150,
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=50,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=USE_FP16 and torch.cuda.is_available(),
        dataloader_num_workers=0,   # CrossAttnTrainer handles DataLoaders internally
        seed=RANDOM_SEED + fold_idx,
        report_to="wandb" if USE_WANDB else "none",
        run_name=wandb_run_name if USE_WANDB else None,
        push_to_hub=False,
    )

    # ==================== Trainer ====================
    early_stop = EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)

    trainer = CrossAttnTrainer(
        class_weights=fold_weights if USE_CLASS_WEIGHTS else None,
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics, callbacks=[early_stop],
    )

    # ==================== Train ====================
    try:
        train_result = trainer.train()
        print(f"\nFold {fold_idx} training complete.")
        for k, v in train_result.metrics.items():
            print_metric(k, v)

    except KeyboardInterrupt:
        print(f"\nInterrupted at fold {fold_idx}.")
        os.makedirs(f"{fold_output_dir}/interrupted_model", exist_ok=True)
        torch.save(model.state_dict(), f"{fold_output_dir}/interrupted_model/pytorch_model.bin")

    except Exception as e:
        tb = traceback.format_exc()
        print(f"\nFold {fold_idx} failed: {e}")

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

    print(f"\nClassification report — Fold {fold_idx}:")
    print(classification_report(y_true, y_pred, digits=4))

    # ==================== Save per-fold predictions ====================
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

    # ==================== Per-class metrics ====================
    prec_pc, rec_pc, f1_pc, sup_pc = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    per_class_df = pd.DataFrame({
        'Class':     range(NUM_LABELS),
        'Precision': prec_pc,
        'Recall':    rec_pc,
        'F1-Score':  f1_pc,
        'Support':   sup_pc,
    })
    per_class_df.to_csv(f"{fold_output_dir}/per_class_metrics.csv", index=False)

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(
        cm,
        index=[f"True_{i}"  for i in range(NUM_LABELS)],
        columns=[f"Pred_{i}" for i in range(NUM_LABELS)],
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
                x=[f"Pred_{i}" for i in range(NUM_LABELS)],
                y=[f"True_{i}" for i in range(NUM_LABELS)],
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
            'stage': STAGE_TAG, 'hidden_size': hidden_size,
            'num_labels': NUM_LABELS, 'cross_attn_heads': CROSS_ATTN_HEADS,
            'chunk_size': CHUNK_SIZE, 'chunk_stride': CHUNK_STRIDE,
            'max_chunks': MAX_CHUNKS, 'comment_max_length': COMMENT_MAX_LENGTH,
        }]).to_csv(f"{BEST_MODEL_DIR}/arch_config.csv", index=False)
        print(f"\nNew best model saved (fold {fold_idx}, F1={best_fold_f1:.4f}) → {BEST_MODEL_DIR}")

    print(f"\nFold {fold_idx} done — macro F1: {fold_m['f1']:.4f}")

    # Cross-attention weight inspection (last fold's model, 5 val samples)
    print(f"\nCross-attention weight inspection (fold {fold_idx}, 5 samples):")
    model.eval()
    device    = next(model.parameters()).device
    attn_rows = []

    with torch.no_grad():
        for si in range(min(5, len(val_ds))):
            sample = val_ds[si]
            batch  = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}

            n_real      = batch['num_chunks'].item()
            cls_vecs, pad_mask = model.encode_chunks(
                batch['chunk_ids'], batch['chunk_mask'], batch['num_chunks']
            )
            comment_vec = model.encode_comment(
                batch['comment_ids'], batch['comment_mask']
            )

            ca  = model.cross_attn
            B   = 1
            C   = cls_vecs.shape[1]
            h, d = ca.num_heads, ca.head_dim

            Q = ca.q_proj(comment_vec.unsqueeze(1)).view(B, 1, h, d).transpose(1, 2)
            K = ca.k_proj(cls_vecs).view(B, C, h, d).transpose(1, 2)
            raw = torch.matmul(Q, K.transpose(-2, -1)) / ca.scale
            raw = raw.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            weights = F.softmax(raw, dim=-1).squeeze()
            if weights.dim() == 2:
                chunk_importance = weights.mean(dim=0)[:n_real].cpu().numpy()
            else:
                chunk_importance = weights[:n_real].cpu().numpy()

            comment_text = fold_val_texts[si][:80]
            true_lbl     = str(fold_val_labels[si])
            pred_lbl     = str(int(all_y_pred[-(len(val_ds) - si)]))
            correct_sym  = "✓" if fold_val_labels[si] == int(all_y_pred[-(len(val_ds) - si)]) else "✗"

            print(f"Sample {si+1} [{correct_sym}]  True: {true_lbl}  Pred: {pred_lbl}")
            print(f"Comment: {comment_text}...")
            for ci, w in enumerate(chunk_importance):
                bar = "█" * int(w * 30)
                print(f"      chunk {ci:2d}: {w:.4f}  {bar}")

            for ci, w in enumerate(chunk_importance):
                attn_rows.append({
                    'fold': fold_idx, 'sample_idx': si,
                    'comment': comment_text,
                    'true_label': true_lbl, 'pred_label': pred_lbl,
                    'chunk_idx': ci, 'attn_weight': float(w),
                })

    if attn_rows:
        attn_df = pd.DataFrame(attn_rows)
        attn_df.to_csv(f"{fold_output_dir}/cross_attn_weights.csv", index=False)
        if USE_WANDB:
            wandb.log({"cross_attention_weights": wandb.Table(dataframe=attn_df)})
        print(f"✓ Attention weights saved → {fold_output_dir}/cross_attn_weights.csv")

    if USE_WANDB:
        wandb.finish()
        print(f"✓ W&B fold {fold_idx} run finished")

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
pd.DataFrame(
    oof_cm,
    index=[f"True_{i}"  for i in range(NUM_LABELS)],
    columns=[f"Pred_{i}" for i in range(NUM_LABELS)],
).to_csv(f"{OUTPUT_DIR}/oof_confusion_matrix.csv")

pd.DataFrame({
    'text':            all_oof_texts,
    'true_label':      oof_y_true,
    'predicted_label': oof_y_pred,
    'correct':         oof_y_true == oof_y_pred,
}).to_csv(f"{OUTPUT_DIR}/oof_predictions.csv", index=False)

print(f"\n✓ OOF confusion matrix → {OUTPUT_DIR}/oof_confusion_matrix.csv")
print(f"✓ OOF predictions      → {OUTPUT_DIR}/oof_predictions.csv")


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
    })
    try:
        import plotly.figure_factory as ff
        fig = ff.create_annotated_heatmap(
            z=oof_cm,
            x=[f"Pred_{i}" for i in range(NUM_LABELS)],
            y=[f"True_{i}" for i in range(NUM_LABELS)],
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
    'Architecture':         'Shared BERT + Multi-Head Cross-Attention + Interaction Fusion',
    'Stage':                STAGE_TAG,
    'Task':                 'News Source Classification',
    'Balancing':            'Oversample + Class Weights',
    'N Folds':              N_FOLDS,
    'Num Classes':          NUM_LABELS,
    'Total Samples':        len(all_texts),
    'Cross-Attn Heads':     CROSS_ATTN_HEADS,
    'Chunk Size/Stride':    f'{CHUNK_SIZE}/{CHUNK_STRIDE}',
    'Max Chunks':           MAX_CHUNKS,
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
}

pd.DataFrame([cv_summary]).to_csv(f'{OUTPUT_DIR}/cv_summary.csv', index=False)

print("\n" + "=" * 80)
print("FINAL CROSS-VALIDATION SUMMARY")
print("=" * 80)
for k, v in cv_summary.items():
    print(f"  {k:<28}: {v}")

print("\n" + "=" * 80)
print("5-FOLD CROSS-VALIDATION (CONTEXT-AWARE CROSS-ATTENTION) COMPLETE!")
print("=" * 80)
print(f"\nOutputs saved to: {OUTPUT_DIR}/")
print(f"fold_1/ … fold_{N_FOLDS}/      per-fold predictions, metrics, confusion matrix,")
print(f"                               cross_attn_weights.csv")
print(f"best_model/                    best model weights (fold {best_fold_idx})")
print(f"cv_fold_metrics.csv            per-fold metric table")
print(f"cv_summary.csv                 overall CV summary")
print(f"oof_predictions.csv            out-of-fold predictions (full dataset)")
print(f"oof_confusion_matrix.csv       OOF confusion matrix")
if USE_WANDB and wandb_group_url:
    print(f"\nW&B group: {wandb_group_url}")