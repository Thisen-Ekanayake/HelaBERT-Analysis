"""
BERT Fine-tuning for Sentiment Analysis — 5-Fold Cross Validation
— Balanced training via oversampling + weighted loss —
— Stage 3: Bidirectional Context-Aware (Bidirectional Cross-Attention) —

Architecture:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Article body → sliding window (512 tok, stride 256, 50% overlap)        │
  │               → BERT encoder per chunk → [CLS] vectors                   │
  │               → chunk_matrix  [B, MAX_CHUNKS, H]                         │
  │                                                                          │
  │  Comment text → BERT encoder → [CLS] → comment_vec  [B, H]               │
  │                                                                          │
  │  FORWARD Cross-Attention (comment → chunks):                             │
  │    Query  = comment_vec  [B, 1, H]                                       │
  │    Key    = chunk_matrix [B, MAX_CHUNKS, H]                              │
  │    Value  = chunk_matrix [B, MAX_CHUNKS, H]                              │
  │    → attended_context_fwd    [B, H]                                      │
  │                                                                          │
  │  REVERSE Cross-Attention (chunks → comment):                             │
  │    Query  = chunk_matrix [B, MAX_CHUNKS, H]                              │
  │    Key    = comment_vec  [B, 1, H]                                       │
  │    Value  = comment_vec  [B, 1, H]                                       │
  │    → attended_comment_bwd    [B, MAX_CHUNKS, H]                          │
  │    → aggregate via mean pool [B, H]                                      │
  │                                                                          │
  │  Fusion:                                                                 │
  │    [comment_vec ; attended_ctx_fwd ; attended_comment_bwd;              │
  │     comment_vec ⊙ attended_ctx_fwd ; comment_vec ⊙ attended_comment_bwd]│
  │    → LayerNorm → Dropout → Linear → num_labels                           │
  └──────────────────────────────────────────────────────────────────────────┘

  Both encoders share the same BERT weights (weight-tied).
  Bidirectional attention captures:
  • What parts of the article the comment is responding to (forward)
  • How the article chunks should relate to the comment (reverse)
  • Symmetric interaction tensor between both modalities

Cross-validation strategy:
  • StratifiedKFold(n_splits=5) over training data
  • Oversampling applied ONLY to each fold's training split (never the val split)
  • Class weights recomputed per fold from that fold's raw training distribution
  • Fresh model loaded at the start of every fold
  • Best model across all folds (highest val macro-F1) is saved as the final model
  • Mean ± std reported across all folds at the end
  • Bidirectional cross-attention weights inspected on 5 val samples per fold
  • Out-of-fold (OOF) report generated from predictions on all training data

Expected CSV columns (via BODY_COL / COMMENT_COL / LABEL_COL config):
    body, comment_phrase, comment_sentiment
"""

import os
import traceback
import math
import random as stdlib_random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from transformers import (
    BertConfig, BertModel, BertForMaskedLM,
    Trainer, TrainingArguments, EvalPrediction,
    EarlyStoppingCallback,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import random
import wandb

# ==================== CONFIGURATION ====================
print("=" * 80)
print("BERT SENTIMENT — STAGE 3: BIDIRECTIONAL CROSS-ATTENTION  [5-FOLD CROSS-VALIDATION]")
print("=" * 80)

# ==================== File paths ====================
BERT_MODEL_PATH  = "HelaBERT"
TOKENIZER_MODEL  = "tokenizer/unigram_32000_0.9995.model"
BERT_CONFIG_FILE = "HelaBERT/config.json"

DATA_PATH        = "data/sinhala-sentiment-analysis/outputs/train.csv"

# ==================== TSV column names ====================
BODY_COL    = "body"
COMMENT_COL = "comment_phrase"
LABEL_COL   = "comment_sentiment"

# ==================== Sliding window ====================
CHUNK_SIZE   = 512    # tokens per chunk
CHUNK_STRIDE = 256    # 50% overlap
MAX_CHUNKS   = 16     # cap per article — reduce to 8 if OOM

# ==================== Sequence lengths ====================
COMMENT_MAX_LENGTH = 256

# ==================== Bidirectional Cross-attention ====================
CROSS_ATTN_HEADS   = 8     # must divide hidden_size (768/8 = 96 per head)
CROSS_ATTN_DROPOUT = 0.1

# ==================== Training ====================
TRAIN_BATCH_SIZE            = 4     # lower: MAX_CHUNKS+1 BERT passes per sample
EVAL_BATCH_SIZE             = 8
LEARNING_RATE               = 3e-5
NUM_EPOCHS                  = 3    # early stopping decides actual stop point
WARMUP_RATIO                = 0.1
WEIGHT_DECAY                = 0.05
GRADIENT_ACCUMULATION_STEPS = 4     # effective batch = 16
EARLY_STOPPING_PATIENCE     = 3

# ==================== Cross-validation ====================
N_FOLDS = 5

# ==================== Balancing ====================
OVERSAMPLE_TRAIN  = True
USE_CLASS_WEIGHTS = True

# ==================== Output ====================
OUTPUT_DIR     = "HelaBERT_sentiment_bidirectional_cv"
BEST_MODEL_DIR = f"{OUTPUT_DIR}/best_model"
STAGE_TAG      = "bidirectional_cross_attention"

# ==================== Misc ====================
RANDOM_SEED = 42
USE_FP16    = True
USE_BF16    = False
NUM_WORKERS = 2

# ==================== W&B ====================
USE_WANDB     = True
WANDB_PROJECT = "bert-sentiment-analysis"
WANDB_GROUP   = f"5fold_cv_bidirectional_lr{LEARNING_RATE}_bs{TRAIN_BATCH_SIZE}"
WANDB_ENTITY  = None

print(f"\n✓ Config loaded — {N_FOLDS}-fold CV, oversampling={'on' if OVERSAMPLE_TRAIN else 'off'}, "
      f"class_weights={'on' if USE_CLASS_WEIGHTS else 'off'}")
print(f"  Model path:        {BERT_MODEL_PATH}")
print(f"  Tokenizer:         {TOKENIZER_MODEL}")
print(f"  Data:              {DATA_PATH}")
print(f"  Output directory:  {OUTPUT_DIR}")
print(f"  W&B logging:       {'Enabled' if USE_WANDB else 'Disabled'}")
print(f"  Architecture:      shared BERT + bidirectional {CROSS_ATTN_HEADS}-head cross-attention + interaction fusion")
print(f"  Chunk size/stride: {CHUNK_SIZE}/{CHUNK_STRIDE}  max chunks: {MAX_CHUNKS}")
print(f"  Effective batch:   {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")


# ==================== SEEDS ====================
random.seed(RANDOM_SEED)
stdlib_random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
print("\n✓ Random seeds set for reproducibility")


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
    print("CPU only — sliding window will be slow")


# ==================== VERIFY PATHS ====================
print("\n" + "=" * 80)
print("VERIFYING PATHS")
print("=" * 80)
assert os.path.exists(BERT_MODEL_PATH), f"{BERT_MODEL_PATH}"
assert os.path.exists(TOKENIZER_MODEL), f"{TOKENIZER_MODEL}"
assert os.path.exists(DATA_PATH),       f"{DATA_PATH}"
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
    sep = '\t' if path.endswith('.tsv') else ','
    try:
        df = pd.read_csv(path, sep=sep)
    except pd.errors.ParserError:
        df = pd.read_csv(path, sep=sep, engine='python', on_bad_lines='skip')
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

    dummy_ids  = torch.full((chunk_size,), PAD_ID, dtype=torch.long)
    dummy_mask = torch.zeros(chunk_size,           dtype=torch.long)
    while len(chunks) < max_chunks:
        chunks.append((dummy_ids.clone(), dummy_mask.clone()))

    return chunks, num_real


# ==================== LOAD DATA ====================
print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)
df = load_tsv(DATA_PATH)
print(f"✓ Loaded: {len(df):,} samples")


# ==================== ENCODE LABELS ====================
print("\n" + "=" * 80)
print("ENCODING LABELS")
print("=" * 80)
all_labels_raw = df['label'].unique()
le = LabelEncoder()
le.fit(sorted(all_labels_raw))
df['label_id'] = le.transform(df['label'])
NUM_LABELS  = len(le.classes_)
id_to_label = {i: lbl for i, lbl in enumerate(le.classes_)}

mapping_df = pd.DataFrame({'label_id': list(id_to_label.keys()),
                            'label_name': list(id_to_label.values())})
mapping_df.to_csv(f"{OUTPUT_DIR}/label_mapping.csv", index=False)
print(f"✓ {NUM_LABELS} labels: {', '.join(le.classes_)}")
for idx, lbl in sorted(id_to_label.items()):
    cnt = (df['label_id'] == idx).sum()
    print(f"  [{idx}] {lbl:20s}: {cnt:5d}")


# ==================== BUILD CV POOL ====================
print("\n" + "=" * 80)
print("BUILDING CV POOL (training data only)")
print("=" * 80)
full_df = df.copy()
print(f"✓ CV pool: {len(full_df):,} samples")
print("Dataset label distribution:")
for idx, cnt in sorted(Counter(full_df['label_id'].tolist()).items()):
    print(f"  [{idx}] {id_to_label[idx]:20s}: {cnt:6d} ({100*cnt/len(full_df):.1f}%)")

all_texts  = full_df['comment'].tolist()
all_bodies = full_df['body'].tolist()
all_labels = full_df['label_id'].tolist()


# ==================== BODY LENGTH ANALYSIS ====================
print("\n" + "=" * 80)
print("ARTICLE BODY LENGTH ANALYSIS")
print("=" * 80)
lengths = full_df['body'].apply(lambda x: len(sp.encode(x)) if x else 0)
has_bodies = (lengths > 0).sum()
print(f"Samples with body text: {has_bodies:,} / {len(full_df):,}")
if has_bodies > 0:
    bl = lengths[lengths > 0]
    print(f"min    : {bl.min():,}")
    print(f"mean   : {bl.mean():.0f}")
    print(f"median : {bl.median():.0f}")
    print(f"90th % : {bl.quantile(0.90):.0f}")
    print(f"max    : {bl.max():,}")
    avg_c = bl.apply(
        lambda l: min(math.ceil(max(l - CHUNK_SIZE, 0) / CHUNK_STRIDE) + 1, MAX_CHUNKS)
    ).mean()
    print(f"\nAvg chunks/article (capped {MAX_CHUNKS}): {avg_c:.1f}")
    print(f"Overlap per boundary: {CHUNK_SIZE - CHUNK_STRIDE} tokens")


# ==================== DATASET CLASS ====================
class BidirectionalAttnDataset(Dataset):
    """
    Each sample exposes:
      chunk_ids    [MAX_CHUNKS, CHUNK_SIZE]
      chunk_mask   [MAX_CHUNKS, CHUNK_SIZE]
      num_chunks   scalar
      comment_ids  [COMMENT_MAX_LENGTH]
      comment_mask [COMMENT_MAX_LENGTH]
      labels       scalar
    """
    def __init__(self, texts, bodies, labels):
        self.texts  = texts
        self.bodies = bodies
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        chunks, num_real = tokenize_chunks(
            self.bodies[idx], CHUNK_SIZE, CHUNK_STRIDE, MAX_CHUNKS
        )
        chunk_ids  = torch.stack([c[0] for c in chunks])
        chunk_mask = torch.stack([c[1] for c in chunks])

        c_ids  = sp.encode(str(self.texts[idx]))[:COMMENT_MAX_LENGTH]
        c_mask = [1] * len(c_ids)
        pad    = COMMENT_MAX_LENGTH - len(c_ids)
        c_ids  += [PAD_ID] * pad
        c_mask += [0]      * pad

        return {
            'chunk_ids':    chunk_ids,
            'chunk_mask':   chunk_mask,
            'num_chunks':   torch.tensor(num_real,              dtype=torch.long),
            'comment_ids':  torch.tensor(c_ids,                dtype=torch.long),
            'comment_mask': torch.tensor(c_mask,               dtype=torch.long),
            'labels':       torch.tensor(self.labels[idx],     dtype=torch.long),
        }


# ==================== BALANCING HELPERS ====================
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


# ==================== BIDIRECTIONAL CROSS-ATTENTION MODULE ====================
class MultiHeadCrossAttention(nn.Module):
    """
    Flexible multi-head cross-attention.
    Can attend in either direction:
      Query  : query_vecs  [B, L_q, H]
      Key/Val: context_vecs [B, L_c, H]
      Output : attended    [B, L_q, H]

    The key_padding_mask masks the context (Key/Value) side.
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
        query            : [B, L_q, H]
        context          : [B, L_c, H]
        key_padding_mask : [B, L_c] — True = ignore (dummy chunk)
        Returns          : [B, L_q, H]
        """
        B = query.shape[0]
        L_q = query.shape[1]
        L_c = context.shape[1]
        h = self.num_heads
        d = self.head_dim

        def proj_and_split(linear, x):
            # x: [B, L, H] -> [B, L, H] (linear) -> [B, L, h, d] (view) -> [B, h, L, d] (transpose)
            return linear(x).view(B, -1, h, d).transpose(1, 2)

        Q = proj_and_split(self.q_proj, query)      # [B, h, L_q, d]
        K = proj_and_split(self.k_proj, context)    # [B, h, L_c, d]
        V = proj_and_split(self.v_proj, context)    # [B, h, L_c, d]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, h, L_q, L_c]

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        weights  = F.softmax(scores, dim=-1)
        weights  = torch.nan_to_num(weights, nan=0.0)
        weights  = self.dropout(weights)
        attended = torch.matmul(weights, V)                          # [B, h, L_q, d]
        attended = attended.transpose(1, 2).contiguous().view(B, L_q, -1) # [B, L_q, H]
        return self.out_proj(attended)


# ==================== FULL BIDIRECTIONAL MODEL ====================
class BidirectionalCrossAttnSentimentModel(nn.Module):
    """
    Shared BERT + bidirectional multi-head cross-attention + interaction fusion.

    Forward pass:  comment → chunks
    Reverse pass:  chunks → comment (aggregated via mean pooling)

    Fusion vector = [comment_vec ; 
                      attended_ctx_fwd ; attended_comment_bwd ;
                      comment_vec ⊙ attended_ctx_fwd ; comment_vec ⊙ attended_comment_bwd]
    """

    def __init__(self, bert, hidden_size, num_labels, num_heads, attn_dropout):
        super().__init__()
        self.bert              = bert
        self.hidden_size       = hidden_size
        
        # Bidirectional attention modules
        self.cross_attn_fwd    = MultiHeadCrossAttention(hidden_size, num_heads, attn_dropout)
        self.cross_attn_bwd    = MultiHeadCrossAttention(hidden_size, num_heads, attn_dropout)
        
        # Fusion: comment (H) + fwd_ctx (H) + bwd_comment (H) + interaction1 (H) + interaction2 (H) = 5*H
        fusion_dim = hidden_size * 5
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.dropout     = nn.Dropout(0.1)
        self.classifier  = nn.Linear(fusion_dim, num_labels)

    def encode_chunks(self, chunk_ids, chunk_mask, num_chunks):
        B, C, L = chunk_ids.shape
        out      = self.bert(
            input_ids=chunk_ids.view(B * C, L),
            attention_mask=chunk_mask.view(B * C, L)
        )
        cls_vecs = out.last_hidden_state[:, 0, :].view(B, C, -1)

        idx_range      = torch.arange(C, device=chunk_ids.device).unsqueeze(0)
        chunk_pad_mask = idx_range >= num_chunks.unsqueeze(1)
        return cls_vecs, chunk_pad_mask

    def encode_comment(self, comment_ids, comment_mask):
        out = self.bert(input_ids=comment_ids, attention_mask=comment_mask)
        return out.last_hidden_state[:, 0, :]

    def forward(self, chunk_ids, chunk_mask, num_chunks,
                comment_ids, comment_mask, labels=None):

        # Encode chunks and comment
        cls_vecs, pad_mask = self.encode_chunks(chunk_ids, chunk_mask, num_chunks)
        comment_vec        = self.encode_comment(comment_ids, comment_mask)

        # ── FORWARD: comment queries chunks ───────────────────────────────────
        attended_ctx_fwd = self.cross_attn_fwd(
            query=comment_vec.unsqueeze(1),    # [B, 1, H]
            context=cls_vecs,                  # [B, C, H]
            key_padding_mask=pad_mask
        ).squeeze(1)  # [B, H]

        # ── REVERSE: chunks query comment ─────────────────────────────────────
        attended_comment_all = self.cross_attn_bwd(
            query=cls_vecs,                    # [B, C, H]
            context=comment_vec.unsqueeze(1),  # [B, 1, H]
            key_padding_mask=None              # no mask needed for comment (single token)
        )  # [B, C, H]
        
        # Aggregate the attended comment representations across chunks (mean pooling)
        # Mask out dummy chunks before aggregation
        mask_for_agg = (~pad_mask).float()  # [B, C], 1 for real, 0 for dummy
        attended_comment_bwd = (attended_comment_all * mask_for_agg.unsqueeze(-1)).sum(dim=1) / \
                               mask_for_agg.sum(dim=1, keepdim=True).clamp(min=1)  # [B, H]

        # ── Fusion ──────────────────────────────────────────────────────────────
        fusion = torch.cat([
            comment_vec,                                    # [B, H]
            attended_ctx_fwd,                               # [B, H]
            attended_comment_bwd,                           # [B, H]
            comment_vec * attended_ctx_fwd,                 # [B, H]
            comment_vec * attended_comment_bwd,             # [B, H]
        ], dim=-1)  # [B, 5*H]

        logits = self.classifier(self.dropout(self.fusion_norm(fusion)))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


# ==================== COLLATOR ====================
def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ==================== CUSTOM TRAINER (supports class weights) ====================
class BidirectionalAttnTrainer(Trainer):
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
            pin_memory=torch.cuda.is_available()
        )

    def get_train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True)

    def get_eval_dataloader(self, eval_dataset=None):
        return self._make_loader(eval_dataset or self.eval_dataset, shuffle=False)

    def get_test_dataloader(self, test_dataset):
        return self._make_loader(test_dataset, shuffle=False)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        if self.class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
            loss    = loss_fn(outputs.logits, labels)
        else:
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


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


# ==================== FRESH MODEL LOADER ====================
def load_fresh_model():
    """Load a fresh shared-BERT bidirectional cross-attention model for each fold."""
    if os.path.exists(BERT_CONFIG_FILE):
        cfg = BertConfig.from_json_file(BERT_CONFIG_FILE)
        print(f"  ✓ Config from {BERT_CONFIG_FILE}")
    else:
        try:
            cfg = BertConfig.from_pretrained(BERT_MODEL_PATH)
            print("  ✓ Config from model dir")
        except Exception:
            cfg = None
            print("  ⚠️  Config not found — using defaults")

    try:
        backbone = BertModel.from_pretrained(BERT_MODEL_PATH)
        print("  ✓ Weights loaded via BertModel")
    except Exception as e:
        print(f"  ⚠️  BertModel failed ({e}), trying MLM checkpoint...")
        mlm      = BertForMaskedLM.from_pretrained(BERT_MODEL_PATH)
        backbone = mlm.bert
        if cfg is None:
            cfg = mlm.config
        print("  ✓ Weights loaded via BertForMaskedLM")

    hs = cfg.hidden_size if cfg else backbone.config.hidden_size

    assert hs % CROSS_ATTN_HEADS == 0, (
        f"CROSS_ATTN_HEADS ({CROSS_ATTN_HEADS}) must divide hidden_size ({hs}). "
        f"Valid choices: {[h for h in [1,2,4,8,12,16] if hs % h == 0]}"
    )

    m = BidirectionalCrossAttnSentimentModel(
        bert=backbone,
        hidden_size=hs,
        num_labels=NUM_LABELS,
        num_heads=CROSS_ATTN_HEADS,
        attn_dropout=CROSS_ATTN_DROPOUT,
    )
    return m, cfg, hs


# ==================== LOAD MODEL ONCE (to get hidden_size for W&B config) ====================
print("\n" + "=" * 80)
print("PROBING MODEL ARCHITECTURE")
print("=" * 80)
_probe_model, bert_config, hidden_size = load_fresh_model()
extra_params = sum(p.numel() for p in
                   list(_probe_model.cross_attn_fwd.parameters()) +
                   list(_probe_model.cross_attn_bwd.parameters()) +
                   list(_probe_model.fusion_norm.parameters()) +
                   list(_probe_model.classifier.parameters()))
total_params     = sum(p.numel() for p in _probe_model.parameters())
trainable_params = sum(p.numel() for p in _probe_model.parameters() if p.requires_grad)
print(f"\nTotal params      : {total_params:,}")
print(f"Trainable params  : {trainable_params:,}  ({100*trainable_params/total_params:.1f}%)")
print(f"Extra params      : {extra_params:,}  (bidirectional cross-attn + fusion norm + classifier)")
print(f"Cross-attn heads  : {CROSS_ATTN_HEADS}  head dim: {hidden_size // CROSS_ATTN_HEADS}")
print(f"Fusion input dim  : {hidden_size * 5}  (comment + fwd_ctx + bwd_comment + interaction1 + interaction2)")
del _probe_model   # free memory before fold loop


# ==================== CV STATE ====================
skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
fold_metrics = []
all_y_true   = []
all_y_pred   = []
all_oof_texts = []
best_fold_f1  = -1.0
best_fold_idx = -1
wandb_group_url = None


# ==================== FOLD LOOP ====================
for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(all_texts, all_labels), start=1):

    print("\n" + "=" * 80)
    print(f"FOLD {fold_idx} / {N_FOLDS}")
    print("=" * 80)

    fold_output_dir = f"{OUTPUT_DIR}/fold_{fold_idx}"
    os.makedirs(fold_output_dir, exist_ok=True)

    # ── Split ───────────────────────────────────────────────────────────────
    fold_train_texts  = [all_texts[i]  for i in train_idx]
    fold_train_bodies = [all_bodies[i] for i in train_idx]
    fold_train_labels = [all_labels[i] for i in train_idx]

    fold_val_texts    = [all_texts[i]  for i in val_idx]
    fold_val_bodies   = [all_bodies[i] for i in val_idx]
    fold_val_labels   = [all_labels[i] for i in val_idx]

    print(f"Train: {len(fold_train_texts)} samples  |  Val: {len(fold_val_texts)} samples")

    print("Train label distribution (before oversampling):")
    for lbl, cnt in sorted(Counter(fold_train_labels).items()):
        print(f"  [{lbl:2d}] {id_to_label[lbl]:20s}: {cnt}")

    # ── Oversample (train split only) ───────────────────────────────────────
    if OVERSAMPLE_TRAIN:
        fold_train_texts, fold_train_bodies, fold_train_labels = oversample(
            fold_train_texts, fold_train_bodies, fold_train_labels,
            seed=RANDOM_SEED + fold_idx
        )
        print(f"After oversampling: {len(fold_train_texts)} train samples")
        print("Train label distribution (after oversampling):")
        for lbl, cnt in sorted(Counter(fold_train_labels).items()):
            print(f"  [{lbl:2d}] {id_to_label[lbl]:20s}: {cnt}")

    # ── Class weights (from raw pre-oversample fold distribution) ───────────
    raw_fold_labels = [all_labels[i] for i in train_idx]
    fold_weights    = compute_class_weights(raw_fold_labels, NUM_LABELS)
    if USE_CLASS_WEIGHTS:
        print("Class weights:")
        for i, w in enumerate(fold_weights):
            print(f"  [{i:2d}] {id_to_label[i]:20s}: {w.item():.4f}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = BidirectionalAttnDataset(fold_train_texts, fold_train_bodies, fold_train_labels)
    val_ds   = BidirectionalAttnDataset(fold_val_texts,   fold_val_bodies,   fold_val_labels)

    # ── Fresh model ──────────────────────────────────────────────────────────
    print(f"Loading fresh model for fold {fold_idx}...")
    model, bert_config, hidden_size = load_fresh_model()
    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_p:,} total, {trainable_p:,} trainable")

    # ── W&B (one run per fold, all in the same group) ────────────────────────
    fold_wandb_name = f"fold_{fold_idx}"
    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            group=WANDB_GROUP,
            name=fold_wandb_name,
            config={
                'fold':               fold_idx,
                'n_folds':            N_FOLDS,
                'train_samples':      len(fold_train_texts),
                'val_samples':        len(fold_val_texts),
                'architecture':       'bidirectional_cross_attention',
                'cross_attn_heads':   CROSS_ATTN_HEADS,
                'cross_attn_dropout': CROSS_ATTN_DROPOUT,
                'hidden_size':        hidden_size,
                'fusion_dim':         hidden_size * 5,
                'learning_rate':      LEARNING_RATE,
                'batch_size':         TRAIN_BATCH_SIZE,
                'grad_accum_steps':   GRADIENT_ACCUMULATION_STEPS,
            },
            reinit=True,
        )
        wandb_group_url = f"https://wandb.ai/{wandb.config.get('entity', 'user')}/{WANDB_PROJECT}/groups/{WANDB_GROUP}"

    # ── Training args ────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        logging_steps=max(1, len(train_ds) // (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * 4)),
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        report_to=['wandb'] if USE_WANDB else [],
        remove_unused_columns=False,
        fp16=USE_FP16 and torch.cuda.is_available(),
        bf16=USE_BF16,
        optim='adamw_8bit' if torch.cuda.is_available() else 'adamw_torch',
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = BidirectionalAttnTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=0.0,
            )
        ],
        class_weights=fold_weights if USE_CLASS_WEIGHTS else None,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\nTraining fold {fold_idx}...")
    try:
        train_result = trainer.train()
        print(f"✓ Training complete")
        print(f"  loss: {train_result.training_loss:.4f}")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        traceback.print_exc()
        if USE_WANDB:
            wandb.finish()
        continue

    # ── Evaluation ───────────────────────────────────────────────────────────
    print(f"\nEvaluating fold {fold_idx}...")
    eval_result = trainer.evaluate()
    print(f"✓ Evaluation metrics:")
    for metric, value in eval_result.items():
        print(f"  {metric:<30} {value:.4f}")

    val_f1 = eval_result.get('eval_f1', -1.0)
    if val_f1 > best_fold_f1:
        best_fold_f1  = val_f1
        best_fold_idx = fold_idx
        print(f"  → New best fold! (F1 = {best_fold_f1:.4f})")
        os.makedirs(BEST_MODEL_DIR, exist_ok=True)
        trainer.save_model(BEST_MODEL_DIR)
        arch_cfg = pd.DataFrame({
            'param': ['stage', 'architecture', 'hidden_size', 'num_labels', 'cross_attn_heads',
                      'fusion_dim', 'chunk_size', 'chunk_stride', 'max_chunks', 'comment_max_length'],
            'value': [STAGE_TAG, 'bidirectional_cross_attention', hidden_size, NUM_LABELS,
                      CROSS_ATTN_HEADS, hidden_size * 5, CHUNK_SIZE, CHUNK_STRIDE, MAX_CHUNKS, COMMENT_MAX_LENGTH]
        })
        arch_cfg.to_csv(f"{BEST_MODEL_DIR}/arch_config.csv", index=False)
        mapping_df.to_csv(f"{BEST_MODEL_DIR}/label_mapping.csv", index=False)
        print(f"  → Saved to {BEST_MODEL_DIR}")

    fold_metrics.append({
        'fold': fold_idx,
        'accuracy':      eval_result.get('eval_accuracy', 0.0),
        'precision':     eval_result.get('eval_precision', 0.0),
        'recall':        eval_result.get('eval_recall', 0.0),
        'f1':            eval_result.get('eval_f1', 0.0),
        'f1_weighted':   eval_result.get('eval_f1_weighted', 0.0),
    })

    # ── Predictions (for OOF) ────────────────────────────────────────────────
    preds_result = trainer.predict(val_ds)
    val_preds = np.argmax(preds_result.predictions, axis=1)
    all_y_true.extend(fold_val_labels)
    all_y_pred.extend(val_preds)
    all_oof_texts.extend(fold_val_texts)

    # ── Validation sample cross-attention analysis ───────────────────────────
    print(f"\nAnalyzing bidirectional cross-attention weights...")
    model.eval()
    sample_count = min(5, len(val_ds))
    attn_rows = []

    with torch.no_grad():
        for si in range(sample_count):
            sample = val_ds[si]
            for k in sample:
                sample[k] = sample[k].unsqueeze(0)
            if torch.cuda.is_available():
                sample = {k: v.cuda() for k, v in sample.items()}

            chunk_ids    = sample['chunk_ids']
            chunk_mask   = sample['chunk_mask']
            num_chunks   = sample['num_chunks']
            comment_ids  = sample['comment_ids']
            comment_mask = sample['comment_mask']

            B, C, L = chunk_ids.shape
            out_chunks = model.bert(
                input_ids=chunk_ids.view(B * C, L),
                attention_mask=chunk_mask.view(B * C, L)
            )
            cls_vecs = out_chunks.last_hidden_state[:, 0, :].view(B, C, -1)

            out_comment = model.bert(input_ids=comment_ids, attention_mask=comment_mask)
            comment_vec = out_comment.last_hidden_state[:, 0, :]

            idx_range    = torch.arange(C, device=chunk_ids.device).unsqueeze(0)
            pad_mask     = idx_range >= num_chunks.unsqueeze(1)
            n_real       = num_chunks[0].item()

            # Forward attention (comment → chunks)
            ca = model.cross_attn_fwd
            h = ca.num_heads
            d = ca.head_dim
            Q = ca.q_proj(comment_vec.unsqueeze(1)).view(B, 1, h, d).transpose(1, 2)
            K = ca.k_proj(cls_vecs).view(B, C, h, d).transpose(1, 2)
            raw_fwd = torch.matmul(Q, K.transpose(-2, -1)) / ca.scale
            raw_fwd = raw_fwd.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            w_fwd = F.softmax(raw_fwd, dim=-1).squeeze()
            chunk_importance_fwd = (w_fwd.mean(dim=0) if w_fwd.dim() == 2 else w_fwd)[:n_real].cpu().numpy()

            # Reverse attention (chunks → comment)
            ca_bwd = model.cross_attn_bwd
            Q_bwd = ca_bwd.q_proj(cls_vecs).view(B, C, h, d).transpose(1, 2)
            K_bwd = ca_bwd.k_proj(comment_vec.unsqueeze(1)).view(B, 1, h, d).transpose(1, 2)
            raw_bwd = torch.matmul(Q_bwd, K_bwd.transpose(-2, -1)) / ca_bwd.scale
            w_bwd = F.softmax(raw_bwd, dim=-1).squeeze()
            chunk_importance_bwd = (w_bwd.mean(dim=0) if w_bwd.dim() == 2 else w_bwd)[:n_real].cpu().numpy()

            comment_text = fold_val_texts[si][:80]
            true_lbl     = id_to_label[fold_val_labels[si]]
            pred_lbl     = id_to_label[int(all_y_pred[-(len(val_ds) - si)])]
            correct_sym  = "✓" if fold_val_labels[si] == int(all_y_pred[-(len(val_ds) - si)]) else "✗"

            print(f"\nSample {si+1} [{correct_sym}]  True: {true_lbl}  Pred: {pred_lbl}")
            print(f"Comment: {comment_text}...")
            print(f"Forward (comment → chunks):")
            for ci, cw in enumerate(chunk_importance_fwd):
                bar = "█" * int(cw * 40)
                print(f"  chunk {ci:2d}: {cw:.4f}  {bar}")
            print(f"Reverse (chunks → comment):")
            for ci, cw in enumerate(chunk_importance_bwd):
                bar = "█" * int(cw * 40)
                print(f"  chunk {ci:2d}: {cw:.4f}  {bar}")

            for ci in range(n_real):
                attn_rows.append({
                    'fold': fold_idx, 'sample_idx': si,
                    'comment': comment_text,
                    'true_sentiment': true_lbl, 'pred_sentiment': pred_lbl,
                    'chunk_idx': ci,
                    'attn_weight_fwd': float(chunk_importance_fwd[ci]),
                    'attn_weight_bwd': float(chunk_importance_bwd[ci]),
                })

    if attn_rows:
        attn_df = pd.DataFrame(attn_rows)
        attn_df.to_csv(f"{fold_output_dir}/bidirectional_attn_weights.csv", index=False)
        if USE_WANDB:
            wandb.log({"bidirectional_attention_weights": wandb.Table(dataframe=attn_df)})
        print(f"✓ Bidirectional attention weights saved → {fold_output_dir}/bidirectional_attn_weights.csv")

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

oof_y_true   = np.array(all_y_true)
oof_y_pred   = np.array(all_y_pred)
target_names = [id_to_label[i] for i in range(NUM_LABELS)]
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
    index=[f"True_{id_to_label[i]}"  for i in range(NUM_LABELS)],
    columns=[f"Pred_{id_to_label[i]}" for i in range(NUM_LABELS)],
).to_csv(f"{OUTPUT_DIR}/oof_confusion_matrix.csv")

pd.DataFrame({
    'comment':             all_oof_texts,
    'true_label_id':       oof_y_true,
    'true_sentiment':      [id_to_label[l] for l in oof_y_true],
    'predicted_label_id':  oof_y_pred,
    'predicted_sentiment': [id_to_label[l] for l in oof_y_pred],
    'correct':             oof_y_true == oof_y_pred,
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
            x=[f"Pred_{id_to_label[i]}" for i in range(NUM_LABELS)],
            y=[f"True_{id_to_label[i]}" for i in range(NUM_LABELS)],
            colorscale='Blues',
            showscale=True,
        )
        fig.update_layout(
            title="OOF Confusion Matrix — All Folds",
            xaxis_title="Predicted Sentiment",
            yaxis_title="True Sentiment",
        )
        wandb.log({"oof_confusion_matrix": fig})
    except ImportError:
        pass
    wandb.finish()
    print(f"✓ CV summary logged — group: {WANDB_GROUP}")


# ==================== FINAL SUMMARY ====================
cv_summary = {
    'Model':                'BERT with SentencePiece',
    'Architecture':         'Shared BERT + Bidirectional Multi-Head Cross-Attention + Interaction Fusion',
    'Stage':                STAGE_TAG,
    'Task':                 'Sentiment Analysis',
    'Balancing':            'Oversample + Class Weights',
    'N Folds':              N_FOLDS,
    'Num Classes':          NUM_LABELS,
    'Classes':              ', '.join(le.classes_),
    'Total Samples':        len(full_df),
    'Cross-Attn Heads':     CROSS_ATTN_HEADS,
    'Chunk Size/Stride':    f'{CHUNK_SIZE}/{CHUNK_STRIDE}',
    'Max Chunks':           MAX_CHUNKS,
    'Comment Max Length':   COMMENT_MAX_LENGTH,
    'Fusion Dim':           f'{hidden_size * 5}  (5 components)',
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
print("5-FOLD CROSS-VALIDATION (BIDIRECTIONAL CONTEXT-AWARE CROSS-ATTENTION) COMPLETE!")
print("=" * 80)
print(f"\nOutputs saved to: {OUTPUT_DIR}/")
print(f"fold_1/ … fold_{N_FOLDS}/      per-fold predictions, metrics, confusion matrix,")
print(f"                               bidirectional_attn_weights.csv")
print(f"best_model/                    best model weights (fold {best_fold_idx}) + label_mapping.csv + arch_config.csv")
print(f"label_mapping.csv              label id ↔ sentiment name")
print(f"cv_fold_metrics.csv            per-fold metric table")
print(f"cv_summary.csv                 overall CV summary")
print(f"oof_predictions.csv            out-of-fold predictions (full dataset)")
print(f"oof_confusion_matrix.csv       OOF confusion matrix")
if USE_WANDB and wandb_group_url:
    print(f"\nW&B group: {wandb_group_url}")