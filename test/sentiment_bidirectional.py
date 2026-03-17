"""
Test script for Bidirectional Sentiment Analysis Model

This script loads the best trained bidirectional model and evaluates it on a test dataset.
Two variables need to be manually configured:
  - TEST_DATA_PATH: Path to test CSV file with columns: body, comment_phrase, comment_sentiment
  - BEST_MODEL_PATH: Path to best_model directory from training output
"""

import os
import sys
import json
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
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import BertModel, BertConfig

# ==================== CONFIGURATION ====================
# *** MANUALLY SET THESE TWO VARIABLES ***
TEST_DATA_PATH = "data/sinhala-sentiment-analysis/outputs/test.csv"  # e.g., "data/test_sentiment.csv"
BEST_MODEL_PATH = "HelaBERT_sentiment_bidirectional_cv/best_model"   # e.g., "utils/finetuned_models/HelaBERT_sentiment_bidirectional_cv/best_model"

# Fixed configuration
BERT_MODEL_PATH = "HelaBERT"
TOKENIZER_MODEL = "tokenizer/unigram_32000_0.9995.model"
BERT_CONFIG_FILE = "HelaBERT/config.json"

# Sliding window configuration
CHUNK_SIZE = 512
CHUNK_STRIDE = 256
MAX_CHUNKS = 16

# Sequence lengths
COMMENT_MAX_LENGTH = 256

# Bidirectional Cross-attention
CROSS_ATTN_HEADS = 8
CROSS_ATTN_DROPOUT = 0.1

BATCH_SIZE = 8
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("SENTIMENT ANALYSIS - TEST EVALUATION (BIDIRECTIONAL)")
print("=" * 80)
print(f"\nTest data:  {TEST_DATA_PATH}")
print(f"Model path: {BEST_MODEL_PATH}")
print(f"Device:     {DEVICE}")


# ==================== SEEDS ====================
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ==================== TOKENIZER ====================
print("\n" + "=" * 80)
print("LOADING TOKENIZER")
print("=" * 80)

assert os.path.exists(TOKENIZER_MODEL), f"Tokenizer not found: {TOKENIZER_MODEL}"
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)
PAD_ID = sp.pad_id()
print(f"✓ SentencePiece loaded — vocab size: {sp.get_piece_size()}")


# ==================== HELPER FUNCTIONS ====================
def tokenize_chunks(text, chunk_size, stride, max_chunks):
    """
    Tokenize text and split into overlapping chunks.
    Returns (chunks_list, num_real_chunks).
    Each entry in chunks_list is (ids_tensor, mask_tensor) of length chunk_size.
    The list is padded with dummy (all-PAD) entries up to max_chunks.
    """
    ids = sp.encode(str(text))
    chunks = []
    start = 0
    while start < len(ids) and len(chunks) < max_chunks:
        end = min(start + chunk_size, len(ids))
        seg = ids[start:end]
        mask = [1] * len(seg)
        pad = chunk_size - len(seg)
        seg += [PAD_ID] * pad
        mask += [0] * pad
        chunks.append(
            (torch.tensor(seg, dtype=torch.long), torch.tensor(mask, dtype=torch.long))
        )
        if end == len(ids):
            break
        start += stride

    num_real = max(len(chunks), 1)

    dummy_ids = torch.full((chunk_size,), PAD_ID, dtype=torch.long)
    dummy_mask = torch.zeros(chunk_size, dtype=torch.long)
    while len(chunks) < max_chunks:
        chunks.append((dummy_ids.clone(), dummy_mask.clone()))

    return chunks, num_real


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
        self.texts = texts
        self.bodies = bodies
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        chunks, num_real = tokenize_chunks(
            self.bodies[idx], CHUNK_SIZE, CHUNK_STRIDE, MAX_CHUNKS
        )
        chunk_ids = torch.stack([c[0] for c in chunks])
        chunk_mask = torch.stack([c[1] for c in chunks])

        c_ids = sp.encode(str(self.texts[idx]))[: COMMENT_MAX_LENGTH]
        c_mask = [1] * len(c_ids)
        pad = COMMENT_MAX_LENGTH - len(c_ids)
        c_ids += [PAD_ID] * pad
        c_mask += [0] * pad

        return {
            "chunk_ids": chunk_ids,
            "chunk_mask": chunk_mask,
            "num_chunks": torch.tensor(num_real, dtype=torch.long),
            "comment_ids": torch.tensor(c_ids, dtype=torch.long),
            "comment_mask": torch.tensor(c_mask, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


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
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

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

        Q = proj_and_split(self.q_proj, query)  # [B, h, L_q, d]
        K = proj_and_split(self.k_proj, context)  # [B, h, L_c, d]
        V = proj_and_split(self.v_proj, context)  # [B, h, L_c, d]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, h, L_q, L_c]

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        weights = self.dropout(weights)
        attended = torch.matmul(weights, V)  # [B, h, L_q, d]
        attended = attended.transpose(1, 2).contiguous().view(B, L_q, -1)  # [B, L_q, H]
        return self.out_proj(attended)


# ==================== FULL BIDIRECTIONAL MODEL ====================
class BidirectionalCrossAttnSentimentModel(nn.Module):
    """
    Shared BERT + bidirectional multi-head cross-attention + interaction fusion.

    Forward pass:  comment → chunks
    Reverse pass:  chunks → comment (aggregated via mean pooling)

    Fusion vector = [comment_vec;
                      attended_ctx_fwd ; attended_comment_bwd ;
                      comment_vec ⊙ attended_ctx_fwd ; comment_vec ⊙ attended_comment_bwd]
    """

    def __init__(self, bert, hidden_size, num_labels, num_heads, attn_dropout):
        super().__init__()
        self.bert = bert
        self.hidden_size = hidden_size

        # Bidirectional attention modules
        self.cross_attn_fwd = MultiHeadCrossAttention(hidden_size, num_heads, attn_dropout)
        self.cross_attn_bwd = MultiHeadCrossAttention(hidden_size, num_heads, attn_dropout)

        # Fusion: comment (H) + fwd_ctx (H) + bwd_comment (H) + interaction1 (H) + interaction2 (H) = 5*H
        fusion_dim = hidden_size * 5
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(fusion_dim, num_labels)

    def encode_chunks(self, chunk_ids, chunk_mask, num_chunks):
        B, C, L = chunk_ids.shape
        out = self.bert(
            input_ids=chunk_ids.view(B * C, L), attention_mask=chunk_mask.view(B * C, L)
        )
        cls_vecs = out.last_hidden_state[:, 0, :].view(B, C, -1)

        idx_range = torch.arange(C, device=chunk_ids.device).unsqueeze(0)
        chunk_pad_mask = idx_range >= num_chunks.unsqueeze(1)
        return cls_vecs, chunk_pad_mask

    def encode_comment(self, comment_ids, comment_mask):
        out = self.bert(input_ids=comment_ids, attention_mask=comment_mask)
        return out.last_hidden_state[:, 0, :]

    def forward(self, chunk_ids, chunk_mask, num_chunks, comment_ids, comment_mask, labels=None):

        # Encode chunks and comment
        cls_vecs, pad_mask = self.encode_chunks(chunk_ids, chunk_mask, num_chunks)
        comment_vec = self.encode_comment(comment_ids, comment_mask)

        # ── FORWARD: comment queries chunks ───────────────────────────────────
        attended_ctx_fwd = self.cross_attn_fwd(
            query=comment_vec.unsqueeze(1),  # [B, 1, H]
            context=cls_vecs,  # [B, C, H]
            key_padding_mask=pad_mask,
        ).squeeze(1)  # [B, H]

        # ── REVERSE: chunks query comment ─────────────────────────────────────
        attended_comment_all = self.cross_attn_bwd(
            query=cls_vecs,  # [B, C, H]
            context=comment_vec.unsqueeze(1),  # [B, 1, H]
            key_padding_mask=None,  # no mask needed for comment (single token)
        )  # [B, C, H]

        # Aggregate the attended comment representations across chunks (mean pooling)
        # Mask out dummy chunks before aggregation
        mask_for_agg = (~pad_mask).float()  # [B, C], 1 for real, 0 for dummy
        attended_comment_bwd = (
            attended_comment_all * mask_for_agg.unsqueeze(-1)
        ).sum(dim=1) / mask_for_agg.sum(dim=1, keepdim=True).clamp(min=1)  # [B, H]

        # ── Fusion ──────────────────────────────────────────────────────────────
        fusion = torch.cat(
            [
                comment_vec,  # [B, H]
                attended_ctx_fwd,  # [B, H]
                attended_comment_bwd,  # [B, H]
                comment_vec * attended_ctx_fwd,  # [B, H]
                comment_vec * attended_comment_bwd,  # [B, H]
            ],
            dim=1,
        )  # [B, 5H]

        fusion = self.fusion_norm(fusion)
        fusion = self.dropout(fusion)
        logits = self.classifier(fusion)

        return logits


# ==================== DATA LOADING ====================
print("\n" + "=" * 80)
print("LOADING TEST DATA")
print("=" * 80)

assert os.path.exists(TEST_DATA_PATH), f"Test data not found: {TEST_DATA_PATH}"
df = pd.read_csv(TEST_DATA_PATH)
print(f"✓ Loaded {len(df)} test samples")

# Detect columns dynamically
body_col = None
comment_col = None
label_col = None

for col in ["body", "article_body", "article"]:
    if col in df.columns:
        body_col = col
        break

for col in ["comment_phrase", "comment", "text", "comments"]:
    if col in df.columns:
        comment_col = col
        break

for col in ["comment_sentiment", "sentiment", "label", "labels"]:
    if col in df.columns:
        label_col = col
        break

assert body_col is not None, "No body column found. Expected one of: body, article_body, article"
assert comment_col is not None, "No comment column found. Expected one of: comment_phrase, comment, text, comments"
assert label_col is not None, "No label column found. Expected one of: comment_sentiment, sentiment, label, labels"

test_bodies = df[body_col].values
test_comments = df[comment_col].values
test_labels = df[label_col].values

print(f"  Body column:     {body_col}")
print(f"  Comment column:  {comment_col}")
print(f"  Label column:    {label_col}")
print(f"  Bodies shape:    {test_bodies.shape}")
print(f"  Comments shape:  {test_comments.shape}")
print(f"  Labels shape:    {test_labels.shape}")
print(f"  Unique labels:   {np.unique(test_labels)}")
print(f"  Label distribution:\n{pd.Series(test_labels).value_counts()}")

# Create label encoder
le = LabelEncoder()
le.fit(test_labels)
NUM_LABELS = len(le.classes_)
print(f"\n  Classes: {list(le.classes_)}")
print(f"  Num labels: {NUM_LABELS}")

# Create dataset and dataloader
test_dataset = BidirectionalAttnDataset(
    test_comments,
    test_bodies,
    le.transform(test_labels),
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"\n✓ Dataset created — {len(test_loader)} batches")


# ==================== MODEL LOADING ====================
print("\n" + "=" * 80)
print("LOADING MODEL")
print("=" * 80)

# Load BERT config
assert os.path.exists(BERT_CONFIG_FILE), f"BERT config not found: {BERT_CONFIG_FILE}"
bert_config = BertConfig.from_pretrained(BERT_CONFIG_FILE)
hidden_size = bert_config.hidden_size
print(f"✓ BERT config loaded — hidden_size: {hidden_size}")

# Load BERT model
assert os.path.exists(BERT_MODEL_PATH), f"BERT model not found: {BERT_MODEL_PATH}"
bert_model = BertModel.from_pretrained(BERT_MODEL_PATH, config=bert_config)
print(f"✓ BERT model loaded")

# Create bidirectional model
model = BidirectionalCrossAttnSentimentModel(
    bert_model,
    hidden_size,
    NUM_LABELS,
    CROSS_ATTN_HEADS,
    CROSS_ATTN_DROPOUT,
)

# Load best model weights
# Try safetensors first, fall back to pytorch_model.bin
safetensors_path = f"{BEST_MODEL_PATH}/model.safetensors"
pytorch_bin_path = f"{BEST_MODEL_PATH}/pytorch_model.bin"

if os.path.exists(safetensors_path):
    from safetensors.torch import load_file

    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict)
    print(f"✓ Best model weights loaded from {safetensors_path}")
elif os.path.exists(pytorch_bin_path):
    model.load_state_dict(torch.load(pytorch_bin_path, map_location=DEVICE))
    print(f"✓ Best model weights loaded from {pytorch_bin_path}")
else:
    raise AssertionError(
        f"Best model weights not found. Expected one of:\n"
        f"  - {safetensors_path}\n"
        f"  - {pytorch_bin_path}"
    )

model.to(DEVICE)
model.eval()
print(f"✓ Model moved to {DEVICE} and set to eval mode")


# ==================== INFERENCE ====================
print("\n" + "=" * 80)
print("RUNNING INFERENCE")
print("=" * 80)

all_logits = []
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        chunk_ids = batch["chunk_ids"].to(DEVICE)
        chunk_mask = batch["chunk_mask"].to(DEVICE)
        num_chunks = batch["num_chunks"].to(DEVICE)
        comment_ids = batch["comment_ids"].to(DEVICE)
        comment_mask = batch["comment_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(chunk_ids, chunk_mask, num_chunks, comment_ids, comment_mask)
        preds = torch.argmax(logits, dim=1)

        all_logits.append(logits.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        if (batch_idx + 1) % max(1, len(test_loader) // 10) == 0:
            print(f"  Batch {batch_idx + 1}/{len(test_loader)}")

all_logits = np.concatenate(all_logits, axis=0)
all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

print(f"\n✓ Inference complete")
print(f"  Predictions shape: {all_preds.shape}")
print(f"  Logits shape: {all_logits.shape}")


# ==================== EVALUATION ====================
print("\n" + "=" * 80)
print("EVALUATION METRICS")
print("=" * 80)

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

print(f"\nOverall Metrics:")
print(f"  Accuracy:       {accuracy:.4f}")
print(f"  Precision (macro): {precision:.4f}")
print(f"  Recall (macro):    {recall:.4f}")
print(f"  F1 (macro):        {f1_macro:.4f}")
print(f"  F1 (weighted):     {f1_weighted:.4f}")

print("\n" + "-" * 80)
print("PER-CLASS CLASSIFICATION REPORT")
print("-" * 80)

target_names = list(le.classes_)
print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

print("\n" + "-" * 80)
print("CONFUSION MATRIX")
print("-" * 80)

cm = confusion_matrix(all_labels, all_preds)
cm_df = pd.DataFrame(
    cm,
    index=[f"True_{name}" for name in le.classes_],
    columns=[f"Pred_{name}" for name in le.classes_],
)
print(cm_df)


# ==================== SAVE RESULTS ====================
output_dir = f"{BEST_MODEL_PATH}/test_results"
os.makedirs(output_dir, exist_ok=True)

# Save confusion matrix
cm_df.to_csv(f"{output_dir}/confusion_matrix.csv")

# Save per-class metrics
per_class_metrics = pd.DataFrame(
    precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0),
    index=["precision", "recall", "f1", "support"],
    columns=list(le.classes_),
).T
per_class_metrics.to_csv(f"{output_dir}/per_class_metrics.csv")

# Save predictions with original labels
pred_labels = le.inverse_transform(all_preds)
true_labels = le.inverse_transform(all_labels)

predictions_df = pd.DataFrame(
    {
        "true_label": true_labels,
        "predicted_label": pred_labels,
        "correct": all_labels == all_preds,
        "confidence": np.max(
            torch.softmax(torch.tensor(all_logits), dim=1).numpy(), axis=1
        ),
    }
)
predictions_df.to_csv(f"{output_dir}/predictions.csv", index=False)

# Save summary metrics
summary = pd.DataFrame(
    {
        "metric": ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"],
        "value": [accuracy, precision, recall, f1_macro, f1_weighted],
    }
)
summary.to_csv(f"{output_dir}/test_metrics.csv", index=False)

# Save label encoder mapping
label_map = {int(idx): name for idx, name in enumerate(le.classes_)}
with open(f"{output_dir}/label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)

print(f"\n" + "=" * 80)
print("RESULTS SAVED")
print("=" * 80)
print(f"  Confusion matrix   → {output_dir}/confusion_matrix.csv")
print(f"  Per-class metrics  → {output_dir}/per_class_metrics.csv")
print(f"  Predictions        → {output_dir}/predictions.csv")
print(f"  Summary metrics    → {output_dir}/test_metrics.csv")
print(f"  Label mapping      → {output_dir}/label_map.json")

print("\n" + "=" * 80)
print("TEST EVALUATION COMPLETE!")
print("=" * 80)