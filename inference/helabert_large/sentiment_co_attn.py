"""
HelaBERT_large Inference — Sentiment Analysis (Co-Attention)
Loads the best-performing run from HelaBERT_large_coattention_sentiment,
evaluates on the held-out test set (outputs/test.csv),
prints metrics, and saves a summary CSV.

Task:        3-class sentiment (NEGATIVE, NEUTRAL, POSITIVE)
Test data:   data/sinhala-sentiment-analysis/outputs/test.csv
Text col:    comment_phrase   |  Label col: comment_sentiment
Base model:  HelaBERT_large (hidden_size=768)
Head:        CoAttention([CLS] ↔ token_seq) → LayerNorm → concat → MLP
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report,
)
from transformers import BertConfig, BertModel
from safetensors.torch import load_file

# ==================== CONFIGURATION ====================
TOKENIZER_MODEL  = "tokenizer/unigram_32000_0.9995.model"
BERT_CONFIG_FILE = "HelaBERT_large/config.json"
MODEL_DIR        = "models/new/large/HelaBERT_large_coattention_sentiment"
TEST_DATA_PATH   = "data/sinhala-sentiment-analysis/outputs/test.csv"

MAX_SEQ_LENGTH = 512
BATCH_SIZE     = 32
DROPOUT        = 0.1

OUTPUT_DIR = "results_test/HelaBERT_large_coattention_sentiment"

print("=" * 80)
print("HelaBERT_large INFERENCE — SENTIMENT ANALYSIS (CO-ATTENTION)")
print("=" * 80)


# ==================== PICK BEST RUN ====================
results_df = pd.read_csv(f"{MODEL_DIR}/results.csv")
best_idx   = results_df['macro_f1'].idxmax()
best_run   = int(results_df.loc[best_idx, 'run'])
best_f1    = results_df.loc[best_idx, 'macro_f1']
print(f"Best run: run_{best_run}  (train macro-F1 = {best_f1:.4f})")

run_dir = f"{MODEL_DIR}/run_{best_run}"
checkpoints = [d for d in os.listdir(run_dir) if d.startswith("checkpoint")]
assert checkpoints, f"No checkpoint found in {run_dir}"
checkpoint_dir = os.path.join(run_dir, sorted(checkpoints)[-1])
print(f"Checkpoint: {checkpoint_dir}")


# ==================== LABEL MAP ====================
label_map_df = pd.read_csv(f"{MODEL_DIR}/label_map.csv")
id_to_label  = dict(zip(label_map_df['id'], label_map_df['label']))
label_to_id  = {v: k for k, v in id_to_label.items()}
num_labels   = len(id_to_label)
print(f"Labels ({num_labels}): {id_to_label}")


# ==================== TOKENIZER ====================
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)
PAD_ID = sp.pad_id()
print(f"Tokenizer loaded — vocab: {sp.get_piece_size()}")


# ==================== MODEL ARCHITECTURE ====================
class CoAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W_cls   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_token = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v       = nn.Linear(hidden_size, 1,           bias=False)
        self.scale   = hidden_size ** 0.5

    @staticmethod
    def softmax_safe(logits, mask=None):
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e4)
        return F.softmax(logits, dim=-1)

    def forward(self, cls_vec, token_seq, key_mask=None):
        B, T, H = token_seq.shape

        cls_proj   = self.W_cls(cls_vec).unsqueeze(1).expand(-1, T, -1)
        token_proj = self.W_token(token_seq)
        affinity   = self.v(torch.tanh(cls_proj + token_proj)).squeeze(-1) / self.scale

        alpha        = self.softmax_safe(affinity, key_mask)
        attended_cls = torch.bmm(alpha.unsqueeze(1), token_seq).squeeze(1)

        beta = torch.sigmoid(affinity)
        if key_mask is not None:
            beta = beta * key_mask.float()
        beta_norm       = beta / (beta.sum(dim=-1, keepdim=True) + 1e-9)
        attended_tokens = torch.bmm(beta_norm.unsqueeze(1), token_seq).squeeze(1)

        return attended_cls, attended_tokens


class SentimentModelCoAttention(nn.Module):
    def __init__(self, bert, hidden_size, num_labels, dropout=0.1):
        super().__init__()
        self.bert        = bert
        self.co_attn     = CoAttention(hidden_size)
        self.norm_cls    = nn.LayerNorm(hidden_size)
        self.norm_tokens = nn.LayerNorm(hidden_size)
        self.dropout     = nn.Dropout(dropout)
        self.classifier  = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state

        cls_vec    = hidden[:, 0, :]
        token_seq  = hidden[:, 1:, :]
        token_mask = attention_mask[:, 1:]

        attended_cls, attended_tokens = self.co_attn(
            cls_vec, token_seq, key_mask=token_mask
        )

        attended_cls    = self.norm_cls(attended_cls)
        attended_tokens = self.norm_tokens(attended_tokens)

        combined = torch.cat([attended_cls, attended_tokens], dim=-1)
        logits   = self.classifier(self.dropout(combined))
        return logits


# ==================== LOAD MODEL ====================
cfg   = BertConfig.from_json_file(BERT_CONFIG_FILE)
bert  = BertModel(cfg)
model = SentimentModelCoAttention(bert, cfg.hidden_size, num_labels, DROPOUT)

state_dict = load_file(os.path.join(checkpoint_dir, "model.safetensors"))
model.load_state_dict(state_dict)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on {device}")


# ==================== LOAD TEST DATA ====================
df = pd.read_csv(TEST_DATA_PATH)
df.columns = df.columns.str.strip()

text_col  = next(c for c in df.columns if 'comment' in c.lower() and 'sentiment' not in c.lower())
label_col = next(c for c in df.columns if 'sentiment' in c.lower())

df = df[[text_col, label_col]].dropna()
df[text_col]  = df[text_col].astype(str).str.strip()
df[label_col] = df[label_col].astype(str).str.strip()
df = df[df[text_col].str.len() > 0].reset_index(drop=True)

df['label_id'] = df[label_col].map(label_to_id)
df = df.dropna(subset=['label_id']).reset_index(drop=True)
df['label_id'] = df['label_id'].astype(int)

texts     = df[text_col].tolist()
label_ids = df['label_id'].tolist()
print(f"\nTest samples: {len(df):,}")
for lid, lname in id_to_label.items():
    cnt = sum(1 for l in label_ids if l == lid)
    print(f"  [{lid}] {lname}: {cnt}")


# ==================== DATASET ====================
class TextDataset(Dataset):
    def __init__(self, texts, labels, sp_processor, max_length):
        self.texts      = texts
        self.labels     = labels
        self.sp         = sp_processor
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids  = self.sp.encode(str(self.texts[idx]))[:self.max_length]
        mask = [1] * len(ids)
        pad  = self.max_length - len(ids)
        ids  += [PAD_ID] * pad
        mask += [0]      * pad
        return {
            'input_ids':      torch.tensor(ids,              dtype=torch.long),
            'attention_mask': torch.tensor(mask,             dtype=torch.long),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ==================== INFERENCE ====================
dataset = TextDataset(texts, label_ids, sp, MAX_SEQ_LENGTH)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

all_preds  = []
all_labels = []

print("\nRunning inference...")
with torch.no_grad():
    for batch in loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        logits = model(input_ids, attention_mask)
        preds  = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch['label'].numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)


# ==================== METRICS ====================
label_names = [id_to_label[i] for i in range(num_labels)]

accuracy    = accuracy_score(all_labels, all_preds)
f1_macro    = f1_score(all_labels, all_preds, average='macro',    zero_division=0)
f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
precision   = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall      = recall_score(all_labels, all_preds,    average='macro', zero_division=0)

print("\n" + "=" * 80)
print("TEST RESULTS — SENTIMENT ANALYSIS (CO-ATTENTION)")
print("=" * 80)
print(f"  Accuracy:    {accuracy:.4f}")
print(f"  Macro-F1:    {f1_macro:.4f}")
print(f"  Weighted-F1: {f1_weighted:.4f}")
print(f"  Precision:   {precision:.4f}")
print(f"  Recall:      {recall:.4f}")
print()
print(classification_report(all_labels, all_preds, target_names=label_names, zero_division=0, digits=4))


# ==================== SAVE RESULTS ====================
os.makedirs(OUTPUT_DIR, exist_ok=True)

summary = pd.DataFrame([{
    'task':        'sentiment_co_attn',
    'model':       'HelaBERT_large',
    'best_run':    best_run,
    'checkpoint':  checkpoint_dir,
    'test_samples':len(all_labels),
    'accuracy':    round(accuracy,    4),
    'f1_macro':    round(f1_macro,    4),
    'f1_weighted': round(f1_weighted, 4),
    'precision':   round(precision,   4),
    'recall':      round(recall,      4),
}])
summary.to_csv(f"{OUTPUT_DIR}/results.csv", index=False)
print(f"Results saved to {OUTPUT_DIR}/results.csv")
