"""
HelaBERT Inference — News Source Classification (Co-Attention)
Loads the best-performing run from HelaBERT_coattention_news_source,
evaluates on the held-out test set (test/news_source_test.csv),
prints metrics, and saves a summary CSV.

Task:        9-class news source
Test data:   data/Sinhala-News-Source-classification/test/news_source_test.csv
Text col:    comment   |  Label col: label
Base model:  HelaBERT_small (hidden_size=384)
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
BERT_CONFIG_FILE = "HelaBERT_small/config.json"
MODEL_DIR        = "HelaBERT_coattention_news_source"
TEST_DATA_PATH   = "data/Sinhala-News-Source-classification/test/news_source_test.csv"

MAX_SEQ_LENGTH = 512
BATCH_SIZE     = 32
DROPOUT        = 0.1

OUTPUT_DIR = "results_test/HelaBERT_small_coattention_news_source"

print("=" * 80)
print("HelaBERT INFERENCE — NEWS SOURCE CLASSIFICATION (CO-ATTENTION)")
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


class NewsSourceModelCoAttention(nn.Module):
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


# ==================== LOAD TEST DATA ====================
df = pd.read_csv(TEST_DATA_PATH)
df.columns = df.columns.str.strip()

text_col  = 'comment'
label_col = 'label'

df = df[[text_col, label_col]].dropna()
df[text_col] = df[text_col].astype(str).str.strip()
df = df[df[text_col].str.len() > 0].reset_index(drop=True)

label_ids = df[label_col].astype(int).tolist()
texts     = df[text_col].tolist()
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


dataset = TextDataset(texts, label_ids, sp, MAX_SEQ_LENGTH)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# ==================== MODEL ====================
cfg    = BertConfig.from_json_file(BERT_CONFIG_FILE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def run_inference(checkpoint_dir):
    bert  = BertModel(cfg)
    model = NewsSourceModelCoAttention(bert, cfg.hidden_size, num_labels, DROPOUT)

    state_dict = load_file(os.path.join(checkpoint_dir, "model.safetensors"))
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    preds_list  = []
    labels_list = []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids, attention_mask)
            preds  = logits.argmax(dim=-1).cpu().numpy()
            preds_list.extend(preds)
            labels_list.extend(batch['label'].numpy())
    return np.array(preds_list), np.array(labels_list)


# ==================== EVALUATE ALL 5 SEEDS ====================
label_names = [str(id_to_label[i]) for i in range(num_labels)]
metric_names = ['accuracy', 'f1_macro', 'f1_weighted', 'precision', 'recall']

per_run_records = []
best_preds, best_labels = None, None

print("\nRunning inference across all seeds...")
for _, row in results_df.iterrows():
    run_idx_i = int(row['run'])
    run_dir_i = f"{MODEL_DIR}/run_{run_idx_i}"
    checkpoints_i = [d for d in os.listdir(run_dir_i) if d.startswith("checkpoint")]
    assert checkpoints_i, f"No checkpoint found in {run_dir_i}"
    checkpoint_dir_i = os.path.join(run_dir_i, sorted(checkpoints_i)[-1])

    preds_i, labels_i = run_inference(checkpoint_dir_i)

    acc_i  = accuracy_score(labels_i, preds_i)
    f1m_i  = f1_score(labels_i, preds_i, average='macro',    zero_division=0)
    f1w_i  = f1_score(labels_i, preds_i, average='weighted', zero_division=0)
    prec_i = precision_score(labels_i, preds_i, average='macro', zero_division=0)
    rec_i  = recall_score(labels_i, preds_i,    average='macro', zero_division=0)

    per_run_records.append({
        'run': run_idx_i, 'seed': row['seed'], 'checkpoint': checkpoint_dir_i,
        'accuracy': round(acc_i, 4), 'f1_macro': round(f1m_i, 4),
        'f1_weighted': round(f1w_i, 4), 'precision': round(prec_i, 4), 'recall': round(rec_i, 4),
    })
    print(f"  run_{run_idx_i} (seed={row['seed']}): "
          f"acc={acc_i:.4f}  f1_macro={f1m_i:.4f}  f1_weighted={f1w_i:.4f}  "
          f"precision={prec_i:.4f}  recall={rec_i:.4f}")

    if run_idx_i == best_run:
        best_preds, best_labels = preds_i, labels_i
        accuracy, f1_macro, f1_weighted, precision, recall = acc_i, f1m_i, f1w_i, prec_i, rec_i
        checkpoint_dir = checkpoint_dir_i

all_preds, all_labels = best_preds, best_labels

print("\n" + "=" * 80)
print("TEST RESULTS — NEWS SOURCE CLASSIFICATION (CO-ATTENTION)")
print("=" * 80)
print(f"  Accuracy:    {accuracy:.4f}")
print(f"  Macro-F1:    {f1_macro:.4f}")
print(f"  Weighted-F1: {f1_weighted:.4f}")
print(f"  Precision:   {precision:.4f}")
print(f"  Recall:      {recall:.4f}")
print()
print(classification_report(all_labels, all_preds, target_names=label_names, zero_division=0, digits=4))


# ==================== MEAN ± STD OVER 5 SEEDS (TEST SET) ====================
per_run_df = pd.DataFrame(per_run_records)
print("\n" + "=" * 80)
print("TEST RESULTS — MEAN ± STD OVER 5 SEEDS")
print("=" * 80)
mean_std_summary = {}
for metric in metric_names:
    m = float(np.mean(per_run_df[metric]))
    s = float(np.std(per_run_df[metric]))
    mean_std_summary[metric] = (m, s)
    print(f"  {metric:12s}: {m:.4f} ± {s:.4f}")


# ==================== SAVE RESULTS ====================
os.makedirs(OUTPUT_DIR, exist_ok=True)

summary = pd.DataFrame([{
    'task':        'news_source_co_attn',
    'model':       'HelaBERT_small',
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
print(f"\nResults saved to {OUTPUT_DIR}/results.csv")

mean_std_row = {'run': 'mean_std', 'seed': '', 'checkpoint': ''}
for metric in metric_names:
    m, s = mean_std_summary[metric]
    mean_std_row[metric] = f"{m:.4f} ± {s:.4f}"
per_seed_df = pd.concat([per_run_df, pd.DataFrame([mean_std_row])], ignore_index=True)
per_seed_df.to_csv(f"{OUTPUT_DIR}/results_per_seed.csv", index=False)
print(f"Per-seed results + mean±std saved to {OUTPUT_DIR}/results_per_seed.csv")
