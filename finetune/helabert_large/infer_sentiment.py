"""
HelaBERT Inference for Sentiment Analysis
Loads saved checkpoints from HelaBERT_paper_sentiment/run_N/ and evaluates
on the same test split used during training (reproduces seed-based 4:1 splits).

Usage (from /ml/HelaBERT_Analysis):
    python finetune/helabert_small/infer_sentiment.py
    python finetune/helabert_small/infer_sentiment.py --run 1
    python finetune/helabert_small/infer_sentiment.py --all-runs
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report,
)
from transformers import (
    BertConfig, BertForMaskedLM, BertModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

# ==================== CONFIGURATION ====================
BERT_MODEL_PATH  = "HelaBERT"
TOKENIZER_MODEL  = "tokenizer/unigram_32000_0.9995.model"
BERT_CONFIG_FILE = "HelaBERT/config.json"
DATA_PATH        = "data/sinhala-sentiment-analysis/train.csv"
OUTPUT_DIR       = "HelaBERT_paper_sentiment"

MAX_SEQ_LENGTH = 512
BATCH_SIZE     = 32
DROPOUT        = 0.1
TEST_SIZE      = 0.2
SEEDS          = [42, 123, 456, 789, 1024]


# ==================== DATASET ====================
class SentimentDataset(Dataset):
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
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ==================== MODEL ====================
class SentimentModel(nn.Module):
    def __init__(self, bert, hidden_size, num_labels, dropout=0.1):
        super().__init__()
        self.bert       = bert
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls    = out.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


# ==================== HELPERS ====================
def find_checkpoint_dir(run_dir):
    """Return the checkpoint subdirectory saved by Trainer, or run_dir itself."""
    ckpts = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")))
    if ckpts:
        return ckpts[-1]
    if os.path.exists(os.path.join(run_dir, "pytorch_model.bin")):
        return run_dir
    raise FileNotFoundError(
        f"No checkpoint found in {run_dir}. Has training completed for this run?"
    )


def load_model_from_checkpoint(ckpt_dir, num_labels, hidden_size):
    """Reconstruct model architecture and load saved weights."""
    if os.path.exists(BERT_CONFIG_FILE):
        cfg = BertConfig.from_json_file(BERT_CONFIG_FILE)
    else:
        cfg = BertConfig.from_pretrained(BERT_MODEL_PATH)

    try:
        backbone = BertModel.from_pretrained(BERT_MODEL_PATH)
    except Exception:
        mlm      = BertForMaskedLM.from_pretrained(BERT_MODEL_PATH)
        backbone = mlm.bert

    model = SentimentModel(backbone, hidden_size, num_labels, DROPOUT)

    state_dict_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    state_dict = torch.load(state_dict_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def run_inference(model, dataset, device):
    """Run forward pass over dataset; return (all_preds, all_labels)."""
    model.eval()
    model.to(device)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels']

            out   = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(out.logits, dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    return np.array(all_preds), np.array(all_labels)


def evaluate_run(run_idx, all_texts, all_labels_enc, le, num_labels, device):
    seed    = SEEDS[run_idx - 1]
    run_dir = os.path.join(OUTPUT_DIR, f"run_{run_idx}")

    print(f"\n{'='*80}")
    print(f"RUN {run_idx}/5  (seed={seed})")
    print("=" * 80)

    ckpt_dir = find_checkpoint_dir(run_dir)
    print(f"Loading checkpoint: {ckpt_dir}")

    if os.path.exists(BERT_CONFIG_FILE):
        cfg = BertConfig.from_json_file(BERT_CONFIG_FILE)
    else:
        cfg = BertConfig.from_pretrained(BERT_MODEL_PATH)

    model = load_model_from_checkpoint(ckpt_dir, num_labels, cfg.hidden_size)

    _, test_texts, _, test_labels = train_test_split(
        all_texts, all_labels_enc,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=all_labels_enc,
    )
    print(f"Test samples: {len(test_texts):,}")

    test_ds = SentimentDataset(test_texts, test_labels, sp, MAX_SEQ_LENGTH)
    preds, labels = run_inference(model, test_ds, device)

    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    accuracy = accuracy_score(labels, preds)

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Macro-F1:  {macro_f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(labels, preds, target_names=le.classes_, zero_division=0))

    return macro_f1


# ==================== MAIN ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for HelaBERT Sentiment Analysis")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", type=int, choices=range(1, 6),
                       help="Evaluate a specific run (1–5). Default: run 1.")
    group.add_argument("--all-runs", action="store_true",
                       help="Evaluate all 5 runs and report aggregate metrics.")
    args = parser.parse_args()

    print("=" * 80)
    print("HelaBERT INFERENCE — SENTIMENT ANALYSIS")
    print("=" * 80)

    # --- Environment ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    assert os.path.exists(BERT_MODEL_PATH),  f"{BERT_MODEL_PATH} not found"
    assert os.path.exists(TOKENIZER_MODEL),  f"{TOKENIZER_MODEL} not found"
    assert os.path.exists(DATA_PATH),        f"{DATA_PATH} not found"
    assert os.path.exists(OUTPUT_DIR),       f"{OUTPUT_DIR} not found — has training completed?"

    # --- Tokenizer ---
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    PAD_ID = sp.pad_id()

    # --- Data ---
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    text_col  = next(c for c in df.columns if 'comment'   in c.lower())
    label_col = next(c for c in df.columns if 'sentiment' in c.lower())

    df = df[[text_col, label_col]].dropna()
    df[text_col]  = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 0].reset_index(drop=True)

    le = LabelEncoder()
    df['label_id'] = le.fit_transform(df[label_col])
    num_labels     = len(le.classes_)

    all_texts      = df[text_col].tolist()
    all_labels_enc = df['label_id'].tolist()

    print(f"Loaded {len(df):,} samples, {num_labels} classes: {list(le.classes_)}")

    # --- Runs ---
    runs_to_eval = list(range(1, 6)) if args.all_runs else [args.run or 1]
    run_f1s = []

    for run_idx in runs_to_eval:
        macro_f1 = evaluate_run(run_idx, all_texts, all_labels_enc, le, num_labels, device)
        run_f1s.append(macro_f1)

    # --- Summary ---
    if len(run_f1s) > 1:
        print("\n" + "=" * 80)
        print("AGGREGATE RESULTS")
        print("=" * 80)
        for i, f1 in zip(runs_to_eval, run_f1s):
            print(f"  Run {i}: {f1:.4f}")
        print(f"\nMacro-F1: {np.mean(run_f1s):.4f} ± {np.std(run_f1s):.4f}")
