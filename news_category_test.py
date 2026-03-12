"""
Evaluation script for the fine-tuned BERT News Category Classification model.

Loads the best model saved by news_category_finetune.py and evaluates it
against a held-out test CSV.

Expected CSV format:
    comments, labels          (or any column names containing 'comment' / 'label')
    <sinhala text>, <0-4>
    ...

Usage:
    python evaluate_news_category.py
    python evaluate_news_category.py --test_path path/to/test.csv --model_dir path/to/best_model
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


# ==================== DEFAULTS (mirror the training script) ====================
DEFAULT_MODEL_DIR      = "HelaBERT_finetuned_news_category_cv/best_model"
DEFAULT_TOKENIZER_MODEL = "tokenizer/unigram_32000_0.9995.model"
DEFAULT_TEST_PATH      = "data/Sinhala-News-Category-classification/test/news_test.csv"
DEFAULT_OUTPUT_DIR     = "eval_results"

NUM_LABELS    = 5
MAX_LENGTH    = 256
BATCH_SIZE    = 16
NUM_WORKERS   = 2
RANDOM_SEED   = 42

CATEGORY_NAMES = {i: f"Category_{i}" for i in range(NUM_LABELS)}


# ==================== DATASET ====================
class SentencePieceDataset(Dataset):
    def __init__(self, texts, labels, sp_processor, max_length=256):
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

        attn_mask = [1] * len(token_ids)
        pad_len   = self.max_length - len(token_ids)
        token_ids += [self.pad_id] * pad_len
        attn_mask += [0] * pad_len

        return {
            'input_ids':      torch.tensor(token_ids,        dtype=torch.long),
            'attention_mask': torch.tensor(attn_mask,        dtype=torch.long),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ==================== HELPERS ====================
def load_csv(path: str) -> pd.DataFrame:
    """Load and normalise the test CSV (same logic as training script)."""
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError:
        df = pd.read_csv(path, engine='python', on_bad_lines='skip')

    df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

    possible_comment_cols = [c for c in df.columns if 'comment' in c.lower()]
    possible_label_cols   = [c for c in df.columns if 'label'   in c.lower()]

    if possible_comment_cols and possible_label_cols:
        df = df.rename(columns={
            possible_comment_cols[0]: 'comment',
            possible_label_cols[0]:  'label',
        })
    else:
        # Fall back: take last two columns
        df = df.iloc[:, -2:]
        df.columns = ['comment', 'label']

    df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c], errors='ignore')
    df = df.dropna()
    df['comment'] = df['comment'].astype(str).str.strip()
    df['label']   = df['label'].astype(str).str.strip().astype(int)
    return df


def load_model(model_dir: str, num_labels: int, device: torch.device):
    """Load the saved BertForSequenceClassification from model_dir."""
    config            = BertConfig.from_pretrained(model_dir)
    config.num_labels = num_labels
    model             = BertForSequenceClassification.from_pretrained(
        model_dir, config=config, ignore_mismatched_sizes=True
    )
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, dataloader, device):
    all_preds, all_labels = [], []
    for batch in dataloader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].numpy()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds   = outputs.logits.argmax(dim=-1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)

    return np.array(all_labels), np.array(all_preds)


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(description="Evaluate saved BERT news-category model")
    parser.add_argument("--test_path",  default=DEFAULT_TEST_PATH,       help="Path to test CSV")
    parser.add_argument("--model_dir",  default=DEFAULT_MODEL_DIR,       help="Path to saved model directory")
    parser.add_argument("--tokenizer",  default=DEFAULT_TOKENIZER_MODEL, help="Path to SentencePiece .model file")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,      help="Directory for evaluation outputs")
    parser.add_argument("--batch_size", default=BATCH_SIZE, type=int,    help="Inference batch size")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("BERT NEWS CATEGORY — EVALUATION")
    print("=" * 70)
    print(f"Device    : {device}")
    print(f"Model dir : {args.model_dir}")
    print(f"Test CSV  : {args.test_path}")
    print(f"Tokenizer : {args.tokenizer}")
    print(f"Output dir: {args.output_dir}")

    # ── Assertions ────────────────────────────────────────────────────────────
    assert os.path.isdir(args.model_dir),  f"Model directory not found: {args.model_dir}"
    assert os.path.isfile(args.tokenizer), f"Tokenizer not found: {args.tokenizer}"
    assert os.path.isfile(args.test_path), f"Test CSV not found: {args.test_path}"

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print("\nLoading tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    print(f"✓ SentencePiece vocab size: {sp.get_piece_size()}")

    # ── Load test data ────────────────────────────────────────────────────────
    print("\nLoading test data...")
    df = load_csv(args.test_path)

    num_labels = df['label'].nunique()
    if num_labels != NUM_LABELS:
        print(f"  Detected {num_labels} classes in test set (expected {NUM_LABELS})")
    category_names = {i: f"Category_{i}" for i in range(num_labels)}
    target_names   = [category_names[i] for i in range(num_labels)]

    print(f"✓ {len(df)} samples, {num_labels} classes")
    print("\nTest label distribution:")
    for lbl, cnt in df['label'].value_counts().sort_index().items():
        print(f"  Label {lbl}: {cnt:>5} samples")

    # ── Build DataLoader ──────────────────────────────────────────────────────
    dataset    = SentencePieceDataset(
        df['comment'].tolist(), df['label'].tolist(), sp, MAX_LENGTH
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda")
    )

    # ── Load model ────────────────────────────────────────────────────────────
    print("\nLoading model...")
    model = load_model(args.model_dir, num_labels, device)
    print("✓ Model loaded")

    # ── Inference ─────────────────────────────────────────────────────────────
    print("\nRunning inference...")
    y_true, y_pred = run_inference(model, dataloader, device)
    print("✓ Inference complete")

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc      = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    f1_wt    = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec     = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec      = recall_score(y_true, y_pred,    average='macro', zero_division=0)

    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Macro Precision   : {prec:.4f}")
    print(f"  Macro Recall      : {rec:.4f}")
    print(f"  Macro F1          : {f1_macro:.4f}")
    print(f"  Weighted F1       : {f1_wt:.4f}")

    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm    = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{category_names[i]}"  for i in range(num_labels)],
        columns=[f"Pred_{category_names[i]}" for i in range(num_labels)],
    )
    print("Confusion matrix:")
    print(cm_df.to_string())

    # ── Save outputs ──────────────────────────────────────────────────────────
    # Predictions CSV
    pred_df = pd.DataFrame({
        'text':            df['comment'].tolist(),
        'true_label':      y_true,
        'predicted_label': y_pred,
        'correct':         y_true == y_pred,
    })
    pred_path = os.path.join(args.output_dir, "test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    # Confusion matrix CSV
    cm_path = os.path.join(args.output_dir, "test_confusion_matrix.csv")
    cm_df.to_csv(cm_path)

    # Summary CSV
    summary = {
        'accuracy':    acc,
        'macro_precision': prec,
        'macro_recall': rec,
        'macro_f1':    f1_macro,
        'weighted_f1': f1_wt,
        'num_samples': len(y_true),
        'num_classes': num_labels,
    }
    summary_path = os.path.join(args.output_dir, "test_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("OUTPUTS SAVED")
    print("=" * 70)
    print(f"  Predictions   → {pred_path}")
    print(f"  Confusion mat → {cm_path}")
    print(f"  Summary       → {summary_path}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()