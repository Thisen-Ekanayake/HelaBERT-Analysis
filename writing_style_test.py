"""
Evaluation script for the fine-tuned BERT Writing Style Classification model.

Loads the best model saved by writing_style_finetune.py and evaluates it
against a held-out test CSV.

The training script saves a label_mapping.csv alongside the best model
(best_model/label_mapping.csv). This script loads that file to map numeric
predictions back to human-readable style names.

Expected CSV format:
    comments, labels                 (or any column names containing 'comment' / 'label')
    "<sinhala text>", "LABEL_NAME"
    ...

Usage:
    python evaluate_writing_style.py
    python evaluate_writing_style.py --test_path path/to/test.csv --model_dir path/to/best_model
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
DEFAULT_MODEL_DIR       = "HelaBERT_finetuned_writing_style_cv/best_model"
DEFAULT_TOKENIZER_MODEL = "tokenizer/unigram_32000_0.9995.model"
DEFAULT_TEST_PATH       = "data/Writing-style-classification/test/writing_style_test.csv"
DEFAULT_OUTPUT_DIR      = "eval_results_writing_style"

MAX_LENGTH  = 512      # matches training — writing style uses full context
BATCH_SIZE  = 16
NUM_WORKERS = 2


# ==================== DATASET ====================
class SentencePieceDataset(Dataset):
    def __init__(self, texts, labels, sp_processor, max_length=512):
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
            'input_ids':      torch.tensor(token_ids,        dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask,   dtype=torch.long),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ==================== HELPERS ====================
def load_label_mapping(model_dir: str) -> dict:
    """
    Load the id -> label_name mapping saved by the training script.
    Looks for label_mapping.csv inside model_dir first, then one level up.
    Falls back to numeric labels if not found.
    """
    candidates = [
        os.path.join(model_dir, "label_mapping.csv"),
        os.path.join(os.path.dirname(model_dir), "label_mapping.csv"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            mapping_df = pd.read_csv(path)
            id_to_label = dict(zip(mapping_df['label_id'], mapping_df['label_name']))
            print(f"✓ Label mapping loaded from: {path}")
            for lid, lname in sorted(id_to_label.items()):
                print(f"  [{lid:2d}] {lname}")
            return id_to_label
    print("  label_mapping.csv not found — using numeric label IDs as names")
    return {}


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
        df = df.iloc[:, -2:]
        df.columns = ['comment', 'label']

    df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c], errors='ignore')
    df = df.dropna(subset=['comment', 'label'])
    df['comment'] = df['comment'].astype(str).str.strip()
    df['label']   = df['label'].astype(str).str.strip().str.upper()
    df = df[df['comment'].str.len() > 0]
    return df


def encode_labels(df: pd.DataFrame, id_to_label: dict) -> tuple[pd.DataFrame, dict]:
    """
    Convert string label column to integer IDs.
    Uses the id->label mapping from training if available, otherwise fits fresh.
    Returns the updated df and the final id_to_label dict.
    """
    if id_to_label:
        label_to_id = {v: k for k, v in id_to_label.items()}
        unknown = set(df['label'].unique()) - set(label_to_id.keys())
        if unknown:
            print(f"  Labels in test set not seen during training: {unknown}")
        df['label_id'] = df['label'].map(label_to_id)
        df = df.dropna(subset=['label_id'])
        df['label_id'] = df['label_id'].astype(int)
    else:
        # No saved mapping — encode alphabetically (same as sklearn LabelEncoder)
        classes = sorted(df['label'].unique())
        label_to_id = {lbl: idx for idx, lbl in enumerate(classes)}
        id_to_label = {idx: lbl for lbl, idx in label_to_id.items()}
        df['label_id'] = df['label'].map(label_to_id).astype(int)

    return df, id_to_label


def load_model(model_dir: str, num_labels: int, device: torch.device):
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
    parser = argparse.ArgumentParser(description="Evaluate saved BERT writing-style model")
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
    print("BERT WRITING STYLE — EVALUATION")
    print("=" * 70)
    print(f"Device    : {device}")
    print(f"Model dir : {args.model_dir}")
    print(f"Test CSV  : {args.test_path}")
    print(f"Tokenizer : {args.tokenizer}")
    print(f"Output dir: {args.output_dir}")

    # ── Assertions ────────────────────────────────────────────────────────────
    assert os.path.isdir(args.model_dir),  f" Model directory not found: {args.model_dir}"
    assert os.path.isfile(args.tokenizer), f" Tokenizer not found: {args.tokenizer}"
    assert os.path.isfile(args.test_path), f" Test CSV not found: {args.test_path}"

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print("\nLoading tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    print(f"✓ SentencePiece vocab size: {sp.get_piece_size()}")

    # ── Load label mapping saved during training ──────────────────────────────
    print("\nLoading label mapping...")
    id_to_label = load_label_mapping(args.model_dir)

    # ── Load test data ────────────────────────────────────────────────────────
    print("\nLoading test data...")
    df = load_csv(args.test_path)
    df, id_to_label = encode_labels(df, id_to_label)

    num_labels   = len(id_to_label)
    target_names = [id_to_label[i] for i in range(num_labels)]

    print(f"\n✓ {len(df)} samples, {num_labels} classes")
    print("\nTest label distribution:")
    for lid, lname in sorted(id_to_label.items()):
        cnt = (df['label_id'] == lid).sum()
        print(f"  [{lid:2d}] {lname:20s}: {cnt:>5} samples")

    # ── Build DataLoader ──────────────────────────────────────────────────────
    dataset    = SentencePieceDataset(
        df['comment'].tolist(), df['label_id'].tolist(), sp, MAX_LENGTH
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
        index=[f"True_{id_to_label[i]}"  for i in range(num_labels)],
        columns=[f"Pred_{id_to_label[i]}" for i in range(num_labels)],
    )
    print("Confusion matrix:")
    print(cm_df.to_string())

    # ── Save outputs ──────────────────────────────────────────────────────────
    pred_df = pd.DataFrame({
        'text':               df['comment'].tolist(),
        'true_label_id':      y_true,
        'true_style':         [id_to_label[i] for i in y_true],
        'predicted_label_id': y_pred,
        'predicted_style':    [id_to_label[i] for i in y_pred],
        'correct':            y_true == y_pred,
    })
    pred_path = os.path.join(args.output_dir, "test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    cm_path = os.path.join(args.output_dir, "test_confusion_matrix.csv")
    cm_df.to_csv(cm_path)

    summary = {
        'accuracy':        acc,
        'macro_precision': prec,
        'macro_recall':    rec,
        'macro_f1':        f1_macro,
        'weighted_f1':     f1_wt,
        'num_samples':     len(y_true),
        'num_classes':     num_labels,
        'classes':         ', '.join(target_names),
    }
    summary_path = os.path.join(args.output_dir, "test_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("OUTPUTS SAVED")
    print("=" * 70)
    print(f"  Predictions   → {pred_path}")
    print(f"  Confusion mat → {cm_path}")
    print(f"  Summary       → {summary_path}")
    print("\n Evaluation complete!")


if __name__ == "__main__":
    main()