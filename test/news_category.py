"""
Test script for News Category Classification Model

This script loads the best trained model and evaluates it on a test dataset.
Two variables need to be manually configured:
  - TEST_DATA_PATH: Path to test CSV file with columns: body, comments, labels
  - BEST_MODEL_PATH: Path to best_model directory from training output
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from transformers import BertModel, BertConfig

# ==================== CONFIGURATION ====================
# *** MANUALLY SET THESE TWO VARIABLES ***
TEST_DATA_PATH = "data/Sinhala-News-Category-classification/test/news_test.csv"  # e.g., "data/test_news_category.csv"
BEST_MODEL_PATH = "utils/finetuned_models/HelaBERT_finetuned_news_category_baseline_cv/best_model"    # e.g., "HelaBERT_finetuned_news_category_baseline_cv/best_model"

# Fixed configuration
BERT_MODEL_PATH = "HelaBERT"
TOKENIZER_MODEL = "tokenizer/unigram_32000_0.9995.model"
BERT_CONFIG_FILE = "HelaBERT/config.json"

CATEGORY_NAMES = {
    0: "Category_0", 1: "Category_1", 2: "Category_2",
    3: "Category_3", 4: "Category_4"
}

NUM_LABELS = 5
COMMENT_MAX_LENGTH = 256
BATCH_SIZE = 8
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("NEWS CATEGORY CLASSIFICATION - TEST EVALUATION")
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


# ==================== DATASET CLASS ====================
class NewsTestDataset(Dataset):
    """Dataset for test data — comments only"""

    def __init__(self, texts, labels, sp_processor, comment_max_length=256):
        self.texts = texts
        self.labels = labels
        self.sp = sp_processor
        self.comment_max_length = comment_max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = self.sp.encode(str(self.texts[idx]))[: self.comment_max_length]
        mask = [1] * len(ids)
        pad = self.comment_max_length - len(ids)
        ids += [PAD_ID] * pad
        mask += [0] * pad

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ==================== MODEL CLASS ====================
class NewsClassifier(nn.Module):
    """BERT [CLS] classifier for news category"""

    def __init__(self, bert_model, num_labels, hidden_dropout_prob=0.1):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.norm = nn.LayerNorm(self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:,0,:]
        cls_output = self.norm(cls_output)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


# ==================== DATA LOADING ====================
print("\n" + "=" * 80)
print("LOADING TEST DATA")
print("=" * 80)

assert os.path.exists(TEST_DATA_PATH), f"Test data not found: {TEST_DATA_PATH}"
df = pd.read_csv(TEST_DATA_PATH)
print(f"✓ Loaded {len(df)} test samples")

# Extract texts and labels (comments column)
test_texts = df["comments"].values
test_labels = df["labels"].values

print(f"  Texts shape: {test_texts.shape}")
print(f"  Labels shape: {test_labels.shape}")
print(f"  Unique labels: {np.unique(test_labels)}")
print(f"  Label distribution:\n{pd.Series(test_labels).value_counts().sort_index()}")

# Create dataset and dataloader
test_dataset = NewsTestDataset(test_texts, test_labels, sp, COMMENT_MAX_LENGTH)
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

# Create classifier
model = NewsClassifier(bert_model, NUM_LABELS)

# Load best model weights
assert os.path.exists(
    f"{BEST_MODEL_PATH}/pytorch_model.bin"
), f"Best model weights not found: {BEST_MODEL_PATH}/pytorch_model.bin"
model.load_state_dict(torch.load(f"{BEST_MODEL_PATH}/pytorch_model.bin", map_location=DEVICE))
print(f"✓ Best model weights loaded from {BEST_MODEL_PATH}")

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
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(input_ids, attention_mask)
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

target_names = [CATEGORY_NAMES.get(i, f"Category_{i}") for i in range(NUM_LABELS)]
print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

print("\n" + "-" * 80)
print("CONFUSION MATRIX")
print("-" * 80)

cm = confusion_matrix(all_labels, all_preds)
cm_df = pd.DataFrame(
    cm,
    index=[f"True_{CATEGORY_NAMES.get(i, i)}" for i in range(NUM_LABELS)],
    columns=[f"Pred_{CATEGORY_NAMES.get(i, i)}" for i in range(NUM_LABELS)],
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
    columns=[CATEGORY_NAMES.get(i, f"Category_{i}") for i in range(NUM_LABELS)],
).T
per_class_metrics.to_csv(f"{output_dir}/per_class_metrics.csv")

# Save predictions
predictions_df = pd.DataFrame({
    "true_label": all_labels,
    "predicted_label": all_preds,
    "correct": all_labels == all_preds,
    "confidence": np.max(torch.softmax(torch.tensor(all_logits), dim=1).numpy(), axis=1),
})
predictions_df.to_csv(f"{output_dir}/predictions.csv", index=False)

# Save summary metrics
summary = pd.DataFrame({
    "metric": ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"],
    "value": [accuracy, precision, recall, f1_macro, f1_weighted],
})
summary.to_csv(f"{output_dir}/test_metrics.csv", index=False)

print(f"\n" + "=" * 80)
print("RESULTS SAVED")
print("=" * 80)
print(f"  Confusion matrix   → {output_dir}/confusion_matrix.csv")
print(f"  Per-class metrics  → {output_dir}/per_class_metrics.csv")
print(f"  Predictions        → {output_dir}/predictions.csv")
print(f"  Summary metrics    → {output_dir}/test_metrics.csv")

print("\n" + "=" * 80)
print("TEST EVALUATION COMPLETE!")
print("=" * 80)
