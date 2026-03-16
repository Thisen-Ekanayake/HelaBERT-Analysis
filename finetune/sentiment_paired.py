"""
BERT Fine-tuning for Sentiment Analysis — 5-Fold Cross Validation
— Balanced training via oversampling + weighted loss —
— Stage 1: Paired Input Format ([CLS] article [SEP] comment [SEP]) —

Architecture:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Article body + Comment phrase (paired input)                            │
  │                                                                          │
  │  Tokenization format:                                                    │
  │    [CLS] <article_tokens> [SEP] <comment_tokens> [SEP]                  │
  │                                                                          │
  │  Example:                                                                │
  │    [CLS] දකුණු පළෙතේ සංචාරක... [SEP] කවුරුහරි කියනවනම්... [SEP]          │
  │                        ↓                                                 │
  │                     BERT encoder                                         │
  │                        ↓                                                 │
  │                   [CLS] token                                            │
  │                        ↓                                                 │
  │              LayerNorm → Dropout → Classifier                            │
  │                        ↓                                                 │
  │                    Sentiment label                                       │
  └──────────────────────────────────────────────────────────────────────────┘

  Single BERT encoder processes both article and comment together.
  Token type IDs distinguish article (0) from comment (1).
  This paired format captures the relationship between article and comment.

Cross-validation strategy:
  • StratifiedKFold(n_splits=5) preserves class distribution in every fold
  • Oversampling applied ONLY to each fold's training split (never the val split)
  • Class weights recomputed per fold from that fold's raw training distribution
  • Fresh model loaded at the start of every fold
  • Best model across all folds (highest val macro-F1) is saved as the final model
  • Mean ± std reported across all folds at the end

Expected CSV format (with flexible column detection):
    title, body, comment_phrase, comment_sentiment
    <article title>, <article body>, <comment text>, <sentiment label>
    ...
  (Uses LabelEncoder to handle any number of sentiment classes)
"""

import os
import traceback
from collections import Counter
import random as stdlib_random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
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
print("BERT FINE-TUNING — 5-FOLD CV  [PAIRED INPUT: article [SEP] comment]")
print("=" * 80)

BERT_MODEL_PATH  = "HelaBERT"
TOKENIZER_MODEL  = "tokenizer/unigram_32000_0.9995.model"
BERT_CONFIG_FILE = "HelaBERT/config.json"
DATA_PATH        = "data/sinhala-sentiment-analysis/outputs/train.csv"

# Training hyperparameters
ARTICLE_MAX_LENGTH           = 384    # tokens for article body
COMMENT_MAX_LENGTH           = 125    # tokens for comment
# Total max length: [CLS] + article + [SEP] + comment + [SEP] = 1 + 384 + 1 + 128 + 1 = 515
TOTAL_MAX_LENGTH             = ARTICLE_MAX_LENGTH + COMMENT_MAX_LENGTH + 3  # +3 for [CLS], [SEP], [SEP]
TRAIN_BATCH_SIZE             = 4
EVAL_BATCH_SIZE              = 8
LEARNING_RATE                = 5e-5
NUM_EPOCHS                   = 3      # early stopping decides actual epoch count
WARMUP_RATIO                 = 0.1
WEIGHT_DECAY                 = 0.05
GRADIENT_ACCUMULATION_STEPS  = 4      # effective batch = 16
EARLY_STOPPING_PATIENCE      = 3

# Cross-validation
N_FOLDS = 5

# Balancing
OVERSAMPLE_TRAIN  = True
USE_CLASS_WEIGHTS = True

# Output
OUTPUT_DIR     = "HelaBERT_finetuned_sentiment_paired_cv"
BEST_MODEL_DIR = f"{OUTPUT_DIR}/best_model"
STAGE_TAG      = "paired_input"

# Misc
RANDOM_SEED    = 42
USE_FP16       = True
NUM_WORKERS    = 2
USE_WANDB      = True
WANDB_PROJECT  = "bert-sentiment-finetuning"
WANDB_GROUP    = f"5fold_cv_paired_lr{LEARNING_RATE}_bs{TRAIN_BATCH_SIZE}"
WANDB_ENTITY   = None

print(f"\n✓ Config loaded — {N_FOLDS}-fold CV, oversampling={'on' if OVERSAMPLE_TRAIN else 'off'}, "
      f"class_weights={'on' if USE_CLASS_WEIGHTS else 'off'}")
print(f"  Architecture   : BERT [CLS] → LayerNorm → Dropout → Linear  (paired input format)")
print(f"  Input format   : [CLS] article [SEP] comment [SEP]")
print(f"  Article max length: {ARTICLE_MAX_LENGTH}  Comment max length: {COMMENT_MAX_LENGTH}")
print(f"  Total max length: {TOTAL_MAX_LENGTH}")
print(f"  Effective batch: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")


# ==================== SEEDS ====================
random.seed(RANDOM_SEED)
stdlib_random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print("✓ Random seeds set")


# ==================== ENVIRONMENT ====================
print("\n" + "=" * 80)
print("ENVIRONMENT")
print("=" * 80)
print(f"PyTorch: {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CPU only — training will be slow")

assert os.path.exists(BERT_MODEL_PATH),  f"{BERT_MODEL_PATH} not found"
assert os.path.exists(TOKENIZER_MODEL),  f"{TOKENIZER_MODEL} not found"
assert os.path.exists(DATA_PATH),        f"{DATA_PATH} not found"
print("All paths verified")


# ==================== TOKENIZER ====================
print("\n" + "=" * 80)
print("LOADING TOKENIZER")
print("=" * 80)

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)
PAD_ID = sp.pad_id()
SEP_ID = sp.piece_to_id("[SEP]")  # SentencePiece special token
CLS_ID = sp.piece_to_id("[CLS]")
print(f"SentencePiece loaded — vocab size: {sp.get_piece_size()}, PAD_ID: {PAD_ID}")
print(f"  [CLS] ID: {CLS_ID}, [SEP] ID: {SEP_ID}")


# ==================== HELPERS ====================
def find_col(df, name):
    """Find column by exact match or case-insensitive substring match."""
    if name in df.columns:
        return name
    hits = [c for c in df.columns if name.lower() in c.lower()]
    if hits:
        return hits[0]
    raise KeyError(f"Column '{name}' not found in {list(df.columns)}")


def load_csv(path):
    """Load CSV with flexible column detection."""
    sep = '\t' if path.endswith('.tsv') else ','
    try:
        df = pd.read_csv(path, sep=sep)
    except pd.errors.ParserError:
        df = pd.read_csv(path, sep=sep, engine='python', on_bad_lines='skip')
    df.columns = df.columns.str.strip()
    
    # Find columns flexibly
    title_col = find_col(df, "title")
    body_col = find_col(df, "body")
    comment_col = find_col(df, "comment")
    label_col = find_col(df, "sentiment")
    
    df = df[[title_col, body_col, comment_col, label_col]].copy()
    df.columns = ['title', 'body', 'comment', 'label']
    df = df.dropna(subset=['comment', 'label'])
    df['title']   = df['title'].fillna('').astype(str).str.strip()
    df['body']    = df['body'].fillna('').astype(str).str.strip()
    df['comment'] = df['comment'].astype(str).str.strip()
    df['label']   = df['label'].astype(str).str.strip()
    return df[df['comment'].str.len() > 0].reset_index(drop=True)


# ==================== DATASET CLASS ====================
class PairedInputSentimentDataset(Dataset):
    """
    Paired input format: [CLS] article [SEP] comment [SEP]
    
    Each sample exposes:
      input_ids      [TOTAL_MAX_LENGTH]  — [CLS] article [SEP] comment [SEP]
      attention_mask [TOTAL_MAX_LENGTH]
      token_type_ids [TOTAL_MAX_LENGTH]  — 0 for article, 1 for comment
      labels         scalar
    """

    def __init__(self, bodies, comments, labels, sp_processor, 
                 article_max_length=384, comment_max_length=128):
        self.bodies              = bodies
        self.comments            = comments
        self.labels              = labels
        self.sp                  = sp_processor
        self.article_max_length  = article_max_length
        self.comment_max_length  = comment_max_length
        self.total_max_length    = article_max_length + comment_max_length + 3

    def __len__(self):
        return len(self.bodies)

    def __getitem__(self, idx):
        # Tokenize article (body)
        article_ids = self.sp.encode(str(self.bodies[idx]))[:self.article_max_length]
        article_mask = [1] * len(article_ids)
        article_token_types = [0] * len(article_ids)  # token type 0 for article
        
        # Tokenize comment
        comment_ids = self.sp.encode(str(self.comments[idx]))[:self.comment_max_length]
        comment_mask = [1] * len(comment_ids)
        comment_token_types = [1] * len(comment_ids)  # token type 1 for comment
        
        # Build paired sequence: [CLS] article [SEP] comment [SEP]
        # Note: SentencePiece doesn't add special tokens automatically, so we manually add them
        input_ids = [CLS_ID] + article_ids + [SEP_ID] + comment_ids + [SEP_ID]
        token_type_ids = [0] + article_token_types + [0] + comment_token_types + [0]
        attention_mask = [1] + article_mask + [1] + comment_mask + [1]
        
        # Pad to total_max_length
        pad_length = self.total_max_length - len(input_ids)
        input_ids += [PAD_ID] * pad_length
        token_type_ids += [0] * pad_length
        attention_mask += [0] * pad_length
        
        return {
            'input_ids':      torch.tensor(input_ids[:self.total_max_length], dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids[:self.total_max_length], dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask[:self.total_max_length], dtype=torch.long),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ==================== BALANCING HELPERS ====================
def oversample(bodies, comments, labels, seed=42):
    """Oversample minority classes to match majority class count."""
    stdlib_random.seed(seed)
    counts = Counter(labels)
    max_count = max(counts.values())

    bal_bodies, bal_comments, bal_labels = list(bodies), list(comments), list(labels)
    for label, count in counts.items():
        needed = max_count - count
        if needed == 0:
            continue
        indices = [i for i, l in enumerate(labels) if l == label]
        extras = stdlib_random.choices(indices, k=needed)
        bal_bodies += [bodies[i] for i in extras]
        bal_comments += [comments[i] for i in extras]
        bal_labels += [labels[i] for i in extras]

    combined = list(zip(bal_bodies, bal_comments, bal_labels))
    stdlib_random.shuffle(combined)
    bal_bodies, bal_comments, bal_labels = zip(*combined)
    return list(bal_bodies), list(bal_comments), list(bal_labels)


def compute_class_weights(labels, num_labels):
    """Inverse-frequency weights normalised so they sum to num_labels."""
    counts = Counter(labels)
    weights = torch.tensor(
        [1.0 / counts.get(i, 1) for i in range(num_labels)],
        dtype=torch.float,
    )
    weights = weights / weights.sum() * num_labels
    return weights


# ==================== MODEL ====================
class PairedInputSentimentModel(nn.Module):
    """
    BERT model for paired input sentiment classification.
    Input: [CLS] article [SEP] comment [SEP]
    """

    def __init__(self, bert, hidden_size, num_labels):
        super().__init__()
        self.bert = bert
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Get [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [B, H]
        
        # Apply normalization, dropout, and classifier
        logits = self.classifier(self.dropout(self.norm(cls_output)))
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return SequenceClassifierOutput(loss=loss, logits=logits)


# ==================== COLLATOR ====================
def collate_fn(batch):
    """Collate function for DataLoader."""
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ==================== CUSTOM TRAINER (supports class weights) ====================
class PairedInputTrainer(Trainer):
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
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        if self.class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
            loss = loss_fn(outputs.logits, labels)
        else:
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# ==================== METRICS ====================
def compute_metrics(eval_pred: EvalPrediction):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    return {
        'accuracy':    accuracy_score(labels, preds),
        'precision':   precision_score(labels, preds, average='macro', zero_division=0),
        'recall':      recall_score(labels, preds, average='macro', zero_division=0),
        'f1':          f1_score(labels, preds, average='macro', zero_division=0),
        'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
    }


# ==================== FRESH MODEL LOADER ====================
def load_fresh_model():
    """Load a fresh BERT model for each fold."""
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
        mlm = BertForMaskedLM.from_pretrained(BERT_MODEL_PATH)
        backbone = mlm.bert
        if cfg is None:
            cfg = mlm.config
        print("  ✓ Weights loaded via BertForMaskedLM")

    hs = cfg.hidden_size if cfg else backbone.config.hidden_size

    m = PairedInputSentimentModel(
        bert=backbone,
        hidden_size=hs,
        num_labels=NUM_LABELS,
    )
    return m, cfg, hs


# ==================== LOAD DATA ====================
print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

df = load_csv(DATA_PATH)
print(f"✓ Loaded {len(df):,} samples")


# ==================== CREATE OUTPUT DIRECTORY ====================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== ENCODE LABELS ====================
print("\n" + "=" * 80)
print("ENCODING LABELS")
print("=" * 80)

le = LabelEncoder()
le.fit(sorted(df['label'].unique()))
df['label_id'] = le.transform(df['label'])
NUM_LABELS = len(le.classes_)
id_to_label = {i: lbl for i, lbl in enumerate(le.classes_)}

mapping_df = pd.DataFrame({'label_id': list(id_to_label.keys()),
                            'label_name': list(id_to_label.values())})
mapping_df.to_csv(f"{OUTPUT_DIR}/label_mapping.csv", index=False)
print(f"✓ {NUM_LABELS} labels: {', '.join(le.classes_)}")
for idx, lbl in sorted(id_to_label.items()):
    cnt = (df['label_id'] == idx).sum()
    print(f"  [{idx}] {lbl:20s}: {cnt:6d} ({100*cnt/len(df):.1f}%)")

all_bodies = df['body'].tolist()
all_comments = df['comment'].tolist()
all_labels = df['label_id'].tolist()


# ==================== PROBE MODEL ARCHITECTURE ====================
print("\n" + "=" * 80)
print("PROBING MODEL ARCHITECTURE")
print("=" * 80)
os.makedirs(OUTPUT_DIR, exist_ok=True)
_probe_model, bert_config, hidden_size = load_fresh_model()
total_params = sum(p.numel() for p in _probe_model.parameters())
trainable_params = sum(p.numel() for p in _probe_model.parameters() if p.requires_grad)
print(f"\nTotal params      : {total_params:,}")
print(f"Trainable params  : {trainable_params:,}  ({100*trainable_params/total_params:.1f}%)")
print(f"Hidden size       : {hidden_size}")
print(f"Num labels        : {NUM_LABELS}")
del _probe_model


# ==================== CV STATE ====================
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
fold_metrics = []
all_y_true = []
all_y_pred = []
all_oof_texts = []
best_fold_f1 = -1.0
best_fold_idx = -1
wandb_group_url = None


# ==================== FOLD LOOP ====================
for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(all_comments, all_labels), start=1):

    print("\n" + "=" * 80)
    print(f"FOLD {fold_idx} / {N_FOLDS}")
    print("=" * 80)

    fold_output_dir = f"{OUTPUT_DIR}/fold_{fold_idx}"
    os.makedirs(fold_output_dir, exist_ok=True)

    # ── Split ───────────────────────────────────────────────────────────────
    fold_train_bodies = [all_bodies[i] for i in train_idx]
    fold_train_comments = [all_comments[i] for i in train_idx]
    fold_train_labels = [all_labels[i] for i in train_idx]

    fold_val_bodies = [all_bodies[i] for i in val_idx]
    fold_val_comments = [all_comments[i] for i in val_idx]
    fold_val_labels = [all_labels[i] for i in val_idx]

    print(f"Train: {len(fold_train_labels)} samples  |  Val: {len(fold_val_labels)} samples")

    print("Train label distribution (before oversampling):")
    for lbl, cnt in sorted(Counter(fold_train_labels).items()):
        print(f"  [{lbl:2d}] {id_to_label[lbl]:20s}: {cnt}")

    # ── Oversample (train split only) ───────────────────────────────────────
    if OVERSAMPLE_TRAIN:
        fold_train_bodies, fold_train_comments, fold_train_labels = oversample(
            fold_train_bodies, fold_train_comments, fold_train_labels,
            seed=RANDOM_SEED + fold_idx
        )
        print(f"After oversampling: {len(fold_train_labels)} train samples")
        print("Train label distribution (after oversampling):")
        for lbl, cnt in sorted(Counter(fold_train_labels).items()):
            print(f"  [{lbl:2d}] {id_to_label[lbl]:20s}: {cnt}")

    # ── Class weights ───────────────────────────────────────────────────────
    raw_fold_labels = [all_labels[i] for i in train_idx]
    fold_weights = compute_class_weights(raw_fold_labels, NUM_LABELS)
    if USE_CLASS_WEIGHTS:
        print("Class weights:")
        for i, w in enumerate(fold_weights):
            print(f"  [{i:2d}] {id_to_label[i]:20s}: {w.item():.4f}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = PairedInputSentimentDataset(
        fold_train_bodies, fold_train_comments, fold_train_labels, sp,
        article_max_length=ARTICLE_MAX_LENGTH, comment_max_length=COMMENT_MAX_LENGTH
    )
    val_ds = PairedInputSentimentDataset(
        fold_val_bodies, fold_val_comments, fold_val_labels, sp,
        article_max_length=ARTICLE_MAX_LENGTH, comment_max_length=COMMENT_MAX_LENGTH
    )

    # ── Fresh model ──────────────────────────────────────────────────────────
    print(f"Loading fresh model for fold {fold_idx}...")
    model, bert_config, hidden_size = load_fresh_model()
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_p:,} total, {trainable_p:,} trainable")

    # ── W&B ──────────────────────────────────────────────────────────────────
    fold_wandb_name = f"fold_{fold_idx}"
    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            group=WANDB_GROUP,
            name=fold_wandb_name,
            config={
                'fold': fold_idx,
                'n_folds': N_FOLDS,
                'train_samples': len(fold_train_labels),
                'val_samples': len(fold_val_labels),
                'architecture': 'paired_input_format',
                'article_max_length': ARTICLE_MAX_LENGTH,
                'comment_max_length': COMMENT_MAX_LENGTH,
                'total_max_length': TOTAL_MAX_LENGTH,
                'hidden_size': hidden_size,
                'learning_rate': LEARNING_RATE,
                'batch_size': TRAIN_BATCH_SIZE,
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
        optim='adamw_8bit' if torch.cuda.is_available() else 'adamw_torch',
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = PairedInputTrainer(
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
        best_fold_f1 = val_f1
        best_fold_idx = fold_idx
        print(f"  → New best fold! (F1 = {best_fold_f1:.4f})")
        os.makedirs(BEST_MODEL_DIR, exist_ok=True)
        trainer.save_model(BEST_MODEL_DIR)
        arch_cfg = pd.DataFrame({
            'param': ['stage', 'architecture', 'hidden_size', 'num_labels',
                      'article_max_length', 'comment_max_length', 'total_max_length'],
            'value': [STAGE_TAG, 'paired_input_format', hidden_size, NUM_LABELS,
                      ARTICLE_MAX_LENGTH, COMMENT_MAX_LENGTH, TOTAL_MAX_LENGTH]
        })
        arch_cfg.to_csv(f"{BEST_MODEL_DIR}/arch_config.csv", index=False)
        mapping_df.to_csv(f"{BEST_MODEL_DIR}/label_mapping.csv", index=False)
        print(f"  → Saved to {BEST_MODEL_DIR}")

    fold_m = {
        'fold': fold_idx,
        'accuracy': eval_result.get('eval_accuracy', 0.0),
        'precision': eval_result.get('eval_precision', 0.0),
        'recall': eval_result.get('eval_recall', 0.0),
        'f1': eval_result.get('eval_f1', 0.0),
        'f1_weighted': eval_result.get('eval_f1_weighted', 0.0),
    }
    fold_metrics.append(fold_m)

    # ── Predictions (for OOF) ────────────────────────────────────────────────
    preds_result = trainer.predict(val_ds)
    val_preds = np.argmax(preds_result.predictions, axis=1)
    all_y_true.extend(fold_val_labels)
    all_y_pred.extend(val_preds)
    all_oof_texts.extend(fold_val_comments)

    print(f"\nFold {fold_idx} done — macro F1: {fold_m['f1']:.4f}")

    if USE_WANDB:
        wandb.finish()
        print(f"W&B fold {fold_idx} run finished")

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
std_metrics = metrics_df.drop(columns=['fold']).std()

print("\n" + "-" * 60)
print(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'95% CI':>20}")
print("-" * 60)
for metric in ['accuracy', 'precision', 'recall', 'f1', 'f1_weighted']:
    m = mean_metrics[metric]
    s = std_metrics[metric]
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
target_names = [id_to_label.get(i, f"Label_{i}") for i in range(NUM_LABELS)]
print(classification_report(oof_y_true, oof_y_pred, target_names=target_names, digits=4))

oof_f1 = f1_score(oof_y_true, oof_y_pred, average='macro', zero_division=0)
oof_f1_w = f1_score(oof_y_true, oof_y_pred, average='weighted', zero_division=0)
oof_acc = accuracy_score(oof_y_true, oof_y_pred)

print(f"OOF macro F1:    {oof_f1:.4f}")
print(f"OOF weighted F1: {oof_f1_w:.4f}")
print(f"OOF accuracy:    {oof_acc:.4f}")

oof_cm = confusion_matrix(oof_y_true, oof_y_pred)
pd.DataFrame(
    oof_cm,
    index=[f"True_{id_to_label.get(i, i)}" for i in range(NUM_LABELS)],
    columns=[f"Pred_{id_to_label.get(i, i)}" for i in range(NUM_LABELS)],
).to_csv(f"{OUTPUT_DIR}/oof_confusion_matrix.csv")

pd.DataFrame({
    'comment': all_oof_texts,
    'true_label': oof_y_true,
    'predicted_label': oof_y_pred,
    'correct': oof_y_true == oof_y_pred,
}).to_csv(f"{OUTPUT_DIR}/oof_predictions.csv", index=False)

print(f"\nOOF confusion matrix → {OUTPUT_DIR}/oof_confusion_matrix.csv")
print(f"OOF predictions      → {OUTPUT_DIR}/oof_predictions.csv")


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
            f"cv_std/{metric}": std_metrics[metric],
        })
    wandb.log({
        "oof/f1": oof_f1,
        "oof/f1_weighted": oof_f1_w,
        "oof/accuracy": oof_acc,
        "best_fold": best_fold_idx,
        "best_fold_f1": best_fold_f1,
        "cv_fold_metrics": wandb.Table(dataframe=metrics_df),
    })
    try:
        import plotly.figure_factory as ff
        fig = ff.create_annotated_heatmap(
            z=oof_cm,
            x=[f"Pred_{id_to_label.get(i, i)}" for i in range(NUM_LABELS)],
            y=[f"True_{id_to_label.get(i, i)}" for i in range(NUM_LABELS)],
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
    print(f"CV summary logged — group: {WANDB_GROUP}")


# ==================== FINAL SUMMARY ====================
cv_summary = {
    'Model': 'BERT with SentencePiece',
    'Architecture': 'BERT [CLS] paired input (article + comment)',
    'Stage': STAGE_TAG,
    'Task': 'Sentiment Analysis',
    'Input Format': '[CLS] article [SEP] comment [SEP]',
    'Balancing': 'Oversample + Class Weights',
    'N Folds': N_FOLDS,
    'Num Classes': NUM_LABELS,
    'Classes': ', '.join(le.classes_),
    'Total Samples': len(df),
    'Article Max Length': ARTICLE_MAX_LENGTH,
    'Comment Max Length': COMMENT_MAX_LENGTH,
    'Total Max Length': TOTAL_MAX_LENGTH,
    'Learning Rate': LEARNING_RATE,
    'Effective Batch Size': TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
    'Best Fold': best_fold_idx,
    'Best Fold F1': f'{best_fold_f1:.4f}',
    'Mean F1 (macro)': f'{mean_metrics["f1"]:.4f} ± {std_metrics["f1"]:.4f}',
    'Mean F1 (weighted)': f'{mean_metrics["f1_weighted"]:.4f} ± {std_metrics["f1_weighted"]:.4f}',
    'Mean Accuracy': f'{mean_metrics["accuracy"]:.4f} ± {std_metrics["accuracy"]:.4f}',
    'OOF F1 (macro)': f'{oof_f1:.4f}',
    'OOF F1 (weighted)': f'{oof_f1_w:.4f}',
    'OOF Accuracy': f'{oof_acc:.4f}',
}

pd.DataFrame([cv_summary]).to_csv(f'{OUTPUT_DIR}/cv_summary.csv', index=False)

print("\n" + "=" * 80)
print("FINAL CROSS-VALIDATION SUMMARY")
print("=" * 80)
for k, v in cv_summary.items():
    print(f"  {k:<28}: {v}")

print("\n" + "=" * 80)
print("5-FOLD CROSS-VALIDATION (PAIRED INPUT FORMAT) COMPLETE!")
print("=" * 80)
print(f"\nOutputs saved to: {OUTPUT_DIR}/")
print(f"fold_1/ … fold_{N_FOLDS}/       per-fold predictions, metrics, confusion matrix")
print(f"best_model/                    best model weights (fold {best_fold_idx})")
print(f"cv_fold_metrics.csv            per-fold metric table")
print(f"cv_summary.csv                 overall CV summary")
print(f"oof_predictions.csv            out-of-fold predictions (full dataset)")
print(f"oof_confusion_matrix.csv       OOF confusion matrix")
if USE_WANDB and wandb_group_url:
    print(f"\nW&B group: {wandb_group_url}")