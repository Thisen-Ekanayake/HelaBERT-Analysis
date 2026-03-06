"""
BERT Fine-tuning for Sentiment Analysis — Stage 2: Context-Aware (Cross-Attention)
Hyperparameter Grid Search

Architecture (unchanged from single-run script):
  Article body  → sliding window → BERT per chunk → [CLS] vectors → chunk_matrix [B, C, H]
  Comment text  → BERT encoder  → [CLS] → comment_vec [B, H]
  Cross-Attention: comment queries chunk_matrix → attended_ctx [B, H]
  Fusion: [comment_vec ; attended_ctx ; comment_vec ⊙ attended_ctx]
         → LayerNorm → Dropout → Linear → num_labels
  Both encoders share the same BERT weights (weight-tied).

Runs all combinations of:
    TRAIN_BATCH_SIZE : [8, 16, 32, 64]
    LEARNING_RATE    : [5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
    WARMUP_RATIO     : [0.05, 0.01, 0.15, 0.2, 0.25, 0.3]
    NUM_EPOCHS       : 20  (fixed)

Total runs: 4 × 10 × 6 = 240

Per-epoch val metrics + final test metrics are written to:
    results/sentiment_context_aware_finetune/<run_name>.json

Note: Cross-attention inspection and Stage 1 vs Stage 2 comparison are
skipped per-run (too slow for a grid). Run them manually on the best
checkpoint after the grid completes.
"""

import os
import gc
import math
import json
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    BertConfig, BertModel, BertForMaskedLM,
    Trainer, TrainingArguments, EvalPrediction, TrainerCallback,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import random
import wandb


# ==================== FIXED CONFIGURATION ====================
BERT_MODEL_PATH  = "HelaBERT"
TOKENIZER_MODEL  = "tokenizer/unigram_32000_0.9995.model"
BERT_CONFIG_FILE = "HelaBERT/config.json"

TRAIN_DATA_PATH = "data/sinhala-sentiment-analysis/train.tsv"
TEST_DATA_PATH  = "data/sinhala-sentiment-analysis/test.tsv"

# Stage 1 predictions CSV for end-of-grid comparison (set None to skip)
STAGE1_PREDICTIONS_CSV = "HelaBERT_sentiment_comments_only/predictions_test.csv"

# TSV column names
BODY_COL    = "body"
COMMENT_COL = "comment_phrase"
LABEL_COL   = "comment_sentiment"

STAGE_TAG = "cross_attention"

# Sliding window (fixed — architectural, not a tuning hyperparameter)
CHUNK_SIZE   = 512
CHUNK_STRIDE = 256
MAX_CHUNKS   = 16

# Sequence lengths
COMMENT_MAX_LENGTH = 256

# Cross-attention (fixed)
CROSS_ATTN_HEADS   = 8
CROSS_ATTN_DROPOUT = 0.1

# Training (fixed across grid)
NUM_EPOCHS                  = 20
WEIGHT_DECAY                = 0.05
GRADIENT_ACCUMULATION_STEPS = 8    # effective batch = TRAIN_BATCH_SIZE * 8
VAL_SPLIT                   = 0.1
EVAL_BATCH_SIZE_FIXED       = 8    # low: each sample has MAX_CHUNKS+1 BERT passes
RANDOM_SEED                 = 42
USE_FP16                    = True
NUM_WORKERS                 = 2

USE_WANDB     = True
WANDB_PROJECT = "bert-sentiment-analysis"
WANDB_ENTITY  = None

# ==================== GRID ====================
TRAIN_BATCH_SIZES = [8, 16, 32, 64]
LEARNING_RATES    = [5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
WARMUP_RATIOS     = [0.05, 0.01, 0.15, 0.2, 0.25, 0.3]

# Results output dir
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
RESULTS_DIR = os.path.join("results", SCRIPT_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== REPRODUCIBILITY ====================
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print("=" * 80)
print("BERT SENTIMENT STAGE 2 (CROSS-ATTENTION) — HYPERPARAMETER GRID SEARCH")
print("=" * 80)
print(f"Grid: {len(TRAIN_BATCH_SIZES)} batch sizes × "
      f"{len(LEARNING_RATES)} learning rates × "
      f"{len(WARMUP_RATIOS)} warmup ratios = "
      f"{len(TRAIN_BATCH_SIZES)*len(LEARNING_RATES)*len(WARMUP_RATIOS)} runs")
print(f"Epochs per run       : {NUM_EPOCHS}")
print(f"Grad accum steps     : {GRADIENT_ACCUMULATION_STEPS}  (effective batch = bs × {GRADIENT_ACCUMULATION_STEPS})")
print(f"Chunk size / stride  : {CHUNK_SIZE} / {CHUNK_STRIDE}  max chunks: {MAX_CHUNKS}")
print(f"Cross-attn heads     : {CROSS_ATTN_HEADS}")
print(f"Results dir          : {RESULTS_DIR}/")
print()


# ==================== VERIFY PATHS ====================
assert os.path.exists(BERT_MODEL_PATH), f"❌ {BERT_MODEL_PATH}"
assert os.path.exists(TOKENIZER_MODEL), f"❌ {TOKENIZER_MODEL}"
assert os.path.exists(TRAIN_DATA_PATH), f"❌ {TRAIN_DATA_PATH}"
assert os.path.exists(TEST_DATA_PATH),  f"❌ {TEST_DATA_PATH}"
print("✓ All paths verified")


# ==================== TOKENIZER ====================
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)
PAD_ID = sp.pad_id()
print(f"✓ SentencePiece loaded  vocab: {sp.get_piece_size()}  PAD_ID: {PAD_ID}")


# ==================== HELPERS ====================
def find_col(df, name):
    if name in df.columns:
        return name
    hits = [c for c in df.columns if name.lower() in c.lower()]
    if hits:
        return hits[0]
    raise KeyError(f"Column '{name}' not found in {list(df.columns)}")


def load_tsv(path):
    try:
        df = pd.read_csv(path, sep='\t')
    except pd.errors.ParserError:
        df = pd.read_csv(path, sep='\t', engine='python', on_bad_lines='skip')
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
        chunks.append((torch.tensor(seg,  dtype=torch.long),
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


# ==================== LOAD DATA (once) ====================
print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)
train_df = load_tsv(TRAIN_DATA_PATH)
test_df  = load_tsv(TEST_DATA_PATH)
print(f"✓ Train: {len(train_df):,}  Test: {len(test_df):,}")


# ==================== ENCODE LABELS (once) ====================
print("\n" + "=" * 80)
print("ENCODING LABELS")
print("=" * 80)
all_labels = pd.concat([train_df['label'], test_df['label']]).unique()
le = LabelEncoder()
le.fit(sorted(all_labels))
train_df['label_id'] = le.transform(train_df['label'])
test_df['label_id']  = le.transform(test_df['label'])
NUM_LABELS  = len(le.classes_)
id_to_label = {i: lbl for i, lbl in enumerate(le.classes_)}

mapping_df = pd.DataFrame({'label_id':   list(id_to_label.keys()),
                            'label_name': list(id_to_label.values())})

print(f"✓ {NUM_LABELS} labels: {', '.join(le.classes_)}")
for idx, lbl in sorted(id_to_label.items()):
    tr = (train_df['label_id'] == idx).sum()
    te = (test_df['label_id']  == idx).sum()
    print(f"  [{idx}] {lbl:20s}  train: {tr:5d}  test: {te:5d}")


# ==================== TRAIN / VAL SPLIT (once) ====================
tr_idx, val_idx = train_test_split(
    range(len(train_df)), test_size=VAL_SPLIT,
    random_state=RANDOM_SEED, stratify=train_df['label_id'].tolist()
)
tr_df  = train_df.iloc[tr_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)
print(f"\n✓ Split — train: {len(tr_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")


# ==================== DATASET ====================
class CrossAttnDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        chunks, num_real = tokenize_chunks(
            row['body'], CHUNK_SIZE, CHUNK_STRIDE, MAX_CHUNKS
        )
        chunk_ids  = torch.stack([c[0] for c in chunks])   # [MAX_CHUNKS, CHUNK_SIZE]
        chunk_mask = torch.stack([c[1] for c in chunks])   # [MAX_CHUNKS, CHUNK_SIZE]

        c_ids  = sp.encode(str(row['comment']))[:COMMENT_MAX_LENGTH]
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
            'labels':       torch.tensor(int(row['label_id']), dtype=torch.long),
        }


train_dataset = CrossAttnDataset(tr_df)
val_dataset   = CrossAttnDataset(val_df)
test_dataset  = CrossAttnDataset(test_df)
print(f"✓ Datasets — train: {len(train_dataset):,}  val: {len(val_dataset):,}  test: {len(test_dataset):,}")


# ==================== CROSS-ATTENTION MODULE ====================
class MultiHeadCrossAttention(nn.Module):
    """
    Comment queries article chunks.
      Query  : comment_vec  [B, 1, H]
      Key/Val: chunk_vecs   [B, C, H]
      Output : attended_ctx [B, H]
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

        weights  = self.dropout(F.softmax(scores, dim=-1))          # [B, h, 1, C]
        attended = torch.matmul(weights, V).squeeze(2)              # [B, h, d]
        attended = attended.transpose(1, 2).contiguous().view(B, -1) # [B, H]
        return self.out_proj(attended)


# ==================== FULL MODEL ====================
class CrossAttnSentimentModel(nn.Module):
    """
    Shared BERT + multi-head cross-attention + interaction fusion.
    Fusion = [comment_vec ; attended_ctx ; comment_vec ⊙ attended_ctx]
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


# ==================== COLLATOR + CUSTOM TRAINER ====================
def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


class CrossAttnTrainer(Trainer):
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


# ==================== PER-EPOCH JSON CALLBACK ====================
class EpochJsonLogger(TrainerCallback):
    """
    Accumulates per-epoch val metrics, then writes JSON after training
    (with test metrics appended via write()).
    """
    def __init__(self, save_path: str, run_config: dict):
        self.save_path  = save_path
        self.run_config = run_config
        self.epochs     = []
        self._pending   = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if 'loss' in logs and 'eval_loss' not in logs:
            self._pending['train_loss']    = logs.get('loss')
            self._pending['learning_rate'] = logs.get('learning_rate')
            self._pending['global_step']   = state.global_step

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        self.epochs.append({
            'epoch':            metrics.get('epoch', state.epoch),
            'eval_loss':        metrics.get('eval_loss'),
            'eval_accuracy':    metrics.get('eval_accuracy'),
            'eval_precision':   metrics.get('eval_precision'),
            'eval_recall':      metrics.get('eval_recall'),
            'eval_f1':          metrics.get('eval_f1'),
            'eval_f1_weighted': metrics.get('eval_f1_weighted'),
            'train_loss':       self._pending.get('train_loss'),
            'learning_rate':    self._pending.get('learning_rate'),
            'global_step':      state.global_step,
        })

    def write(self, test_metrics: dict = None):
        output = {
            'run_config':        self.run_config,
            'total_epochs':      len(self.epochs),
            'per_epoch_metrics': self.epochs,
            'test_metrics':      test_metrics or {},
        }
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Metrics saved → {self.save_path}")

    def on_train_end(self, args, state, control, **kwargs):
        if not os.path.exists(self.save_path):
            self.write()


# ==================== MODEL LOADER ====================
def load_fresh_model(bert_config_file, bert_model_path,
                     num_labels, num_heads, attn_dropout):
    """Load a fresh BERT backbone and wrap it in CrossAttnSentimentModel."""
    if os.path.exists(bert_config_file):
        bert_cfg = BertConfig.from_json_file(bert_config_file)
    else:
        try:
            bert_cfg = BertConfig.from_pretrained(bert_model_path)
        except Exception:
            bert_cfg = None

    try:
        bert_backbone = BertModel.from_pretrained(bert_model_path)
    except Exception as e:
        print(f"  BertModel failed ({e}), extracting from MLM checkpoint...")
        mlm           = BertForMaskedLM.from_pretrained(bert_model_path)
        bert_backbone = mlm.bert
        if bert_cfg is None:
            bert_cfg = mlm.config

    h_size = (bert_cfg.hidden_size if bert_cfg is not None
               else bert_backbone.config.hidden_size)

    assert h_size % num_heads == 0, (
        f"CROSS_ATTN_HEADS ({num_heads}) must divide hidden_size ({h_size})."
    )

    mdl = CrossAttnSentimentModel(
        bert=bert_backbone,
        hidden_size=h_size,
        num_labels=num_labels,
        num_heads=num_heads,
        attn_dropout=attn_dropout,
    )
    return mdl, bert_cfg, h_size


# ==================== GRID SEARCH LOOP ====================
grid = list(itertools.product(TRAIN_BATCH_SIZES, LEARNING_RATES, WARMUP_RATIOS))
total_runs = len(grid)

print(f"\n{'='*80}")
print(f"STARTING GRID SEARCH — {total_runs} runs")
print(f"{'='*80}\n")

completed = 0
skipped   = 0

for run_idx, (bs, lr, wr) in enumerate(grid, start=1):

    run_name  = f"bs{bs}_lr{lr:.0e}_wr{wr}_ep{NUM_EPOCHS}"
    json_path = os.path.join(RESULTS_DIR, f"{run_name}.json")

    # ---- skip already-completed runs (safe to resume) ----
    if os.path.exists(json_path):
        print(f"[{run_idx:3d}/{total_runs}] SKIP  {run_name}  (json exists)")
        skipped += 1
        continue

    print(f"\n[{run_idx:3d}/{total_runs}] START {run_name}")
    print(f"  batch_size={bs}  lr={lr}  warmup_ratio={wr}  epochs={NUM_EPOCHS}"
          f"  effective_batch={bs * GRADIENT_ACCUMULATION_STEPS}")

    run_config = {
        'run_index':               run_idx,
        'stage':                   STAGE_TAG,
        'train_batch_size':        bs,
        'effective_batch_size':    bs * GRADIENT_ACCUMULATION_STEPS,
        'learning_rate':           lr,
        'warmup_ratio':            wr,
        'num_epochs':              NUM_EPOCHS,
        'weight_decay':            WEIGHT_DECAY,
        'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
        'comment_max_length':      COMMENT_MAX_LENGTH,
        'chunk_size':              CHUNK_SIZE,
        'chunk_stride':            CHUNK_STRIDE,
        'max_chunks':              MAX_CHUNKS,
        'cross_attn_heads':        CROSS_ATTN_HEADS,
        'cross_attn_dropout':      CROSS_ATTN_DROPOUT,
        'val_split':               VAL_SPLIT,
        'num_labels':              NUM_LABELS,
        'label_names':             list(le.classes_),
        'train_samples':           len(tr_df),
        'val_samples':             len(val_df),
        'test_samples':            len(test_df),
        'random_seed':             RANDOM_SEED,
    }

    output_dir = f"checkpoints/{SCRIPT_NAME}/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    # -- W&B run --
    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=f"{STAGE_TAG}_{run_name}",
            config=run_config,
            reinit=True,
        )

    # -- fresh model --
    model, bert_cfg, hidden_size = load_fresh_model(
        BERT_CONFIG_FILE, BERT_MODEL_PATH,
        NUM_LABELS, CROSS_ATTN_HEADS, CROSS_ATTN_DROPOUT
    )

    # -- training args --
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=EVAL_BATCH_SIZE_FIXED,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=wr,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        fp16=USE_FP16 and torch.cuda.is_available(),
        dataloader_num_workers=0,   # CrossAttnTrainer manages its own workers
        seed=RANDOM_SEED,
        report_to="wandb" if USE_WANDB else "none",
        run_name=f"{STAGE_TAG}_{run_name}" if USE_WANDB else None,
        push_to_hub=False,
    )

    epoch_logger = EpochJsonLogger(save_path=json_path, run_config=run_config)

    trainer = CrossAttnTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[epoch_logger],
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted at run {run_idx}. Saving partial results...")
        epoch_logger.write()
        if USE_WANDB:
            wandb.finish(exit_code=1)
        raise
    except Exception as exc:
        print(f"  ❌ Run {run_idx} failed during training: {exc}")
        epoch_logger.write()
        if USE_WANDB:
            wandb.finish(exit_code=1)
        del model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        continue

    # -- evaluate on held-out test set (best model loaded automatically) --
    try:
        test_output  = trainer.predict(test_dataset)
        y_pred       = np.argmax(test_output.predictions, axis=-1)
        y_true       = test_df['label_id'].values
        test_metrics = {
            'accuracy':    accuracy_score(y_true, y_pred),
            'precision':   precision_score(y_true, y_pred, average='macro',    zero_division=0),
            'recall':      recall_score(y_true, y_pred,    average='macro',    zero_division=0),
            'f1':          f1_score(y_true, y_pred,        average='macro',    zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred,        average='weighted', zero_division=0),
        }
        print(f"  test f1={test_metrics['f1']:.4f}  acc={test_metrics['accuracy']:.4f}")

        if USE_WANDB:
            wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
    except Exception as exc:
        print(f"  ⚠️  Test evaluation failed: {exc}")
        test_metrics = {}

    # -- write JSON --
    epoch_logger.write(test_metrics=test_metrics)
    completed += 1
    print(f"  ✓ Run {run_idx} complete")

    if USE_WANDB:
        wandb.finish()

    # -- free GPU memory between runs --
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==================== GRID SEARCH SUMMARY ====================
print("\n" + "=" * 80)
print("GRID SEARCH COMPLETE")
print("=" * 80)
print(f"  Total runs   : {total_runs}")
print(f"  Completed    : {completed}")
print(f"  Skipped      : {skipped}")
print(f"  Results dir  : {RESULTS_DIR}/")
print()

# Build a summary CSV from all result JSONs
summary_rows = []
for fname in sorted(os.listdir(RESULTS_DIR)):
    if not fname.endswith('.json'):
        continue
    fpath = os.path.join(RESULTS_DIR, fname)
    try:
        with open(fpath) as f:
            data = json.load(f)
        cfg  = data['run_config']
        epcs = data['per_epoch_metrics']
        last = epcs[-1] if epcs else {}
        best_val = max(epcs, key=lambda e: e.get('eval_f1') or 0, default={})
        tm   = data.get('test_metrics', {})
        summary_rows.append({
            'run_file':           fname,
            'batch_size':         cfg['train_batch_size'],
            'effective_batch':    cfg['effective_batch_size'],
            'learning_rate':      cfg['learning_rate'],
            'warmup_ratio':       cfg['warmup_ratio'],
            'num_epochs':         cfg['num_epochs'],
            # best val epoch
            'best_val_epoch':     best_val.get('epoch'),
            'best_val_f1':        best_val.get('eval_f1'),
            'best_val_acc':       best_val.get('eval_accuracy'),
            'best_val_loss':      best_val.get('eval_loss'),
            'best_val_f1_w':      best_val.get('eval_f1_weighted'),
            # final val epoch
            'final_val_f1':       last.get('eval_f1'),
            'final_val_acc':      last.get('eval_accuracy'),
            'final_val_loss':     last.get('eval_loss'),
            # test set (held-out)
            'test_f1':            tm.get('f1'),
            'test_f1_weighted':   tm.get('f1_weighted'),
            'test_accuracy':      tm.get('accuracy'),
            'test_precision':     tm.get('precision'),
            'test_recall':        tm.get('recall'),
        })
    except Exception as e:
        print(f"  ⚠️  Could not read {fname}: {e}")

if summary_rows:
    summary_df = pd.DataFrame(summary_rows).sort_values('test_f1', ascending=False)
    summary_csv = os.path.join(RESULTS_DIR, "grid_search_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Summary CSV written: {summary_csv}")
    print("\nTop-5 runs by test F1:")
    print(summary_df.head(5).to_string(index=False))


# ==================== OPTIONAL: STAGE 1 vs STAGE 2 COMPARISON ====================
# Runs once after all grid runs finish, comparing the best Stage 2 run
# against Stage 1 results (if available).
print("\n" + "=" * 80)
print("STAGE 1 vs STAGE 2 COMPARISON  (best grid run)")
print("=" * 80)

if summary_rows:
    best_row = summary_df.iloc[0]
    s2_metrics = {
        'accuracy':    best_row.get('test_accuracy'),
        'precision':   best_row.get('test_precision'),
        'recall':      best_row.get('test_recall'),
        'f1':          best_row.get('test_f1'),
        'f1_weighted': best_row.get('test_f1_weighted'),
    }
    print(f"\nBest Stage 2 run : {best_row['run_file']}")
    print(f"  test_f1 = {s2_metrics['f1']:.4f}  test_acc = {s2_metrics['accuracy']:.4f}")

    stage1_metrics = None
    if STAGE1_PREDICTIONS_CSV and os.path.exists(STAGE1_PREDICTIONS_CSV):
        try:
            s1 = pd.read_csv(STAGE1_PREDICTIONS_CSV)
            s1_true = s1['true_label_id'].values
            s1_pred = s1['predicted_label_id'].values
            stage1_metrics = {
                'accuracy':    accuracy_score(s1_true, s1_pred),
                'precision':   precision_score(s1_true, s1_pred, average='macro',    zero_division=0),
                'recall':      recall_score(s1_true, s1_pred,    average='macro',    zero_division=0),
                'f1':          f1_score(s1_true, s1_pred,        average='macro',    zero_division=0),
                'f1_weighted': f1_score(s1_true, s1_pred,        average='weighted', zero_division=0),
            }
            print(f"\nStage 1 results loaded from: {STAGE1_PREDICTIONS_CSV}")
        except Exception as e:
            print(f"⚠️  Could not load Stage 1 results: {e}")
    else:
        print(f"\n⚠️  STAGE1_PREDICTIONS_CSV not found — set it to enable comparison.")

    metrics_order = ['accuracy', 'precision', 'recall', 'f1', 'f1_weighted']
    W = 22
    print()
    print(f"  {'Metric':18s}  {'Stage 1':>{W}}  {'Stage 2 (best)':>{W}}"
          + (f"  {'Δ (S2-S1)':>{W}}" if stage1_metrics else ""))
    print("  " + "-" * (80 if stage1_metrics else 60))
    for m in metrics_order:
        s2 = s2_metrics.get(m)
        if s2 is None:
            continue
        if stage1_metrics:
            s1    = stage1_metrics[m]
            delta = s2 - s1
            sign  = "+" if delta >= 0 else ""
            print(f"  {m:18s}  {s1:>{W}.4f}  {s2:>{W}.4f}  {sign}{delta:>{W-1}.4f}")
        else:
            print(f"  {m:18s}  {'N/A':>{W}}  {s2:>{W}.4f}")

    if stage1_metrics:
        comp_rows = [{'metric': m,
                      'stage1_comments_only':   stage1_metrics[m],
                      'stage2_cross_attention': s2_metrics[m],
                      'delta': s2_metrics[m] - stage1_metrics[m]}
                     for m in metrics_order if s2_metrics.get(m) is not None]
        comp_csv = os.path.join(RESULTS_DIR, "stage_comparison.csv")
        pd.DataFrame(comp_rows).to_csv(comp_csv, index=False)
        print(f"\n  ✓ Comparison saved to {comp_csv}")

        delta_f1 = s2_metrics['f1'] - stage1_metrics['f1']
        if delta_f1 > 0.01:
            print(f"\n  ✅ Cross-attention context improves macro-F1 by +{delta_f1:.4f}")
        elif delta_f1 < -0.01:
            print(f"\n  ⚠️  Baseline is better by {abs(delta_f1):.4f} — "
                  f"try more epochs or reduce MAX_CHUNKS")
        else:
            print(f"\n  ↔️  Roughly equivalent (Δ macro-F1 = {delta_f1:+.4f})")

print("\n" + "=" * 80)
print("🎉 ALL DONE!")
print("=" * 80)
print(f"\nResults in: {RESULTS_DIR}/")
print(f"  grid_search_summary.csv   — all 240 runs ranked by test F1")
print(f"  stage_comparison.csv      — Stage 1 vs best Stage 2 (if Stage 1 CSV provided)")
print(f"\nNote: Cross-attention weight inspection can be run manually on the best")
print(f"      checkpoint found in checkpoints/{SCRIPT_NAME}/<best_run_name>/")
print("=" * 80)