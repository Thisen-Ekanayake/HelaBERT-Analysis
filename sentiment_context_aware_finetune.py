"""
sentiment_context_aware_finetune.py
BERT fine-tuning for Sentiment Analysis — Stage 2: Context-Aware (Cross-Attention).
Uses Ray Tune ASHA + Optuna HPO (mirrors hpo_lora_classification.py style).

Architecture:
  Article body  → sliding window → BERT per chunk → [CLS] vectors → chunk_matrix [B, C, H]
  Comment text  → BERT encoder  → [CLS] → comment_vec [B, H]
  Cross-Attention: comment queries chunk_matrix → attended_ctx [B, H]
  Fusion: [comment_vec ; attended_ctx ; comment_vec ⊙ attended_ctx]
         → LayerNorm → Dropout → Linear → num_labels
  Both encoders share the same BERT weights (weight-tied).

Logs per-epoch eval metrics (loss, accuracy, F1) to JSONL in real time.
Final summary written to LOG_DIR/<run_name>_summary.json.
Test-set evaluation is run after training with the best checkpoint.

Single-run mode : HPO=0  (default)
HPO mode        : HPO=1  → Ray Tune ASHA + Optuna search
"""

import os
import gc
import math
import json
import time
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    BertConfig,
    BertModel,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    TrainerCallback,
    EarlyStoppingCallback,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune import CLIReporter


# ============================================================
# CONFIG  (override via env vars)
# ============================================================

BERT_MODEL_PATH  = os.environ.get("BERT_MODEL_PATH",  "HelaBERT")
TOKENIZER_MODEL  = os.environ.get("TOKENIZER_MODEL",  "tokenizer/unigram_32000_0.9995.model")
BERT_CONFIG_FILE = os.environ.get("BERT_CONFIG_FILE", "HelaBERT/config.json")

TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA_PATH", "data/sinhala-sentiment-analysis/train.tsv")
TEST_DATA_PATH  = os.environ.get("TEST_DATA_PATH",  "data/sinhala-sentiment-analysis/test.tsv")

# Stage 1 predictions CSV for comparison after training (set None to skip)
STAGE1_PREDICTIONS_CSV = os.environ.get("STAGE1_PREDICTIONS_CSV",
                                         "HelaBERT_sentiment_comments_only/predictions_test.csv")

BODY_COL    = "body"
COMMENT_COL = "comment_phrase"
LABEL_COL   = "comment_sentiment"
STAGE_TAG   = "cross_attention"

OUT_DIR = os.environ.get("OUT_DIR", "output/sentiment_context_aware")
LOG_DIR = os.environ.get("LOG_DIR", os.path.join(OUT_DIR, "logs"))

# Sliding window (architectural — not a tuning hyperparameter)
CHUNK_SIZE   = int(os.environ.get("CHUNK_SIZE",   "512"))
CHUNK_STRIDE = int(os.environ.get("CHUNK_STRIDE", "256"))
MAX_CHUNKS   = int(os.environ.get("MAX_CHUNKS",   "16"))

COMMENT_MAX_LENGTH = int(os.environ.get("COMMENT_MAX_LENGTH", "256"))

CROSS_ATTN_HEADS   = int(os.environ.get("CROSS_ATTN_HEADS",   "8"))
CROSS_ATTN_DROPOUT = float(os.environ.get("CROSS_ATTN_DROPOUT","0.1"))

VAL_SPLIT   = float(os.environ.get("VAL_SPLIT",  "0.1"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED",  "42"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS",   "2"))

# HPO mode: HPO=1 to run Ray Tune search, HPO=0 for single run
HPO_MODE   = bool(int(os.environ.get("HPO",        "0")))
HPO_TRIALS = int(os.environ.get("HPO_TRIALS",      "20"))

# Single-run defaults (ignored when HPO=1)
MICRO_BS     = int(os.environ.get("MICRO_BS",     "8"))
GRAD_ACC     = int(os.environ.get("GRAD_ACC",     "8"))
EPOCHS       = float(os.environ.get("EPOCHS",     "20"))
LR           = float(os.environ.get("LR",         "2e-5"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY","0.05"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO","0.1"))

EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "8"))  # low: each sample has MAX_CHUNKS+1 BERT passes

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "bert-sentiment-analysis")
USE_WANDB     = bool(int(os.environ.get("USE_WANDB", "1")))

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================
# REPRODUCIBILITY
# ============================================================

import random
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ============================================================
# VERIFY PATHS
# ============================================================

assert os.path.exists(BERT_MODEL_PATH), f"❌ {BERT_MODEL_PATH}"
assert os.path.exists(TOKENIZER_MODEL), f"❌ {TOKENIZER_MODEL}"
assert os.path.exists(TRAIN_DATA_PATH), f"❌ {TRAIN_DATA_PATH}"
assert os.path.exists(TEST_DATA_PATH),  f"❌ {TEST_DATA_PATH}"
print("✓ All paths verified")

# ============================================================
# LOAD TOKENIZER (once)
# ============================================================

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)
PAD_ID = sp.pad_id()
print(f"✓ SentencePiece loaded  vocab: {sp.get_piece_size()}  PAD_ID: {PAD_ID}")

# ============================================================
# HELPERS
# ============================================================

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
    Padded with dummy (all-PAD) entries up to max_chunks.
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

# ============================================================
# LOAD DATA (once)
# ============================================================

print("\n" + "="*60)
print("LOADING DATA")
print("="*60)
train_df = load_tsv(TRAIN_DATA_PATH)
test_df  = load_tsv(TEST_DATA_PATH)
print(f"✓ Train: {len(train_df):,}  Test: {len(test_df):,}")

# ============================================================
# ENCODE LABELS (once)
# ============================================================

all_labels = pd.concat([train_df['label'], test_df['label']]).unique()
le = LabelEncoder()
le.fit(sorted(all_labels))
train_df['label_id'] = le.transform(train_df['label'])
test_df['label_id']  = le.transform(test_df['label'])
NUM_LABELS  = len(le.classes_)
id_to_label = {i: lbl for i, lbl in enumerate(le.classes_)}

print(f"✓ {NUM_LABELS} labels: {', '.join(le.classes_)}")
for idx, lbl in sorted(id_to_label.items()):
    tr = (train_df['label_id'] == idx).sum()
    te = (test_df['label_id']  == idx).sum()
    print(f"  [{idx}] {lbl:20s}  train: {tr:5d}  test: {te:5d}")

# ============================================================
# TRAIN / VAL SPLIT (once)
# ============================================================

tr_idx, val_idx = train_test_split(
    range(len(train_df)), test_size=VAL_SPLIT,
    random_state=RANDOM_SEED, stratify=train_df['label_id'].tolist()
)
tr_df  = train_df.iloc[tr_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)
print(f"\n✓ Split — train: {len(tr_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")

# ============================================================
# DATASET
# ============================================================

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
        chunk_ids  = torch.stack([c[0] for c in chunks])
        chunk_mask = torch.stack([c[1] for c in chunks])

        c_ids  = sp.encode(str(row['comment']))[:COMMENT_MAX_LENGTH]
        c_mask = [1] * len(c_ids)
        pad    = COMMENT_MAX_LENGTH - len(c_ids)
        c_ids  += [PAD_ID] * pad
        c_mask += [0]      * pad

        return {
            'chunk_ids':    chunk_ids,
            'chunk_mask':   chunk_mask,
            'num_chunks':   torch.tensor(num_real,              dtype=torch.long),
            'comment_ids':  torch.tensor(c_ids,                 dtype=torch.long),
            'comment_mask': torch.tensor(c_mask,                dtype=torch.long),
            'labels':       torch.tensor(int(row['label_id']),  dtype=torch.long),
        }

train_dataset = CrossAttnDataset(tr_df)
val_dataset   = CrossAttnDataset(val_df)
test_dataset  = CrossAttnDataset(test_df)
print(f"✓ Datasets — train: {len(train_dataset):,}  val: {len(val_dataset):,}  test: {len(test_dataset):,}")

# ============================================================
# CROSS-ATTENTION MODULE
# ============================================================

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
        h = self.num_heads
        d = self.head_dim

        def proj_and_split(linear, x):
            return linear(x).view(B, -1, h, d).transpose(1, 2)

        Q = proj_and_split(self.q_proj, query)
        K = proj_and_split(self.k_proj, context)
        V = proj_and_split(self.v_proj, context)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        weights  = self.dropout(F.softmax(scores, dim=-1))
        attended = torch.matmul(weights, V).squeeze(2)
        attended = attended.transpose(1, 2).contiguous().view(B, -1)
        return self.out_proj(attended)

# ============================================================
# FULL MODEL
# ============================================================

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
        out     = self.bert(
            input_ids=chunk_ids.view(B * C, L),
            attention_mask=chunk_mask.view(B * C, L)
        )
        cls_vecs       = out.last_hidden_state[:, 0, :].view(B, C, -1)
        idx_range      = torch.arange(C, device=chunk_ids.device).unsqueeze(0)
        chunk_pad_mask = idx_range >= num_chunks.unsqueeze(1)
        return cls_vecs, chunk_pad_mask

    def encode_comment(self, comment_ids, comment_mask):
        out = self.bert(input_ids=comment_ids, attention_mask=comment_mask)
        return out.last_hidden_state[:, 0, :]

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

# ============================================================
# COLLATOR + CUSTOM TRAINER
# ============================================================

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

# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred: EvalPrediction):
    preds  = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    return {
        'accuracy':    float(accuracy_score(labels, preds)),
        'precision':   float(precision_score(labels, preds, average='macro',    zero_division=0)),
        'recall':      float(recall_score(labels, preds,    average='macro',    zero_division=0)),
        'f1':          float(f1_score(labels, preds,        average='macro',    zero_division=0)),
        'f1_weighted': float(f1_score(labels, preds,        average='weighted', zero_division=0)),
    }

# ============================================================
# EPOCH JSON LOGGER  (real-time JSONL, mirrors HPO script)
# ============================================================

class EpochJSONLogger(TrainerCallback):
    """
    Writes one JSON record per epoch to  LOG_DIR/<run_name>_epochs.jsonl
    Final summary goes to              LOG_DIR/<run_name>_summary.json
    """

    def __init__(self, run_name: str, log_dir: str):
        self.run_name   = run_name
        self.log_dir    = log_dir
        self.epoch_file = os.path.join(log_dir, f"{run_name}_epochs.jsonl")
        self.history    = []
        self._train_loss_accum = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs and "eval_loss" not in logs:
            self._train_loss_accum.append(logs["loss"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        epoch_record = {
            "epoch":          round(state.epoch, 2),
            "step":           state.global_step,
            "timestamp":      datetime.datetime.utcnow().isoformat(),
            "train_loss_avg": round(float(np.mean(self._train_loss_accum)), 6)
                              if self._train_loss_accum else None,
            **{k: round(float(v), 6) if isinstance(v, float) else v
               for k, v in metrics.items()},
        }
        self.history.append(epoch_record)
        self._train_loss_accum = []

        with open(self.epoch_file, "a") as f:
            f.write(json.dumps(epoch_record) + "\n")

        print(f"\n[EpochLogger] epoch {epoch_record['epoch']} → {epoch_record}")

    def on_train_end(self, args, state, control, **kwargs):
        if not self.history:
            return
        best = max(self.history, key=lambda r: r.get("eval_f1", 0))
        summary = {
            "run_name":   self.run_name,
            "task":       f"sentiment_{STAGE_TAG}",
            "epochs_run": len(self.history),
            "best_epoch": best,
            "all_epochs": self.history,
        }
        summary_path = os.path.join(self.log_dir, f"{self.run_name}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[EpochLogger] Summary saved → {summary_path}")

# ============================================================
# MODEL LOADER
# ============================================================

def load_fresh_model():
    if os.path.exists(BERT_CONFIG_FILE):
        bert_cfg = BertConfig.from_json_file(BERT_CONFIG_FILE)
    else:
        try:
            bert_cfg = BertConfig.from_pretrained(BERT_MODEL_PATH)
        except Exception:
            bert_cfg = None

    try:
        bert_backbone = BertModel.from_pretrained(BERT_MODEL_PATH)
    except Exception as e:
        print(f"  BertModel failed ({e}), extracting from MLM checkpoint...")
        mlm           = BertForMaskedLM.from_pretrained(BERT_MODEL_PATH)
        bert_backbone = mlm.bert
        if bert_cfg is None:
            bert_cfg = mlm.config

    h_size = (bert_cfg.hidden_size if bert_cfg is not None
               else bert_backbone.config.hidden_size)

    assert h_size % CROSS_ATTN_HEADS == 0, (
        f"CROSS_ATTN_HEADS ({CROSS_ATTN_HEADS}) must divide hidden_size ({h_size})."
    )

    return CrossAttnSentimentModel(
        bert=bert_backbone,
        hidden_size=h_size,
        num_labels=NUM_LABELS,
        num_heads=CROSS_ATTN_HEADS,
        attn_dropout=CROSS_ATTN_DROPOUT,
    )

# ============================================================
# CORE TRAIN FUNCTION
# ============================================================

def train(config: dict, run_name: str, report_to: str = "wandb"):
    lr           = config["lr"]
    weight_decay = config["weight_decay"]
    warmup_ratio = config["warmup_ratio"]
    micro_bs     = config["micro_bs"]
    grad_acc     = config["grad_acc"]
    epochs       = config["epochs"]

    model = load_fresh_model()

    run_out = os.path.join(OUT_DIR, run_name)
    os.makedirs(run_out, exist_ok=True)

    with open(os.path.join(run_out, "run_config.json"), "w") as f:
        json.dump({"run_name": run_name, "task": f"sentiment_{STAGE_TAG}", **config}, f, indent=2)

    training_args = TrainingArguments(
        output_dir=run_out,
        run_name=run_name,

        # dtype
        bf16=torch.cuda.is_available(),
        tf32=False,

        # batch / accumulation
        per_device_train_batch_size=micro_bs,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=grad_acc,

        # optimiser
        optim="adamw_torch_fused",
        learning_rate=lr,
        weight_decay=weight_decay,
        max_grad_norm=1.0,

        # schedule
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        num_train_epochs=epochs,

        # eval / save
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        logging_first_step=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,

        # CrossAttnTrainer manages its own DataLoaders
        dataloader_num_workers=0,
        seed=RANDOM_SEED,
        report_to=report_to,
        push_to_hub=False,
    )

    epoch_logger = EpochJSONLogger(run_name=run_name, log_dir=LOG_DIR)

    trainer = CrossAttnTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            epoch_logger,
            EarlyStoppingCallback(early_stopping_patience=2),
        ],
    )

    trainer.train()

    # Evaluate on held-out test set (best model loaded automatically)
    try:
        test_output  = trainer.predict(test_dataset)
        y_pred       = np.argmax(test_output.predictions, axis=-1)
        y_true       = test_df['label_id'].values
        test_metrics = {
            'accuracy':    float(accuracy_score(y_true, y_pred)),
            'precision':   float(precision_score(y_true, y_pred, average='macro',    zero_division=0)),
            'recall':      float(recall_score(y_true, y_pred,    average='macro',    zero_division=0)),
            'f1':          float(f1_score(y_true, y_pred,        average='macro',    zero_division=0)),
            'f1_weighted': float(f1_score(y_true, y_pred,        average='weighted', zero_division=0)),
        }
        print(f"  test f1={test_metrics['f1']:.4f}  acc={test_metrics['accuracy']:.4f}")
        if report_to == "wandb":
            wandb.log({f"test/{k}": v for k, v in test_metrics.items()})

        test_metrics_path = os.path.join(run_out, "test_metrics.json")
        with open(test_metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"  ✓ Test metrics saved → {test_metrics_path}")
    except Exception as exc:
        print(f"  ⚠️  Test evaluation failed: {exc}")

    best_metric = max(
        (r.get("eval_f1", 0) for r in epoch_logger.history),
        default=0.0,
    )

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_metric

# ============================================================
# RAY TUNE WRAPPER
# ============================================================

def ray_train_fn(ray_config):
    ts = int(time.time())
    run_name = (
        f"sentiment_{STAGE_TAG}"
        f"_lr{ray_config['lr']:.0e}"
        f"_bs{ray_config['micro_bs']}"
        f"_wr{ray_config['warmup_ratio']}"
        f"_{ts}"
    )
    best_f1 = train(ray_config, run_name=run_name, report_to="none")
    tune.report({"eval_f1": best_f1})

# ============================================================
# HPO
# ============================================================

def run_hpo():
    print(f"\n{'='*60}")
    print(f"HPO MODE  |  TASK=sentiment_{STAGE_TAG}  TRIALS={HPO_TRIALS}")
    print(f"{'='*60}\n")

    search_space = {
        "lr":           tune.loguniform(5e-6, 5e-5),
        "weight_decay": tune.choice([0.01, 0.05, 0.1]),
        "warmup_ratio": tune.choice([0.05, 0.1, 0.15, 0.2]),
        "micro_bs":     tune.choice([4, 8]),        # kept small for cross-attn memory
        "grad_acc":     tune.choice([4, 8, 16]),
        "epochs":       tune.choice([10, 15, 20]),
    }

    scheduler = ASHAScheduler(
        metric="eval_f1",
        mode="max",
        max_t=20,
        grace_period=3,
        reduction_factor=2,
    )

    search_algo = OptunaSearch(metric="eval_f1", mode="max")

    reporter = CLIReporter(
        metric_columns=["eval_f1", "training_iteration"],
        max_progress_rows=10,
    )

    analysis = tune.run(
        ray_train_fn,
        config=search_space,
        num_samples=HPO_TRIALS,
        scheduler=scheduler,
        search_alg=search_algo,
        progress_reporter=reporter,
        resources_per_trial={"gpu": 1.0, "cpu": NUM_WORKERS},
        storage_path=os.path.join(OUT_DIR, "ray_results"),
        name=f"hpo_sentiment_{STAGE_TAG}",
        verbose=1,
    )

    best_cfg   = analysis.get_best_config(metric="eval_f1", mode="max")
    best_trial = analysis.get_best_trial(metric="eval_f1",  mode="max")

    hpo_summary = {
        "task":         f"sentiment_{STAGE_TAG}",
        "num_trials":   HPO_TRIALS,
        "best_config":  best_cfg,
        "best_eval_f1": best_trial.last_result["eval_f1"],
        "timestamp":    datetime.datetime.utcnow().isoformat(),
    }

    hpo_path = os.path.join(LOG_DIR, f"hpo_sentiment_{STAGE_TAG}_results.json")
    with open(hpo_path, "w") as f:
        json.dump(hpo_summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"HPO COMPLETE")
    print(f"Best config  : {best_cfg}")
    print(f"Best eval_f1 : {best_trial.last_result['eval_f1']:.4f}")
    print(f"Results saved: {hpo_path}")
    print(f"{'='*60}\n")

    return best_cfg

# ============================================================
# SINGLE RUN
# ============================================================

def run_single():
    config = {
        "lr":           LR,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "micro_bs":     MICRO_BS,
        "grad_acc":     GRAD_ACC,
        "epochs":       EPOCHS,
    }

    ts       = int(time.time())
    run_name = f"sentiment_{STAGE_TAG}_lr{LR:.0e}_bs{MICRO_BS}_{ts}"

    print(f"\n{'='*60}")
    print(f"SINGLE RUN  |  TASK=sentiment_{STAGE_TAG}")
    print(f"Config: {config}")
    print(f"Run name: {run_name}")
    print(f"{'='*60}\n")

    if USE_WANDB:
        wandb.init(project=WANDB_PROJECT, name=run_name)

    best_f1 = train(config, run_name=run_name,
                    report_to="wandb" if USE_WANDB else "none")
    print(f"\nBest eval F1: {best_f1:.4f}")

    if USE_WANDB:
        wandb.finish()

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    if HPO_MODE:
        best_config = run_hpo()

        retrain = bool(int(os.environ.get("RETRAIN_BEST", "1")))
        if retrain:
            print("\nRe-training with best config at full budget...")
            best_config["epochs"] = float(os.environ.get("RETRAIN_EPOCHS", "20"))
            ts       = int(time.time())
            run_name = f"sentiment_{STAGE_TAG}_best_{ts}"
            if USE_WANDB:
                wandb.init(project=WANDB_PROJECT, name=run_name)
            best_f1 = train(best_config, run_name=run_name,
                            report_to="wandb" if USE_WANDB else "none")
            print(f"Final best model F1: {best_f1:.4f}")
            if USE_WANDB:
                wandb.finish()
    else:
        run_single()