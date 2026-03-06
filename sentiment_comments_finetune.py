"""
BERT Fine-tuning for Sentiment Analysis — Comments Only — Hyperparameter Grid Search

Runs all combinations of:
    TRAIN_BATCH_SIZE : [8, 16, 32, 64]
    LEARNING_RATE    : [5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
    WARMUP_RATIO     : [0.05, 0.01, 0.15, 0.2, 0.25, 0.3]
    NUM_EPOCHS       : 20  (fixed)

Total runs: 4 × 10 × 6 = 240

Per-epoch val metrics + final test metrics are written to:
    results/sentiment_comments_finetune/<run_name>.json

TSV columns used:
    comment_phrase    → input text
    comment_sentiment → label  (e.g. POSITIVE, NEGATIVE, NEUTRAL, ...)

Separate train/test TSV files are expected; a small validation split is
carved from train for early stopping / per-epoch logging.
"""

import os
import gc
import json
import itertools
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import sentencepiece as spm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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
    BertForSequenceClassification,
    BertForMaskedLM,
    BertModel,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    TrainerCallback,
)
import random
import wandb


# ==================== FIXED CONFIGURATION ====================
BERT_MODEL_PATH  = "HelaBERT"
TOKENIZER_MODEL  = "tokenizer/unigram_32000_0.9995.model"
BERT_CONFIG_FILE = "HelaBERT/config.json"

TRAIN_DATA_PATH  = "data/sinhala-sentiment-analysis/train.tsv"
TEST_DATA_PATH   = "data/sinhala-sentiment-analysis/test.tsv"

COMMENT_COL = "comment_phrase"
LABEL_COL   = "comment_sentiment"

STAGE_TAG   = "comments_only"

MAX_LENGTH                  = 256
VAL_SPLIT                   = 0.1    # fraction of train carved out for validation
WEIGHT_DECAY                = 0.05
GRADIENT_ACCUMULATION_STEPS = 1
EVAL_BATCH_SIZE_FIXED       = 64     # constant eval batch across all runs
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
NUM_EPOCHS        = 20   # fixed

# Results output dir
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]   # "sentiment_comments_finetune"
RESULTS_DIR = os.path.join("results", SCRIPT_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== REPRODUCIBILITY ====================
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print("=" * 80)
print("BERT SENTIMENT (COMMENTS ONLY) — HYPERPARAMETER GRID SEARCH")
print("=" * 80)
print(f"Grid: {len(TRAIN_BATCH_SIZES)} batch sizes × "
      f"{len(LEARNING_RATES)} learning rates × "
      f"{len(WARMUP_RATIOS)} warmup ratios = "
      f"{len(TRAIN_BATCH_SIZES)*len(LEARNING_RATES)*len(WARMUP_RATIOS)} runs")
print(f"Epochs per run : {NUM_EPOCHS}")
print(f"Results dir    : {RESULTS_DIR}/")
print()


# ==================== VERIFY PATHS ====================
assert os.path.exists(BERT_MODEL_PATH), f"❌ Model not found: {BERT_MODEL_PATH}"
assert os.path.exists(TOKENIZER_MODEL), f"❌ Tokenizer not found: {TOKENIZER_MODEL}"
assert os.path.exists(TRAIN_DATA_PATH), f"❌ Train file not found: {TRAIN_DATA_PATH}"
assert os.path.exists(TEST_DATA_PATH),  f"❌ Test file not found: {TEST_DATA_PATH}"
print("✓ All paths verified")


# ==================== LOAD TOKENIZER ====================
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)
PAD_ID = sp.pad_id()
print(f"✓ SentencePiece tokenizer loaded  (vocab size: {sp.get_piece_size()}  |  PAD_ID: {PAD_ID})")


# ==================== HELPER: LOAD TSV ====================
def load_tsv(path: str, comment_col: str, label_col: str) -> pd.DataFrame:
    """Load a TSV, clean column names, extract comment + label columns."""
    try:
        df = pd.read_csv(path, sep='\t')
    except pd.errors.ParserError:
        print(f"  ⚠️  Parsing issues in {path}, retrying with python engine...")
        df = pd.read_csv(path, sep='\t', engine='python', on_bad_lines='skip')

    df.columns = df.columns.str.strip()

    def find_col(df, preferred):
        if preferred in df.columns:
            return preferred
        matches = [c for c in df.columns if preferred.lower() in c.lower()]
        if matches:
            return matches[0]
        raise KeyError(f"Could not find column '{preferred}' in {list(df.columns)}")

    c_col = find_col(df, comment_col)
    l_col = find_col(df, label_col)

    df = df[[c_col, l_col]].rename(columns={c_col: 'comment', l_col: 'label'})
    df = df.dropna(subset=['comment', 'label'])
    df['comment'] = df['comment'].astype(str).str.strip()
    df['label']   = df['label'].astype(str).str.strip().str.upper()
    df = df[df['comment'].str.len() > 0]
    return df


# ==================== LOAD DATA (once) ====================
print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

train_df = load_tsv(TRAIN_DATA_PATH, COMMENT_COL, LABEL_COL)
test_df  = load_tsv(TEST_DATA_PATH,  COMMENT_COL, LABEL_COL)

print(f"✓ Train TSV loaded  →  {len(train_df):,} rows")
print(f"✓ Test  TSV loaded  →  {len(test_df):,} rows")


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
label_to_id = {lbl: i for i, lbl in id_to_label.items()}

print(f"✓ {NUM_LABELS} unique sentiment labels: {list(le.classes_)}")
for idx, lbl in sorted(id_to_label.items()):
    tr = (train_df['label_id'] == idx).sum()
    te = (test_df['label_id']  == idx).sum()
    print(f"  [{idx}] {lbl:20s}  train: {tr:5d}  test: {te:5d}")


# ==================== TRAIN / VALIDATION SPLIT (once) ====================
train_comments = train_df['comment'].tolist()
train_labels   = train_df['label_id'].tolist()

tr_texts, val_texts, tr_labels, val_labels = train_test_split(
    train_comments, train_labels,
    test_size=VAL_SPLIT,
    random_state=RANDOM_SEED,
    stratify=train_labels
)

test_comments = test_df['comment'].tolist()
test_labels   = test_df['label_id'].tolist()

print(f"\n✓ Split — train: {len(tr_texts):,}  |  val: {len(val_texts):,}  |  test: {len(test_comments):,}")


# ==================== DATASET CLASS ====================
class CommentDataset(Dataset):
    """Tokenise a comment and return input_ids + attention_mask + label."""

    def __init__(self, texts, labels, sp_processor, max_length):
        self.texts      = texts
        self.labels     = labels
        self.sp         = sp_processor
        self.max_length = max_length
        self.pad_id     = sp_processor.pad_id()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids  = self.sp.encode(self.texts[idx])[:self.max_length]
        mask = [1] * len(ids)
        pad  = self.max_length - len(ids)
        ids  += [self.pad_id] * pad
        mask += [0] * pad
        return {
            'input_ids':      torch.tensor(ids,               dtype=torch.long),
            'attention_mask': torch.tensor(mask,              dtype=torch.long),
            'labels':         torch.tensor(self.labels[idx],  dtype=torch.long),
        }


train_dataset = CommentDataset(tr_texts,      tr_labels,   sp, MAX_LENGTH)
val_dataset   = CommentDataset(val_texts,     val_labels,  sp, MAX_LENGTH)
test_dataset  = CommentDataset(test_comments, test_labels, sp, MAX_LENGTH)

print(f"✓ train_dataset: {len(train_dataset):,}  |  val_dataset: {len(val_dataset):,}  |  test_dataset: {len(test_dataset):,}")


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
    Accumulates per-epoch val metrics during training, then appends
    final test metrics at the end and writes a single JSON file.
    """

    def __init__(self, save_path: str, run_config: dict):
        self.save_path  = save_path
        self.run_config = run_config
        self.epochs     = []    # one dict per epoch (val metrics)
        self._pending   = {}    # staging: latest train-step logs

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # Capture step-level train loss / lr (not eval logs)
        if 'loss' in logs and 'eval_loss' not in logs:
            self._pending['train_loss']    = logs.get('loss')
            self._pending['learning_rate'] = logs.get('learning_rate')
            self._pending['global_step']   = state.global_step

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        self.epochs.append({
            'epoch':            metrics.get('epoch', state.epoch),
            # val metrics
            'eval_loss':        metrics.get('eval_loss'),
            'eval_accuracy':    metrics.get('eval_accuracy'),
            'eval_precision':   metrics.get('eval_precision'),
            'eval_recall':      metrics.get('eval_recall'),
            'eval_f1':          metrics.get('eval_f1'),
            'eval_f1_weighted': metrics.get('eval_f1_weighted'),
            # train metrics from last logged step
            'train_loss':       self._pending.get('train_loss'),
            'learning_rate':    self._pending.get('learning_rate'),
            'global_step':      state.global_step,
        })

    def write(self, test_metrics: dict = None):
        """Write the JSON file. Call this after training (and after test eval)."""
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

    # Also hook on_train_end as a safety net (test_metrics will be empty if called here)
    def on_train_end(self, args, state, control, **kwargs):
        if not os.path.exists(self.save_path):
            self.write()


# ==================== MODEL LOADER ====================
def load_fresh_model(bert_config_file, bert_model_path, num_labels):
    """Load a fresh copy of the pre-trained BERT model for each run."""
    if os.path.exists(bert_config_file):
        cfg = BertConfig.from_json_file(bert_config_file)
    else:
        try:
            cfg = BertConfig.from_pretrained(bert_model_path)
        except Exception:
            cfg = None

    if cfg:
        cfg.num_labels = num_labels

    try:
        base  = BertModel.from_pretrained(bert_model_path)
        if cfg is None:
            cfg = BertConfig.from_pretrained(bert_model_path)
            cfg.num_labels = num_labels
        mdl   = BertForSequenceClassification(cfg)
        base_sd = base.state_dict()
        mdl_sd  = mdl.state_dict()
        for name, param in base_sd.items():
            if name in mdl_sd:
                mdl_sd[name].copy_(param)
        mdl.load_state_dict(mdl_sd, strict=False)
    except Exception as e:
        print(f"  BertModel load failed ({e}), trying MLM checkpoint...")
        mlm = BertForMaskedLM.from_pretrained(bert_model_path)
        if cfg is None:
            cfg = mlm.config
            cfg.num_labels = num_labels
        mdl = BertForSequenceClassification(cfg)
        mlm_sd = mlm.state_dict()
        mdl_sd = mdl.state_dict()
        for name, param in mlm_sd.items():
            if name.startswith('bert.') and name in mdl_sd:
                mdl_sd[name].copy_(param)
        mdl.load_state_dict(mdl_sd, strict=False)

    return mdl, cfg


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
    print(f"  batch_size={bs}  lr={lr}  warmup_ratio={wr}  epochs={NUM_EPOCHS}")

    run_config = {
        'run_index':        run_idx,
        'stage':            STAGE_TAG,
        'train_batch_size': bs,
        'learning_rate':    lr,
        'warmup_ratio':     wr,
        'num_epochs':       NUM_EPOCHS,
        'weight_decay':     WEIGHT_DECAY,
        'max_length':       MAX_LENGTH,
        'val_split':        VAL_SPLIT,
        'num_labels':       NUM_LABELS,
        'label_names':      list(le.classes_),
        'train_samples':    len(tr_texts),
        'val_samples':      len(val_texts),
        'test_samples':     len(test_comments),
        'random_seed':      RANDOM_SEED,
        'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
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
    model, cfg = load_fresh_model(BERT_CONFIG_FILE, BERT_MODEL_PATH, NUM_LABELS)

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
        save_total_limit=1,          # keep only best checkpoint to save disk
        fp16=USE_FP16 and torch.cuda.is_available(),
        dataloader_num_workers=NUM_WORKERS,
        seed=RANDOM_SEED,
        report_to="wandb" if USE_WANDB else "none",
        run_name=f"{STAGE_TAG}_{run_name}" if USE_WANDB else None,
        push_to_hub=False,
    )

    epoch_logger = EpochJsonLogger(save_path=json_path, run_config=run_config)

    trainer = Trainer(
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
        y_true       = np.array(test_labels)
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

    # -- write JSON (includes per-epoch val + final test metrics) --
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
        cfg   = data['run_config']
        epcs  = data['per_epoch_metrics']
        last  = epcs[-1] if epcs else {}
        best_val = max(epcs, key=lambda e: e.get('eval_f1') or 0, default={})
        tm    = data.get('test_metrics', {})
        summary_rows.append({
            'run_file':           fname,
            'batch_size':         cfg['train_batch_size'],
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

print("\n" + "=" * 80)
print("🎉 ALL DONE!")
print("=" * 80)