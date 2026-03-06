"""
BERT Fine-tuning for Writing Style Classification — Hyperparameter Grid Search

Runs all combinations of:
    TRAIN_BATCH_SIZE : [8, 16, 32, 64]
    LEARNING_RATE    : [5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
    WARMUP_RATIO     : [0.05, 0.01, 0.15, 0.2, 0.25, 0.3]
    NUM_EPOCHS       : 20  (fixed)

Total runs: 4 × 10 × 6 = 240

Per-epoch evaluation metrics are written to:
    results/writing_style_finetune/<run_name>.json

Expected CSV format:
    comments,labels
    "<sinhala text>","LABEL_NAME"
    ...

Labels are string values (e.g. ACADEMIC, SPORTS, POLITICAL, ...) and are
auto-detected from the CSV then encoded to integers via LabelEncoder.
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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

DATA_PATH = "data/Writing-style-classification/writesty3.csv"

MAX_LENGTH                  = 512
NUM_EPOCHS                  = 20       # fixed
WEIGHT_DECAY                = 0.05
GRADIENT_ACCUMULATION_STEPS = 1
EVAL_BATCH_SIZE_FIXED       = 32      # constant eval batch across all runs
RANDOM_SEED                 = 42
TEST_SIZE                   = 0.2
USE_FP16                    = True
NUM_WORKERS                 = 2

USE_WANDB     = True
WANDB_PROJECT = "bert-sentiment-finetuning"
WANDB_ENTITY  = None

# ==================== GRID ====================
TRAIN_BATCH_SIZES = [8, 16, 32, 64]
LEARNING_RATES    = [5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
WARMUP_RATIOS     = [0.05, 0.01, 0.15, 0.2, 0.25, 0.3]

# Results output dir
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]   # "writing_style_finetune"
RESULTS_DIR = os.path.join("results", SCRIPT_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== REPRODUCIBILITY ====================
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print("=" * 80)
print("BERT WRITING STYLE CLASSIFICATION — HYPERPARAMETER GRID SEARCH")
print("=" * 80)
print(f"Grid: {len(TRAIN_BATCH_SIZES)} batch sizes × "
      f"{len(LEARNING_RATES)} learning rates × "
      f"{len(WARMUP_RATIOS)} warmup ratios = "
      f"{len(TRAIN_BATCH_SIZES)*len(LEARNING_RATES)*len(WARMUP_RATIOS)} runs")
print(f"Epochs per run : {NUM_EPOCHS}")
print(f"Results dir    : {RESULTS_DIR}/")
print()


# ==================== VERIFY PATHS ====================
assert os.path.exists(BERT_MODEL_PATH), f"❌ Model path not found: {BERT_MODEL_PATH}"
assert os.path.exists(TOKENIZER_MODEL), f"❌ Tokenizer not found: {TOKENIZER_MODEL}"
assert os.path.exists(DATA_PATH),       f"❌ Data file not found: {DATA_PATH}"
print("✓ All required paths verified")


# ==================== LOAD TOKENIZER ====================
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)
PAD_ID = sp.pad_id()
print(f"✓ SentencePiece tokenizer loaded  (vocab size: {sp.get_piece_size()}  |  PAD_ID: {PAD_ID})")


# ==================== LOAD DATASET (once) ====================
print("\n" + "=" * 80)
print("LOADING DATASET")
print("=" * 80)

try:
    df = pd.read_csv(DATA_PATH)
except pd.errors.ParserError:
    print("⚠️  CSV has formatting issues, using error-tolerant parsing...")
    df = pd.read_csv(DATA_PATH, engine='python', on_bad_lines='skip')

df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

possible_comment_cols = [col for col in df.columns if 'comment' in col.lower()]
possible_label_cols   = [col for col in df.columns if 'label'   in col.lower()]

if possible_comment_cols and possible_label_cols:
    df = df.rename(columns={possible_comment_cols[0]: 'comment',
                             possible_label_cols[0]:   'label'})
else:
    print("⚠️  Could not auto-detect columns — assuming last two are [comment, label]")
    df = df.iloc[:, -2:]
    df.columns = ['comment', 'label']

df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore')
df = df.dropna(subset=['comment', 'label'])
df['comment'] = df['comment'].astype(str).str.strip()
df['label']   = df['label'].astype(str).str.strip().str.upper()
df = df[df['comment'].str.len() > 0]

print(f"✓ Dataset loaded: {len(df)} samples")


# ==================== ENCODE STRING LABELS (once) ====================
print("\n" + "=" * 80)
print("ENCODING LABELS")
print("=" * 80)

le = LabelEncoder()
df['label_id'] = le.fit_transform(df['label'])

label_to_id = {label: idx for idx, label in enumerate(le.classes_)}
id_to_label = {idx: label for label, idx in label_to_id.items()}
NUM_LABELS  = len(le.classes_)

print(f"✓ Found {NUM_LABELS} unique labels:")
for idx, label in sorted(id_to_label.items()):
    count = (df['label_id'] == idx).sum()
    pct   = 100 * count / len(df)
    print(f"  [{idx:2d}] {label:20s} — {count:6d} samples ({pct:.1f}%)")

mapping_df = pd.DataFrame({
    'label_id':   list(id_to_label.keys()),
    'label_name': list(id_to_label.values())
})


# ==================== TRAIN / VAL SPLIT (once) ====================
comments = df['comment'].tolist()
labels   = df['label_id'].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    comments, labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=labels
)
print(f"\n✓ Split — train: {len(train_texts):,}  |  val: {len(val_texts):,}")


# ==================== DATASET CLASS ====================
class SentencePieceDataset(Dataset):
    """Custom Dataset for BERT fine-tuning with SentencePiece tokenizer."""

    def __init__(self, texts, labels, sp_processor, max_length=128):
        self.texts      = texts
        self.labels     = labels
        self.sp         = sp_processor
        self.max_length = max_length
        self.pad_id     = sp_processor.pad_id()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        token_ids      = self.sp.encode(self.texts[idx])[:self.max_length]
        pad_len        = self.max_length - len(token_ids)
        attention_mask = [1] * len(token_ids) + [0] * pad_len
        token_ids      = token_ids + [self.pad_id] * pad_len
        return {
            'input_ids':      torch.tensor(token_ids,            dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask,       dtype=torch.long),
            'labels':         torch.tensor(self.labels[idx],     dtype=torch.long),
        }


train_dataset = SentencePieceDataset(train_texts, train_labels, sp, MAX_LENGTH)
val_dataset   = SentencePieceDataset(val_texts,   val_labels,   sp, MAX_LENGTH)
print(f"✓ train_dataset: {len(train_dataset):,}  |  val_dataset: {len(val_dataset):,}")


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
    """Accumulates per-epoch train + eval metrics and writes a JSON file after training."""

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
            # eval metrics
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

    def on_train_end(self, args, state, control, **kwargs):
        output = {
            'run_config':        self.run_config,
            'total_epochs':      len(self.epochs),
            'per_epoch_metrics': self.epochs,
        }
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Metrics saved → {self.save_path}")


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
        base = BertModel.from_pretrained(bert_model_path)
        if cfg is None:
            cfg = BertConfig.from_pretrained(bert_model_path)
            cfg.num_labels = num_labels
        mdl     = BertForSequenceClassification(cfg)
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
        mdl     = BertForSequenceClassification(cfg)
        mlm_sd  = mlm.state_dict()
        mdl_sd  = mdl.state_dict()
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
        'train_batch_size': bs,
        'learning_rate':    lr,
        'warmup_ratio':     wr,
        'num_epochs':       NUM_EPOCHS,
        'weight_decay':     WEIGHT_DECAY,
        'max_length':       MAX_LENGTH,
        'num_labels':       NUM_LABELS,
        'label_names':      list(le.classes_),
        'train_samples':    len(train_texts),
        'val_samples':      len(val_texts),
        'test_size':        TEST_SIZE,
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
            name=run_name,
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
        run_name=run_name if USE_WANDB else None,
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
        completed += 1
        print(f"  ✓ Run {run_idx} complete")
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted at run {run_idx}. Saving partial results...")
        epoch_logger.on_train_end(training_args, trainer.state, None)
        if USE_WANDB:
            wandb.finish(exit_code=1)
        raise
    except Exception as exc:
        print(f"  ❌ Run {run_idx} failed: {exc}")
        epoch_logger.on_train_end(training_args, trainer.state, None)
        if USE_WANDB:
            wandb.finish(exit_code=1)
        del model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        continue

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
        best_f1_epoch = max(epcs, key=lambda e: e.get('eval_f1') or 0, default={})
        summary_rows.append({
            'run_file':         fname,
            'batch_size':       cfg['train_batch_size'],
            'learning_rate':    cfg['learning_rate'],
            'warmup_ratio':     cfg['warmup_ratio'],
            'num_epochs':       cfg['num_epochs'],
            'num_labels':       cfg['num_labels'],
            # best val epoch
            'best_epoch':       best_f1_epoch.get('epoch'),
            'best_eval_f1':     best_f1_epoch.get('eval_f1'),
            'best_eval_f1_w':   best_f1_epoch.get('eval_f1_weighted'),
            'best_eval_acc':    best_f1_epoch.get('eval_accuracy'),
            'best_eval_loss':   best_f1_epoch.get('eval_loss'),
            # final val epoch
            'final_eval_f1':    last.get('eval_f1'),
            'final_eval_f1_w':  last.get('eval_f1_weighted'),
            'final_eval_acc':   last.get('eval_accuracy'),
            'final_eval_loss':  last.get('eval_loss'),
        })
    except Exception as e:
        print(f"  ⚠️  Could not read {fname}: {e}")

if summary_rows:
    summary_df = pd.DataFrame(summary_rows).sort_values('best_eval_f1', ascending=False)
    summary_csv = os.path.join(RESULTS_DIR, "grid_search_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Summary CSV written: {summary_csv}")
    print("\nTop-5 runs by best eval F1:")
    print(summary_df.head(5).to_string(index=False))

print("\n" + "=" * 80)
print("🎉 ALL DONE!")
print("=" * 80)