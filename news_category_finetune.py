"""
news_category_finetune.py
BERT fine-tuning for News Category Classification.
Uses Ray Tune ASHA + Optuna HPO (mirrors hpo_lora_classification.py style).

Logs per-epoch eval metrics (loss, accuracy, F1) to JSONL in real time.
Final summary written to LOG_DIR/<run_name>_summary.json.

Single-run mode : HPO=0  (default)
HPO mode        : HPO=1  → Ray Tune ASHA + Optuna search

Expected CSV format:
    comments, labels
    <sinhala text>, <0-4>
    ...
"""

import os
import gc
import json
import time
import datetime
import numpy as np
import pandas as pd
import torch
import wandb

from torch.utils.data import Dataset
import sentencepiece as spm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertForMaskedLM,
    BertModel,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    TrainerCallback,
    EarlyStoppingCallback,
)

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
DATA_PATH        = os.environ.get("DATA_PATH",        "data/Sinhala-News-Category-classification/sinhala-news-categories.csv")

OUT_DIR  = os.environ.get("OUT_DIR",  "output/news_category")
LOG_DIR  = os.environ.get("LOG_DIR",  os.path.join(OUT_DIR, "logs"))

MAX_LENGTH   = int(os.environ.get("MAX_LENGTH",   "256"))
TEST_SIZE    = float(os.environ.get("TEST_SIZE",  "0.2"))
RANDOM_SEED  = int(os.environ.get("RANDOM_SEED", "42"))
NUM_WORKERS  = int(os.environ.get("NUM_WORKERS",  "2"))

# HPO mode: HPO=1 to run Ray Tune search, HPO=0 for single run
HPO_MODE   = bool(int(os.environ.get("HPO",        "0")))
HPO_TRIALS = int(os.environ.get("HPO_TRIALS",      "20"))

# Single-run defaults (ignored when HPO=1)
MICRO_BS     = int(os.environ.get("MICRO_BS",     "16"))
GRAD_ACC     = int(os.environ.get("GRAD_ACC",     "1"))
EPOCHS       = float(os.environ.get("EPOCHS",     "20"))
LR           = float(os.environ.get("LR",         "2e-5"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY","0.05"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO","0.1"))

EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "32"))

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "bert-news-category-finetuning")
USE_WANDB     = bool(int(os.environ.get("USE_WANDB", "1")))

os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

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

assert os.path.exists(BERT_MODEL_PATH), f"❌ Model path not found: {BERT_MODEL_PATH}"
assert os.path.exists(TOKENIZER_MODEL), f"❌ Tokenizer not found: {TOKENIZER_MODEL}"
assert os.path.exists(DATA_PATH),       f"❌ Data file not found: {DATA_PATH}"
print("✓ All required paths verified")

# ============================================================
# LOAD TOKENIZER (once)
# ============================================================

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)
PAD_ID = sp.pad_id()
print(f"✓ SentencePiece tokenizer loaded  (vocab size: {sp.get_piece_size()}  |  PAD_ID: {PAD_ID})")

# ============================================================
# LOAD & SPLIT DATASET (once)
# ============================================================

try:
    df = pd.read_csv(DATA_PATH)
except pd.errors.ParserError:
    df = pd.read_csv(DATA_PATH, engine='python', on_bad_lines='skip')

df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

possible_comment_cols = [c for c in df.columns if 'comment' in c.lower()]
possible_label_cols   = [c for c in df.columns if 'label'   in c.lower()]
if possible_comment_cols and possible_label_cols:
    df = df.rename(columns={possible_comment_cols[0]: 'comment',
                             possible_label_cols[0]:   'label'})
else:
    df = df.iloc[:, -2:]
    df.columns = ['comment', 'label']

df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c], errors='ignore')
df = df.dropna()
df['comment'] = df['comment'].astype(str).str.strip()
df['label']   = df['label'].astype(str).str.strip().astype(int)

NUM_LABELS = df['label'].nunique()
print(f"✓ Dataset loaded: {len(df)} samples, {NUM_LABELS} classes")

comments = df['comment'].tolist()
labels   = df['label'].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    comments, labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=labels
)
print(f"✓ Train: {len(train_texts)}  |  Val: {len(val_texts)}")

# ============================================================
# DATASET CLASS
# ============================================================

class SentencePieceDataset(Dataset):
    def __init__(self, texts, labels, sp_processor, max_length):
        self.texts      = texts
        self.labels     = labels
        self.sp         = sp_processor
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        token_ids      = self.sp.encode(self.texts[idx])[:self.max_length]
        pad_len        = self.max_length - len(token_ids)
        attention_mask = [1] * len(token_ids) + [0] * pad_len
        token_ids      = token_ids + [PAD_ID] * pad_len
        return {
            'input_ids':      torch.tensor(token_ids,           dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask,      dtype=torch.long),
            'labels':         torch.tensor(self.labels[idx],    dtype=torch.long),
        }

train_dataset = SentencePieceDataset(train_texts, train_labels, sp, MAX_LENGTH)
val_dataset   = SentencePieceDataset(val_texts,   val_labels,   sp, MAX_LENGTH)

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
            "task":       "news_category",
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
        cfg = BertConfig.from_json_file(BERT_CONFIG_FILE)
    else:
        cfg = BertConfig.from_pretrained(BERT_MODEL_PATH)
    cfg.num_labels = NUM_LABELS

    try:
        base   = BertModel.from_pretrained(BERT_MODEL_PATH)
        mdl    = BertForSequenceClassification(cfg)
        base_sd = base.state_dict()
        mdl_sd  = mdl.state_dict()
        for name, param in base_sd.items():
            if name in mdl_sd:
                mdl_sd[name].copy_(param)
        mdl.load_state_dict(mdl_sd, strict=False)
    except Exception as e:
        print(f"  BertModel load failed ({e}), trying MLM checkpoint...")
        mlm    = BertForMaskedLM.from_pretrained(BERT_MODEL_PATH)
        cfg    = mlm.config
        cfg.num_labels = NUM_LABELS
        mdl    = BertForSequenceClassification(cfg)
        mlm_sd = mlm.state_dict()
        mdl_sd = mdl.state_dict()
        for name, param in mlm_sd.items():
            if name.startswith('bert.') and name in mdl_sd:
                mdl_sd[name].copy_(param)
        mdl.load_state_dict(mdl_sd, strict=False)

    return mdl

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
        json.dump({"run_name": run_name, "task": "news_category", **config}, f, indent=2)

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

        # misc
        dataloader_num_workers=NUM_WORKERS,
        seed=RANDOM_SEED,
        report_to=report_to,
        push_to_hub=False,
    )

    epoch_logger = EpochJSONLogger(run_name=run_name, log_dir=LOG_DIR)

    trainer = Trainer(
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
        f"news_category"
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
    print(f"HPO MODE  |  TASK=news_category  TRIALS={HPO_TRIALS}")
    print(f"{'='*60}\n")

    search_space = {
        "lr":           tune.loguniform(5e-6, 5e-5),
        "weight_decay": tune.choice([0.01, 0.05, 0.1]),
        "warmup_ratio": tune.choice([0.05, 0.1, 0.15, 0.2]),
        "micro_bs":     tune.choice([8, 16, 32]),
        "grad_acc":     tune.choice([1, 2, 4]),
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
        name="hpo_news_category",
        verbose=1,
    )

    best_cfg   = analysis.get_best_config(metric="eval_f1", mode="max")
    best_trial = analysis.get_best_trial(metric="eval_f1",  mode="max")

    hpo_summary = {
        "task":         "news_category",
        "num_trials":   HPO_TRIALS,
        "best_config":  best_cfg,
        "best_eval_f1": best_trial.last_result["eval_f1"],
        "timestamp":    datetime.datetime.utcnow().isoformat(),
    }

    hpo_path = os.path.join(LOG_DIR, "hpo_news_category_results.json")
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
    run_name = f"news_category_lr{LR:.0e}_bs{MICRO_BS}_{ts}"

    print(f"\n{'='*60}")
    print(f"SINGLE RUN  |  TASK=news_category")
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
            run_name = f"news_category_best_{ts}"
            if USE_WANDB:
                wandb.init(project=WANDB_PROJECT, name=run_name)
            best_f1 = train(best_config, run_name=run_name,
                            report_to="wandb" if USE_WANDB else "none")
            print(f"Final best model F1: {best_f1:.4f}")
            if USE_WANDB:
                wandb.finish()
    else:
        run_single()