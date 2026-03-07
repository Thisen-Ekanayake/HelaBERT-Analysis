"""
sentiment_comments_finetune.py
BERT fine-tuning for Sentiment Analysis — Comments Only.
Uses Ray Tune ASHA + Optuna HPO (mirrors hpo_lora_classification.py style).

Logs per-epoch eval metrics (loss, accuracy, F1) to JSONL in real time.
Final summary written to LOG_DIR/<run_name>_summary.json.
Test-set evaluation is run after training with the best checkpoint.

Single-run mode : HPO=0  (default)
HPO mode        : HPO=1  → Ray Tune ASHA + Optuna search

TSV columns used:
    comment_phrase    → input text
    comment_sentiment → label  (e.g. POSITIVE, NEGATIVE, NEUTRAL)

Separate train/test TSV files are expected; a validation split is
carved from train for per-epoch logging and early stopping.
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
from sklearn.preprocessing import LabelEncoder
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

TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA_PATH", "data/sinhala-sentiment-analysis/train.tsv")
TEST_DATA_PATH  = os.environ.get("TEST_DATA_PATH",  "data/sinhala-sentiment-analysis/test.tsv")

COMMENT_COL = "comment_phrase"
LABEL_COL   = "comment_sentiment"
STAGE_TAG   = "comments_only"

OUT_DIR = os.environ.get("OUT_DIR", "output/sentiment_comments")
LOG_DIR = os.environ.get("LOG_DIR", os.path.join(OUT_DIR, "logs"))

MAX_LENGTH  = int(os.environ.get("MAX_LENGTH",  "256"))
VAL_SPLIT   = float(os.environ.get("VAL_SPLIT", "0.1"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS",  "2"))

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

EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "64"))

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

assert os.path.exists(BERT_MODEL_PATH), f"❌ Model not found: {BERT_MODEL_PATH}"
assert os.path.exists(TOKENIZER_MODEL), f"❌ Tokenizer not found: {TOKENIZER_MODEL}"
assert os.path.exists(TRAIN_DATA_PATH), f"❌ Train file not found: {TRAIN_DATA_PATH}"
assert os.path.exists(TEST_DATA_PATH),  f"❌ Test file not found: {TEST_DATA_PATH}"
print("✓ All paths verified")

# ============================================================
# LOAD TOKENIZER (once)
# ============================================================

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)
PAD_ID = sp.pad_id()
print(f"✓ SentencePiece tokenizer loaded  (vocab size: {sp.get_piece_size()}  |  PAD_ID: {PAD_ID})")

# ============================================================
# LOAD DATA (once)
# ============================================================

def load_tsv(path: str, comment_col: str, label_col: str) -> pd.DataFrame:
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
    return df[df['comment'].str.len() > 0]

train_df = load_tsv(TRAIN_DATA_PATH, COMMENT_COL, LABEL_COL)
test_df  = load_tsv(TEST_DATA_PATH,  COMMENT_COL, LABEL_COL)
print(f"✓ Train TSV loaded  →  {len(train_df):,} rows")
print(f"✓ Test  TSV loaded  →  {len(test_df):,} rows")

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

print(f"✓ {NUM_LABELS} unique sentiment labels: {list(le.classes_)}")
for idx, lbl in sorted(id_to_label.items()):
    tr = (train_df['label_id'] == idx).sum()
    te = (test_df['label_id']  == idx).sum()
    print(f"  [{idx}] {lbl:20s}  train: {tr:5d}  test: {te:5d}")

# ============================================================
# TRAIN / VAL SPLIT (once)
# ============================================================

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

# ============================================================
# DATASET CLASS
# ============================================================

class CommentDataset(Dataset):
    def __init__(self, texts, labels, sp_processor, max_length):
        self.texts      = texts
        self.labels     = labels
        self.sp         = sp_processor
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids  = self.sp.encode(self.texts[idx])[:self.max_length]
        mask = [1] * len(ids)
        pad  = self.max_length - len(ids)
        ids  += [PAD_ID] * pad
        mask += [0] * pad
        return {
            'input_ids':      torch.tensor(ids,              dtype=torch.long),
            'attention_mask': torch.tensor(mask,             dtype=torch.long),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }

train_dataset = CommentDataset(tr_texts,      tr_labels,   sp, MAX_LENGTH)
val_dataset   = CommentDataset(val_texts,     val_labels,  sp, MAX_LENGTH)
test_dataset  = CommentDataset(test_comments, test_labels, sp, MAX_LENGTH)
print(f"✓ train_dataset: {len(train_dataset):,}  |  val_dataset: {len(val_dataset):,}  |  test_dataset: {len(test_dataset):,}")

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
        cfg = BertConfig.from_json_file(BERT_CONFIG_FILE)
    else:
        try:
            cfg = BertConfig.from_pretrained(BERT_MODEL_PATH)
        except Exception:
            cfg = None

    if cfg:
        cfg.num_labels = NUM_LABELS

    try:
        base = BertModel.from_pretrained(BERT_MODEL_PATH)
        if cfg is None:
            cfg = BertConfig.from_pretrained(BERT_MODEL_PATH)
            cfg.num_labels = NUM_LABELS
        mdl     = BertForSequenceClassification(cfg)
        base_sd = base.state_dict()
        mdl_sd  = mdl.state_dict()
        for name, param in base_sd.items():
            if name in mdl_sd:
                mdl_sd[name].copy_(param)
        mdl.load_state_dict(mdl_sd, strict=False)
    except Exception as e:
        print(f"  BertModel load failed ({e}), trying MLM checkpoint...")
        mlm = BertForMaskedLM.from_pretrained(BERT_MODEL_PATH)
        if cfg is None:
            cfg = mlm.config
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

    # Evaluate on held-out test set (best model loaded automatically)
    try:
        test_output  = trainer.predict(test_dataset)
        y_pred       = np.argmax(test_output.predictions, axis=-1)
        y_true       = np.array(test_labels)
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
        test_metrics = {}

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