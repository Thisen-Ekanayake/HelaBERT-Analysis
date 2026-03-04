"""
BERT Fine-tuning for News Source Classification with W&B Logging

This script fine-tunes a pre-trained BERT model (trained with SentencePiece tokenizer)
on a news source classification task with Weights & Biases logging.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import sentencepiece as spm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertForMaskedLM,
    BertModel,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
import random
import wandb


# ==================== CONFIGURATION ====================
print("="*80)
print("BERT FINE-TUNING FOR NEWS SOURCE CLASSIFICATION")
print("="*80)

# Model and Tokenizer Paths
BERT_MODEL_PATH = "HelaBERT"  # ← CHANGE THIS to your trained BERT model path
TOKENIZER_MODEL = "tokenizer/unigram_32000_0.9995.model"  # SentencePiece tokenizer
BERT_CONFIG_FILE = "HelaBERT/config.json"  # BERT config from training (optional)

# Dataset Path
DATA_PATH = "data/Sinhala-News-Source-classification/sinhala-news-sources.csv"

# Training Parameters
NUM_LABELS = 9  # Number of news source classes (will be auto-detected)
MAX_LENGTH = 64  # Maximum sequence length
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-5  # Typical fine-tuning LR for BERT
NUM_EPOCHS = 10
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.05
GRADIENT_ACCUMULATION_STEPS = 1  # Increase if you need larger effective batch size

# Output Directory
OUTPUT_DIR = "HelaBERT_finetuned_news_source"

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Test Split Ratio
TEST_SIZE = 0.2

# Hardware Settings
USE_FP16 = True  # Use mixed precision training if GPU available
NUM_WORKERS = 2  # Number of dataloader workers

# Weights & Biases Configuration
USE_WANDB = True  # Set to False to disable W&B logging
WANDB_PROJECT = "bert-news-source-finetuning"  # W&B project name
WANDB_RUN_NAME = f"bert_lr{LEARNING_RATE}_bs{TRAIN_BATCH_SIZE}_ep{NUM_EPOCHS}"  # Run name
WANDB_ENTITY = None  # W&B entity (username or team), set to None for default
WANDB_RUN_ID_FILE = "wandb_run_id.txt"  # File to store W&B run ID for resuming

print("\n✓ Configuration loaded")
print(f"  - Model path: {BERT_MODEL_PATH}")
print(f"  - Tokenizer: {TOKENIZER_MODEL}")
print(f"  - Dataset: {DATA_PATH}")
print(f"  - Output directory: {OUTPUT_DIR}")
print(f"  - W&B logging: {'Enabled' if USE_WANDB else 'Disabled'}")


# ==================== SET RANDOM SEEDS ====================
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print("\n✓ Random seeds set for reproducibility")


# ==================== CHECK ENVIRONMENT ====================
print("\n" + "="*80)
print("ENVIRONMENT CHECK")
print("="*80)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("⚠️  Running on CPU - training will be slower")


# ==================== VERIFY PATHS ====================
print("\n" + "="*80)
print("VERIFYING PATHS")
print("="*80)

assert os.path.exists(BERT_MODEL_PATH), f"❌ Model path not found: {BERT_MODEL_PATH}"
assert os.path.exists(TOKENIZER_MODEL), f"❌ Tokenizer not found: {TOKENIZER_MODEL}"
assert os.path.exists(DATA_PATH), f"❌ Data file not found: {DATA_PATH}"

print("✓ All required paths verified")


# ==================== LOAD TOKENIZER ====================
print("\n" + "="*80)
print("LOADING SENTENCEPIECE TOKENIZER")
print("="*80)

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)

PAD_ID = sp.pad_id()
UNK_ID = sp.unk_id()
BOS_ID = sp.bos_id()
EOS_ID = sp.eos_id()

print("✓ SentencePiece tokenizer loaded")
print(f"  - Vocab size: {sp.get_piece_size()}")
print(f"  - PAD_ID: {PAD_ID}")
print(f"  - UNK_ID: {UNK_ID}")
print(f"  - BOS_ID: {BOS_ID}")
print(f"  - EOS_ID: {EOS_ID}")

# Test tokenization
test_text = "ත්‍රිකුණාමලයේ දී නැගෙනහිර ආරක්ෂක සේනා මූලස්ථානය"
test_tokens = sp.encode(test_text)
print(f"\nTest tokenization:")
print(f"  - Input: {test_text}")
print(f"  - Tokens: {test_tokens[:10]}... (showing first 10)")
print(f"  - Length: {len(test_tokens)}")


# ==================== LOAD DATASET ====================
print("\n" + "="*80)
print("LOADING DATASET")
print("="*80)

# Try different parsing strategies for malformed CSV
try:
    df = pd.read_csv(DATA_PATH)
except pd.errors.ParserError:
    print("⚠️  CSV has formatting issues, using error-tolerant parsing...")
    df = pd.read_csv(DATA_PATH, engine='python', on_bad_lines='skip')
    print(f"✓ Loaded with some lines skipped")

# Clean up column names - strip whitespace and handle special characters
df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace
df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)  # Normalize multiple spaces

# Check what columns we actually have
print(f"\nCleaned columns: {df.columns.tolist()}")

# Try to identify the correct column names
# The CSV appears to have columns: [index, comment, label] with extra spaces
possible_comment_cols = [col for col in df.columns if 'comment' in col.lower()]
possible_label_cols = [col for col in df.columns if 'label' in col.lower()]

if possible_comment_cols and possible_label_cols:
    # Use the first match for each
    comment_col = possible_comment_cols[0]
    label_col = possible_label_cols[0]
    
    print(f"✓ Identified comment column: '{comment_col}'")
    print(f"✓ Identified label column: '{label_col}'")
    
    # Rename them to standard names
    df = df.rename(columns={comment_col: 'comment', label_col: 'label'})
else:
    # If automatic detection fails, assume first column is index (skip), second is comment, third is label
    print("⚠️  Could not automatically identify columns, assuming: col1=comment, col2=label")
    # Take the last two columns (assuming first is an index column)
    df = df.iloc[:, -2:]  # Take last two columns
    df.columns = ['comment', 'label']

# Drop unnamed columns if they exist
df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore')

# Drop any rows with missing values
df = df.dropna()

# Trim whitespace from comment text
if 'comment' in df.columns:
    df['comment'] = df['comment'].astype(str).str.strip()

# Drop unnamed columns if they exist
df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore')

print("✓ Dataset loaded")
print(f"  - Total samples: {len(df)}")
print(f"  - Columns: {df.columns.tolist()}")
print(f"  - Shape: {df.shape}")

# Check for missing values
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print(f"\n⚠️  Missing values found:")
    print(missing_values[missing_values > 0])
    print("  - Dropping rows with missing values...")
    df = df.dropna()
    print(f"  - Remaining samples: {len(df)}")
else:
    print("✓ No missing values")

# Explore label distribution
print("\n" + "-"*80)
print("LABEL DISTRIBUTION")
print("-"*80)
label_counts = df['label'].value_counts().sort_index()
print(label_counts)

actual_num_labels = df['label'].nunique()
print(f"\nNumber of unique labels: {actual_num_labels}")
print(f"Label range: {df['label'].min()} to {df['label'].max()}")

# Update NUM_LABELS if needed
if actual_num_labels != NUM_LABELS:
    print(f"⚠️  Updating NUM_LABELS from {NUM_LABELS} to {actual_num_labels}")
    NUM_LABELS = actual_num_labels

# Show sample data
print("\n" + "-"*80)
print("SAMPLE DATA")
print("-"*80)
print(df.head())


# ==================== PREPARE DATA ====================
print("\n" + "="*80)
print("PREPARING DATA")
print("="*80)

comments = df['comment'].tolist()
labels = df['label'].tolist()

print(f"✓ Extracted {len(comments)} comments and {len(labels)} labels")

# Show some examples
print("\nSample examples:")
for i in range(min(3, len(comments))):
    comment_preview = comments[i][:80] + "..." if len(comments[i]) > 80 else comments[i]
    print(f"  {i+1}. [{labels[i]}] {comment_preview}")

# Split into train and validation sets
print(f"\nSplitting data (train: {1-TEST_SIZE:.0%}, val: {TEST_SIZE:.0%})...")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    comments, 
    labels, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_SEED,
    stratify=labels  # Maintain label distribution
)

print(f"✓ Data split completed")
print(f"  - Training samples: {len(train_texts)}")
print(f"  - Validation samples: {len(val_texts)}")

# Verify label distribution in splits
print("\nTrain label distribution:")
train_dist = pd.Series(train_labels).value_counts().sort_index()
print(train_dist.to_string())

print("\nValidation label distribution:")
val_dist = pd.Series(val_labels).value_counts().sort_index()
print(val_dist.to_string())


# ==================== DEFINE DATASET CLASS ====================
print("\n" + "="*80)
print("CREATING DATASET")
print("="*80)

class SentencePieceDataset(Dataset):
    """
    Custom Dataset for BERT fine-tuning with SentencePiece tokenizer.
    """
    def __init__(self, texts, labels, sp_processor, max_length=512):
        self.texts = texts
        self.labels = labels
        self.sp = sp_processor
        self.max_length = max_length
        self.pad_id = sp_processor.pad_id()
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize with SentencePiece
        token_ids = self.sp.encode(text)
        
        # Truncate if necessary
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(token_ids)
        
        # Pad to max_length
        padding_length = self.max_length - len(token_ids)
        token_ids = token_ids + [self.pad_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Create dataset instances
train_dataset = SentencePieceDataset(train_texts, train_labels, sp, max_length=MAX_LENGTH)
val_dataset = SentencePieceDataset(val_texts, val_labels, sp, max_length=MAX_LENGTH)

print(f"✓ Datasets created")
print(f"  - Train dataset size: {len(train_dataset)}")
print(f"  - Validation dataset size: {len(val_dataset)}")

# Test the dataset
sample = train_dataset[0]
print(f"\nSample from dataset:")
print(f"  - input_ids shape: {sample['input_ids'].shape}")
print(f"  - attention_mask shape: {sample['attention_mask'].shape}") 
print(f"  - label: {sample['labels'].item()}")
print(f"  - Non-padding tokens: {sample['attention_mask'].sum().item()}")


# ==================== LOAD MODEL ====================
print("\n" + "="*80)
print("LOADING MODEL")
print("="*80)

# Load BERT config
if os.path.exists(BERT_CONFIG_FILE):
    config = BertConfig.from_json_file(BERT_CONFIG_FILE)
    print(f"✓ Loaded BERT config from: {BERT_CONFIG_FILE}")
else:
    try:
        config = BertConfig.from_pretrained(BERT_MODEL_PATH)
        print(f"✓ Loaded BERT config from model directory")
    except:
        print(f"⚠️  Could not load config, will try to infer from model")
        config = None

# Update config for classification
if config:
    config.num_labels = NUM_LABELS
    print(f"\nBERT Configuration:")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Num layers: {config.num_hidden_layers}")
    print(f"  - Num attention heads: {config.num_attention_heads}")
    print(f"  - Vocab size: {config.vocab_size}")
    print(f"  - Max position embeddings: {config.max_position_embeddings}")
    print(f"  - Num labels: {config.num_labels}")

# Load the pre-trained BERT model for sequence classification
print(f"\nLoading model from: {BERT_MODEL_PATH}")

try:
    # Try loading the base BERT model first
    base_model = BertModel.from_pretrained(BERT_MODEL_PATH)
    print("✓ Loaded base BERT model from pretrained weights")
    
    # Create classification model with the config
    if config is None:
        config = BertConfig.from_pretrained(BERT_MODEL_PATH)
        config.num_labels = NUM_LABELS
    
    model = BertForSequenceClassification(config)
    
    # Load BERT weights, handling missing keys gracefully
    base_state = base_model.state_dict()
    model_state = model.state_dict()
    
    # Copy weights for matching keys
    for name, param in base_state.items():
        if name in model_state:
            model_state[name].copy_(param)
    
    model.load_state_dict(model_state, strict=False)
    print("✓ Model loaded with classification head")
    
except Exception as e:
    print(f"⚠️  Initial load attempt failed: {e}")
    print("   Attempting to load MLM model and transfer weights...")
    
    try:
        # Load the MLM model
        mlm_model = BertForMaskedLM.from_pretrained(BERT_MODEL_PATH)
        
        # Create classification model
        if config is None:
            config = mlm_model.config
            config.num_labels = NUM_LABELS
        
        model = BertForSequenceClassification(config)
        
        # Copy BERT encoder weights
        mlm_state = mlm_model.state_dict()
        model_state = model.state_dict()
        
        # Copy only the matching keys from MLM model
        for name, param in mlm_state.items():
            if name.startswith('bert.'):
                classification_name = name
                if classification_name in model_state:
                    model_state[classification_name].copy_(param)
        
        model.load_state_dict(model_state, strict=False)
        print("✓ Model weights loaded from MLM checkpoint with classification head initialized")
        
    except Exception as e2:
        print(f"❌ Failed to load model: {e2}")
        raise

# Print model info
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel statistics:")
print(f"  - Total parameters: {total_params:,}")
print(f"  - Trainable parameters: {trainable_params:,}")
print(f"  - Percentage trainable: {100 * trainable_params / total_params:.2f}%")


# ==================== DEFINE METRICS ====================
print("\n" + "="*80)
print("SETTING UP METRICS")
print("="*80)

def compute_metrics(eval_pred: EvalPrediction):
    """
    Compute accuracy, precision, recall, and F1 score.
    """
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

print("✓ Metrics function defined (accuracy, precision, recall, F1)")


# ==================== INITIALIZE WEIGHTS & BIASES ====================
if USE_WANDB:
    print("\n" + "="*80)
    print("INITIALIZING WEIGHTS & BIASES")
    print("="*80)
    
    # Get or create W&B run ID for resuming
    wandb_run_id = None
    if os.path.exists(WANDB_RUN_ID_FILE):
        with open(WANDB_RUN_ID_FILE, 'r') as f:
            wandb_run_id = f.read().strip()
        print(f"✓ Found existing W&B run ID: {wandb_run_id}")
    else:
        print("✓ Starting new W&B run")
    
    # Initialize W&B
    if wandb.run is None:
        wandb_config = {
            # Model config
            "model_architecture": "BERT",
            "tokenizer": "SentencePiece",
            "vocab_size": sp.get_piece_size(),
            "hidden_size": config.hidden_size if config else "unknown",
            "num_layers": config.num_hidden_layers if config else "unknown",
            "num_attention_heads": config.num_attention_heads if config else "unknown",
            
            # Training config
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "max_length": MAX_LENGTH,
            "fp16": USE_FP16 and torch.cuda.is_available(),
            
            # Data config
            "num_labels": NUM_LABELS,
            "train_samples": len(train_texts),
            "val_samples": len(val_texts),
            "test_size": TEST_SIZE,
            
            # Other
            "random_seed": RANDOM_SEED,
        }
        
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=WANDB_RUN_NAME,
            id=wandb_run_id,
            resume="allow",
            config=wandb_config
        )
        
        # Save the run ID for future resumptions
        with open(WANDB_RUN_ID_FILE, 'w') as f:
            f.write(run.id)
        print(f"✓ W&B run ID saved: {run.id}")
        print(f"✓ W&B dashboard: {run.get_url()}")
        
        # Log label distribution
        wandb.log({
            "train_label_distribution": wandb.Histogram(train_labels),
            "val_label_distribution": wandb.Histogram(val_labels)
        })
    else:
        print("✓ W&B already initialized")


# ==================== SETUP TRAINING ====================
print("\n" + "="*80)
print("CONFIGURING TRAINING")
print("="*80)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # Training hyperparameters
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    
    # Evaluation and logging
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    logging_first_step=True,
    
    # Save best model
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=3,
    
    # Hardware optimization
    fp16=USE_FP16 and torch.cuda.is_available(),
    dataloader_num_workers=NUM_WORKERS,
    
    # Reproducibility
    seed=RANDOM_SEED,
    
    # Disable push to hub
    push_to_hub=False,
    
    # W&B Reporting
    report_to="wandb" if USE_WANDB else "none",
    run_name=WANDB_RUN_NAME if USE_WANDB else None,
)

print("✓ Training arguments configured")
print(f"\nTraining configuration:")
print(f"  - Epochs: {NUM_EPOCHS}")
print(f"  - Train batch size: {TRAIN_BATCH_SIZE}")
print(f"  - Eval batch size: {EVAL_BATCH_SIZE}")
print(f"  - Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
print(f"  - Effective batch size: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"  - Learning rate: {LEARNING_RATE}")
print(f"  - Warmup ratio: {WARMUP_RATIO}")
print(f"  - Weight decay: {WEIGHT_DECAY}")
print(f"  - FP16: {training_args.fp16}")
print(f"  - Total optimization steps: ~{len(train_dataset) * NUM_EPOCHS // (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)}")
print(f"  - Reporting to: {'Weights & Biases' if USE_WANDB else 'None'}")


# ==================== INITIALIZE TRAINER ====================
print("\n" + "="*80)
print("INITIALIZING TRAINER")
print("="*80)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("✓ Trainer initialized and ready")


# ==================== TRAIN MODEL ====================
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)
print(f"Training will run for {NUM_EPOCHS} epochs...")
if USE_WANDB:
    print(f"Monitor progress at: {wandb.run.get_url()}")
print("\nYou can monitor progress below:\n")

try:
    train_result = trainer.train()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    
    # Print training metrics
    print("\nTraining Metrics:")
    for key, value in train_result.metrics.items():
        print(f"  - {key}: {value:.4f}")
    
except KeyboardInterrupt:
    print("\n" + "="*80)
    print("TRAINING INTERRUPTED BY USER")
    print("="*80)
    print("Saving current model state...")
    trainer.save_model(f"{OUTPUT_DIR}/interrupted_model")
    print(f"✓ Model saved to {OUTPUT_DIR}/interrupted_model")
    if USE_WANDB:
        wandb.finish(exit_code=1)
    raise

except Exception as e:
    print("\n" + "="*80)
    print("TRAINING FAILED")
    print("="*80)
    print(f"Error: {e}")
    if USE_WANDB:
        wandb.finish(exit_code=1)
    raise


# ==================== SAVE FINAL MODEL ====================
print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

final_model_path = f"{OUTPUT_DIR}/final_model"
trainer.save_model(final_model_path)
print(f"✓ Final model saved to: {final_model_path}")

# Also save the config
if config:
    config.save_pretrained(final_model_path)
    print(f"✓ Config saved to: {final_model_path}")

# Log model to W&B
if USE_WANDB:
    artifact = wandb.Artifact(
        name=f"bert-finetuned-{wandb.run.id}",
        type="model",
        description=f"Fine-tuned BERT for news source classification"
    )
    artifact.add_dir(final_model_path)
    wandb.log_artifact(artifact)
    print(f"✓ Model logged to W&B as artifact")


# ==================== EVALUATE MODEL ====================
print("\n" + "="*80)
print("EVALUATING ON VALIDATION SET")
print("="*80)

eval_results = trainer.evaluate()

print("\nValidation Metrics:")
for key, value in eval_results.items():
    if isinstance(value, float):
        print(f"  - {key}: {value:.4f}")
    else:
        print(f"  - {key}: {value}")


# ==================== GENERATE PREDICTIONS ====================
print("\n" + "="*80)
print("GENERATING PREDICTIONS")
print("="*80)

predictions_output = trainer.predict(val_dataset)

# Get predicted labels
y_pred = np.argmax(predictions_output.predictions, axis=-1)
y_true = np.array(val_labels)

print(f"✓ Generated {len(y_pred)} predictions")


# ==================== DETAILED EVALUATION ====================
print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORT")
print("="*80)
report_text = classification_report(y_true, y_pred, digits=4)
print(report_text)

# Log classification report to W&B
if USE_WANDB:
    wandb.log({"classification_report": wandb.Table(
        data=[[report_text]],
        columns=["report"]
    )})


# ==================== CONFUSION MATRIX ====================
print("\n" + "="*80)
print("CONFUSION MATRIX")
print("="*80)

cm = confusion_matrix(y_true, y_pred)
print(cm)

# Save confusion matrix as DataFrame
cm_df = pd.DataFrame(
    cm,
    index=[f"True_{i}" for i in range(NUM_LABELS)],
    columns=[f"Pred_{i}" for i in range(NUM_LABELS)]
)
cm_df.to_csv(f"{OUTPUT_DIR}/confusion_matrix.csv")
print(f"\n✓ Confusion matrix saved to {OUTPUT_DIR}/confusion_matrix.csv")

# Log confusion matrix to W&B
if USE_WANDB:
    # Create a heatmap-friendly format
    import plotly.figure_factory as ff
    
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=[f"Pred_{i}" for i in range(NUM_LABELS)],
        y=[f"True_{i}" for i in range(NUM_LABELS)],
        colorscale='Blues',
        showscale=True
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label"
    )
    
    wandb.log({"confusion_matrix": fig})
    print("✓ Confusion matrix logged to W&B")


# ==================== PER-CLASS METRICS ====================
print("\n" + "="*80)
print("PER-CLASS PERFORMANCE")
print("="*80)

precision_per_class, recall_per_class, f1_per_class, support_per_class = \
    precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

per_class_report = pd.DataFrame({
    'Class': range(NUM_LABELS),
    'Precision': precision_per_class,
    'Recall': recall_per_class,
    'F1-Score': f1_per_class,
    'Support': support_per_class
})

print(per_class_report.to_string(index=False))

# Save to CSV
per_class_report.to_csv(f'{OUTPUT_DIR}/per_class_metrics.csv', index=False)
print(f"\n✓ Per-class metrics saved to {OUTPUT_DIR}/per_class_metrics.csv")

# Log per-class metrics to W&B
if USE_WANDB:
    wandb.log({"per_class_metrics": wandb.Table(dataframe=per_class_report)})
    
    # Create bar charts for each metric
    for metric in ['Precision', 'Recall', 'F1-Score']:
        data = [[i, per_class_report.loc[i, metric]] for i in range(NUM_LABELS)]
        table = wandb.Table(data=data, columns=["Class", metric])
        wandb.log({f"per_class_{metric.lower().replace('-', '_')}": 
                   wandb.plot.bar(table, "Class", metric, title=f"Per-Class {metric}")})


# ==================== SAMPLE PREDICTIONS ====================
print("\n" + "="*80)
print("SAMPLE PREDICTIONS")
print("="*80)

# Find correct and incorrect predictions
correct_mask = (y_pred == y_true)
correct_indices = np.where(correct_mask)[0]
incorrect_indices = np.where(~correct_mask)[0]

print(f"\nCorrect predictions: {correct_mask.sum()} / {len(y_true)} ({100*correct_mask.sum()/len(y_true):.2f}%)")
print(f"Incorrect predictions: {(~correct_mask).sum()} / {len(y_true)} ({100*(~correct_mask).sum()/len(y_true):.2f}%)")

# Prepare samples for W&B
if USE_WANDB:
    sample_predictions = []

# Show correct predictions
print("\n" + "-"*80)
print("CORRECT PREDICTIONS (showing 5)")
print("-"*80)
for i, idx in enumerate(correct_indices[:5]):
    text = val_texts[idx]
    text_preview = text[:100] + "..." if len(text) > 100 else text
    true_label = y_true[idx]
    pred_label = y_pred[idx]
    
    # Get prediction confidence
    probs = torch.softmax(torch.tensor(predictions_output.predictions[idx]), dim=0)
    confidence = probs[pred_label].item()
    
    print(f"\n{i+1}. Text: {text_preview}")
    print(f"   True: {true_label} | Predicted: {pred_label} | Confidence: {confidence:.4f}")
    
    if USE_WANDB and i < 10:
        sample_predictions.append([text, true_label, pred_label, confidence, "Correct"])

# Show incorrect predictions
if len(incorrect_indices) > 0:
    print("\n" + "-"*80)
    print("INCORRECT PREDICTIONS (showing 5)")
    print("-"*80)
    for i, idx in enumerate(incorrect_indices[:5]):
        text = val_texts[idx]
        text_preview = text[:100] + "..." if len(text) > 100 else text
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        
        # Get prediction confidence
        probs = torch.softmax(torch.tensor(predictions_output.predictions[idx]), dim=0)
        confidence = probs[pred_label].item()
        true_confidence = probs[true_label].item()
        
        print(f"\n{i+1}. Text: {text_preview}")
        print(f"   True: {true_label} (conf: {true_confidence:.4f}) | Predicted: {pred_label} (conf: {confidence:.4f})")
        
        if USE_WANDB and i < 10:
            sample_predictions.append([text, true_label, pred_label, confidence, "Incorrect"])
else:
    print("\n🎉 No incorrect predictions! Perfect accuracy!")

# Log sample predictions to W&B
if USE_WANDB and sample_predictions:
    wandb.log({"sample_predictions": wandb.Table(
        data=sample_predictions,
        columns=["Text", "True Label", "Predicted Label", "Confidence", "Result"]
    )})


# ==================== SAVE ALL RESULTS ====================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save all predictions
results_df = pd.DataFrame({
    'text': val_texts,
    'true_label': y_true,
    'predicted_label': y_pred,
    'correct': y_true == y_pred
})

# Add prediction confidences
confidences = []
for i in range(len(predictions_output.predictions)):
    probs = torch.softmax(torch.tensor(predictions_output.predictions[i]), dim=0)
    pred_label = y_pred[i]
    confidence = probs[pred_label].item()
    confidences.append(confidence)

results_df['confidence'] = confidences
results_df.to_csv(f'{OUTPUT_DIR}/predictions.csv', index=False)
print(f"✓ All predictions saved to {OUTPUT_DIR}/predictions.csv")

# Create summary
summary = {
    'Model': 'BERT with SentencePiece',
    'Task': 'News Source Classification',
    'Num Classes': NUM_LABELS,
    'Train Samples': len(train_texts),
    'Val Samples': len(val_texts),
    'Epochs': NUM_EPOCHS,
    'Learning Rate': LEARNING_RATE,
    'Batch Size': TRAIN_BATCH_SIZE,
    'Max Length': MAX_LENGTH,
    'Accuracy': eval_results['eval_accuracy'],
    'Precision (macro)': eval_results['eval_precision'],
    'Recall (macro)': eval_results['eval_recall'],
    'F1 (macro)': eval_results['eval_f1'],
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(f'{OUTPUT_DIR}/training_summary.csv', index=False)
print(f"✓ Training summary saved to {OUTPUT_DIR}/training_summary.csv")

# Log summary metrics to W&B
if USE_WANDB:
    wandb.log({
        "final/accuracy": eval_results['eval_accuracy'],
        "final/precision": eval_results['eval_precision'],
        "final/recall": eval_results['eval_recall'],
        "final/f1": eval_results['eval_f1'],
    })
    wandb.log({"training_summary": wandb.Table(dataframe=summary_df)})


# ==================== FINAL SUMMARY ====================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

for key, value in summary.items():
    if isinstance(value, float):
        print(f"{key:20s}: {value:.4f}")
    else:
        print(f"{key:20s}: {value}")

print("\n" + "="*80)
print("🎉 FINE-TUNING COMPLETE!")
print("="*80)
print(f"\nAll results saved to: {OUTPUT_DIR}/")
print(f"\nFiles created:")
print(f"  - final_model/ (trained model)")
print(f"  - training_summary.csv (overall metrics)")
print(f"  - per_class_metrics.csv (per-class performance)")
print(f"  - predictions.csv (all validation predictions)")
print(f"  - confusion_matrix.csv (confusion matrix)")
print(f"\nBest model metrics:")
print(f"  - F1 Score: {eval_results['eval_f1']:.4f}")
print(f"  - Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"  - Precision: {eval_results['eval_precision']:.4f}")
print(f"  - Recall: {eval_results['eval_recall']:.4f}")

if USE_WANDB:
    print(f"\n📊 View full results at: {wandb.run.get_url()}")
    wandb.finish()
    print("✓ W&B run finished")

print("\n" + "="*80)