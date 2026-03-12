# BERT Fine-tuning Scripts — Technical Documentation

> All scripts fine-tune a custom Sinhala BERT model using the HuggingFace `Trainer` API. SentencePiece tokenization is used throughout. Training is monitored via Weights & Biases (W&B) and email notifications are sent on completion or failure.

---

## 1. `news_category_finetune.py`

**Task:** Multi-class classification of news articles into categories (5 classes).

**Model:** `BertForSequenceClassification` loaded fresh per fold.

**Data:** CSV with columns `comments` (Sinhala text), `labels` (integer 0–4). Input truncated/padded to 256 tokens via `SentencePieceDataset`.

**Training Strategy:**
- 5-Fold Stratified Cross-Validation (`StratifiedKFold`, `n_splits=5`)
- Training split oversampled to balance classes (`oversample()`)
- Per-fold class weights computed via inverse-frequency (`compute_class_weights()`)
- Weighted cross-entropy loss via custom `WeightedTrainer`
- Early stopping enabled; best model selected by highest validation macro-F1

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Max sequence length | 256 |
| Batch size | 16 |
| Learning rate | 3e-5 |
| Max epochs | 20 |
| Warmup ratio | 0.06 |
| Folds | 5 |

**Evaluation Metrics:** Accuracy, Macro F1, Weighted F1, Classification Report, Confusion Matrix

**Output:** Best fold model saved; mean ± std reported across all folds.

---

## 2. `news_source_finetune.py`

**Task:** Multi-class classification of news articles by source/publisher (9 classes, auto-detected from data).

**Model:** `BertForSequenceClassification` loaded fresh per fold.

**Data:** CSV with columns `comments`, `labels`. Input truncated/padded to **32 tokens** (shorter context sufficient for source identification). Tokenized via `SentencePieceDataset`.

**Training Strategy:** Identical to `news_category_finetune.py` — 5-fold stratified CV, per-fold oversampling, per-fold class weights, `WeightedTrainer`, early stopping.

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Max sequence length | 32 |
| Batch size | 16 |
| Learning rate | 3e-5 |
| Max epochs | 20 |
| Warmup ratio | 0.06 |
| Folds | 5 |

**Evaluation Metrics:** Accuracy, Macro F1, Weighted F1, Classification Report, Confusion Matrix

**Output:** Best fold model saved; mean ± std reported across all folds.

---

## 3. `sentiment_comments_finetune.py`

**Task:** Sentiment classification of Sinhala news comments (labels auto-detected: e.g., POSITIVE, NEGATIVE, NEUTRAL).

**Model:** `BertForSequenceClassification` loaded fresh per fold.

**Data:** TSV file with columns `comment_phrase` (input) and `comment_sentiment` (label). Labels encoded via `LabelEncoder`. Tokenized via `CommentDataset` (distinct from `SentencePieceDataset`). A separate held-out test set is evaluated against the best fold model.

**Training Strategy:** 5-fold stratified CV, per-fold oversampling, per-fold class weights, `WeightedTrainer`, early stopping. Includes held-out test set evaluation at the end.

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Max sequence length | 256 |
| Batch size | 16 |
| Learning rate | 3e-5 |
| Max epochs | 7 |
| Warmup ratio | 0.06 |
| Folds | 5 |

**Evaluation Metrics:** Accuracy, Macro F1, Weighted F1, Classification Report, Confusion Matrix

**Output:** Best fold model saved; held-out test set results reported separately.

---

## 4. `sentiment_context_aware_finetune.py`

**Task:** Context-aware sentiment classification of comments using the parent article as contextual grounding (Stage 2 model).

**Architecture:** Custom `CrossAttnSentimentModel` with shared BERT encoder weights:
- Article body → sliding-window chunking (512 tokens, stride 256, up to 16 chunks) → per-chunk BERT `[CLS]` vectors → chunk matrix `[B, C, H]`
- Comment → BERT `[CLS]` vector `[B, H]`
- Multi-head cross-attention (`MultiHeadCrossAttention`): comment as Query, article chunks as Key/Value → attended context `[B, H]`
- Fusion: `[comment_vec ; attended_ctx ; comment_vec ⊙ attended_ctx]` → LayerNorm → Dropout → Linear classifier

**Data:** TSV files (train/test). Text chunked via `tokenize_chunks()`. Labels encoded via `LabelEncoder`. Dataset class: `CrossAttnDataset`.

**Training Strategy:**
- Single train/validation split (`train_test_split`, **no K-fold**)
- No oversampling; no class weights
- Standard `CrossEntropyLoss` via custom `CrossAttnTrainer`
- Cosine LR scheduler with warmup
- W&B run supports resume (`resume="allow"`)
- Cross-attention weights inspected post-training to identify which article chunks influenced each prediction
- Stage 1 vs. Stage 2 comparison table printed and saved

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Chunk size | 512 tokens |
| Chunk stride | 256 (50% overlap) |
| Max chunks per article | 16 |
| Comment max length | 256 |
| Batch size | 8 |
| Learning rate | 1e-5 |
| Max epochs | 10 |
| Warmup ratio | 0.1 |
| LR scheduler | Cosine |

**Evaluation Metrics:** Accuracy, Macro F1, Weighted F1, Macro Precision, Macro Recall, Classification Report, Confusion Matrix

**Output:** Trained model saved; attention weight visualization on test samples; Stage 1 vs. Stage 2 comparison.

---

## 5. `writing_style_finetune.py`

**Task:** Classification of writing style from Sinhala news text (labels auto-detected from data).

**Model:** `BertForSequenceClassification` loaded fresh per fold.

**Data:** CSV with columns `comments`, `labels`. Labels encoded via `LabelEncoder`. Input truncated/padded to **512 tokens** (full context used for style classification). Tokenized via `SentencePieceDataset`.

**Training Strategy:** 5-fold stratified CV, per-fold oversampling, per-fold class weights, `WeightedTrainer`, early stopping.

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Max sequence length | 512 |
| Batch size | 16 |
| Learning rate | 3e-5 |
| Max epochs | 4 |
| Warmup ratio | 0.06 |
| Folds | 5 |

**Evaluation Metrics:** Accuracy, Macro F1, Weighted F1, **Macro Precision, Macro Recall** (additional vs. news scripts), Classification Report, Confusion Matrix

**Output:** Best fold model saved; mean ± std reported across all folds.

---

## Comparative Summary

| Feature | news\_category | news\_source | sentiment\_comments | sentiment\_context\_aware | writing\_style |
|---|:---:|:---:|:---:|:---:|:---:|
| CV strategy | 5-fold stratified | 5-fold stratified | 5-fold stratified | Train/val split | 5-fold stratified |
| Oversampling | ✅ | ✅ | ✅ | ❌ | ✅ |
| Class weights | ✅ | ✅ | ✅ | ❌ | ✅ |
| Custom architecture | ❌ | ❌ | ❌ | ✅ Cross-Attn | ❌ |
| Max sequence length | 256 | 32 | 256 | 512+256 | 512 |
| Held-out test set | ❌ | ❌ | ✅ | ✅ | ❌ |
| Precision & Recall | ❌ | ❌ | ❌ | ✅ | ✅ |
| LR scheduler | Linear | Linear | Linear | Cosine | Linear |
