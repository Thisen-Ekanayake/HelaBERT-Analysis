# HelaBERT Analysis & Finetuning

This repository provides a comprehensive suite of tools for finetuning and evaluating **HelaBERT**, a BERT-based masked language model specifically pre-trained for the Sinhala language. It includes scripts for diverse NLP tasks such as sentiment analysis, news classification, and writing style identification.

## Repository Overview

The repository is organized into several functional modules:

*   **`finetune/`**: Core training scripts for downstream tasks.
    *   `news_category.py`: Fine-tunes BERT for 5-category news classification using 5-fold CV, oversampling, and weighted loss.
    *   `news_source.py`: Fine-tunes BERT for identifying 9 news sources with balanced training strategies.
    *   `sentiment.py`: Fine-tunes BERT for basic sentiment analysis (Positive, Negative, Neutral).
    *   `sentiment_bidirectional.py`: Advanced sentiment analysis using bidirectional cross-attention to ground comments in article context.
    *   `writing_style.py`: Fine-tunes BERT for classification of writing styles (Academic, News, Blog, Creative).
*   **`test/`**: Evaluation scripts for the fine-tuned models.
    *   `news_category.py`, `news_source.py`, `sentiment.py`, `sentiment_bidirectional.py`, `writing_style.py`: Each script loads its respective best model and evaluates it against held-out test data, generating detailed metrics and confusion matrices.
*   **`utils/`**: Contains experimental scripts and older versions of fine-tuning/testing implementations used for initial research.
*   **`docs/`**: Technical documentation regarding the finetuning strategies, hyperparameters, and model architectures.

## Datasets

The models are trained and evaluated on several cleaned and processed Sinhala corpora.

### 1. Sinhala News Category Classification
- **Content**: News texts (sentences) across 5 categories: Political, Business, Technology, Sports, and Entertainment.
- **Source**: Originally released by Nisansa de Silva (2015), processed to remove single-word texts and English sentences.
- **Task**: Multi-class text classification.

### 2. Sinhala News Source Classification
- **Content**: News headlines from 9 distinct news sources (e.g., Dinamina, Hiru, Newsfirst, ITN).
- **Source**: Processed version of the corpus by Sachintha et al. (2021). Subsampled to handle class imbalance.
- **Task**: Identification of news publishers.

### 3. Sinhala Sentiment Analysis
- **Content**: Sentiment-annotated phrases grounded in article context.
- **Source**: [sinhala-nlp/sinhala-sentiment-analysis](https://huggingface.co/datasets/sinhala-nlp/sinhala-sentiment-analysis) on Hugging Face.
- **Task**: Context-aware sentiment classification (Positive, Negative, Neutral).

### 4. Sinhala Writing Style Classification
- **Content**: Texts belonging to different writing styles (Academic, News, Blog, Creative).
- **Source**: Processed version of the corpus originally created by Upeksha et al. (2015).
- **Task**: Identification of stylistic features.

## Supported Tasks

### 1. Sentiment Analysis
**Context-Aware Sentiment Model** is using a cross-attention mechanism between the comment and the parent article to improve classification accuracy by understanding the broader context of the discussion.

### 2. News Classification
Scripts are provided to categorize news articles and identify their sources. These implementations utilize Stratified K-Fold cross-validation and class-weighting to handle imbalanced datasets typical of Sinhala news corpora.

### 3. Writing Style Identification
A dedicated module for analyzing and classifying the stylistic features of Sinhala text, optimized for longer sequence lengths (up to 512 tokens).

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch & HuggingFace Transformers
- SentencePiece
- Weights & Biases (optional, for experiment tracking)

### Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Finetuning
To start a finetuning run, execute the desired script from the `finetune/` directory. For example, to train the news category classifier:
```bash
python finetune/news_category.py
```

### Evaluation
Once the model is trained, you can evaluate it using scripts `test/` and display final results (Accuracy, F1-score, Confusion Matrix, etc.).
>Note: Adjust model and dataset paths accordingly.
```bash
python test/news_category.py
```

## Citation
If you use HelaBERT or this analysis framework in your research, please use the following citation:

```bibtex
@misc{ekanayake2025helabert,
  author       = {Ekanayake, T. N. D. S. W.},
  title        = {HelaBERT: A BERT-based Masked Language Model for Sinhala},
  year         = {2025},
  howpublished = {\url{https://huggingface.co/ThisenEkanayake/HelaBERT}},
  note         = {Department of Computer Science and Engineering, University of Moratuwa}
}
```
