"""
GPT few-shot baseline for the Sinhala sentiment task (reviewer 1, weakness on
sentiment comparability — see reviews/1.txt).

HelaBERT's sentiment numbers use a public 3-class dataset
(sinhala-nlp/sinhala-sentiment-analysis) because the 4-class set used by the
original baselines is no longer available, so no baseline has ever been run
on the same test set. Fine-tuning every baseline model is out of budget, so
this script produces one additional, directly comparable number: a GPT
few-shot baseline evaluated on the exact same test split.

Task (mirrors utils/finetune/sentiment_context_aware_finetune.py):
    Given a news article (title + body) and a reader COMMENT on that
    article, classify the comment's sentiment as POSITIVE / NEGATIVE /
    NEUTRAL.

Data:
    Few-shot exemplars are sampled from data/sinhala-sentiment-analysis/
    outputs/train.csv (stratified, k per class). Evaluation runs over every
    row of outputs/test.csv (513 rows) -- the same test split HelaBERT was
    scored on.

Usage:
    python baselines/gpt_sentiment_baseline.py
    python baselines/gpt_sentiment_baseline.py --model gpt-4o --few-shot-k 5
    python baselines/gpt_sentiment_baseline.py --limit 20   # cheap smoke test
"""

import argparse
import csv
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

# ==================== DEFAULTS ====================
ROOT              = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRAIN     = os.path.join(ROOT, "data/sinhala-sentiment-analysis/outputs/train.csv")
DEFAULT_TEST      = os.path.join(ROOT, "data/sinhala-sentiment-analysis/outputs/test.csv")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "results_test/GPT_sentiment_baseline")

LABELS       = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
MAX_BODY_CHARS_DEFAULT = 2000  # article context cap, keeps cost/latency bounded

SYSTEM_PROMPT = """You are a sentiment annotator for a Sinhala news-comment dataset.
You will be shown a news article (title + body, may be in Sinhala) and a reader
COMMENT responding to that article. Classify the sentiment the COMMENT expresses
using exactly one of these three labels:

- POSITIVE: the comment expresses approval, praise, support, or a positive reaction.
- NEGATIVE: the comment expresses criticism, anger, disapproval, or a negative reaction.
- NEUTRAL: the comment is factual, ambiguous, sarcastic-but-unclear, or has no clear
  positive/negative stance.

Judge the comment's own sentiment, not the sentiment of the article. Respond with
JSON only, matching the given schema."""

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "sentiment_label",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"label": {"type": "string", "enum": LABELS}},
            "required": ["label"],
            "additionalProperties": False,
        },
    },
}


# ==================== PROMPT BUILDING ====================
def truncate(text, max_chars):
    text = "" if pd.isna(text) else str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " …"


def format_example(title, body, comment, max_body_chars):
    title = "" if pd.isna(title) else str(title).strip()
    body  = truncate(body, max_body_chars)
    return (
        f"ARTICLE TITLE: {title}\n"
        f"ARTICLE BODY: {body}\n"
        f"COMMENT: {comment.strip()}"
    )


def build_few_shot_messages(train_df, k_per_class, seed, max_body_chars):
    rng = random.Random(seed)
    shots = []
    for label in LABELS:
        pool = train_df[train_df["comment_sentiment"] == label]
        n = min(k_per_class, len(pool))
        shots.extend(pool.sample(n=n, random_state=rng.randint(0, 2**31)).to_dict("records"))
    rng.shuffle(shots)

    messages = []
    for row in shots:
        messages.append({
            "role": "user",
            "content": format_example(row["title"], row["body"], row["comment_phrase"], max_body_chars),
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps({"label": row["comment_sentiment"]}),
        })
    return messages


# ==================== API CALL ====================
@retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(8))
def classify(client, model, few_shot_messages, title, body, comment, max_body_chars):
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + few_shot_messages
        + [{"role": "user", "content": format_example(title, body, comment, max_body_chars)}]
    )
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=RESPONSE_SCHEMA,
        temperature=0,
    )
    raw = resp.choices[0].message.content
    label = json.loads(raw)["label"]
    if label not in LABELS:
        raise ValueError(f"unexpected label: {label!r}")
    return label, raw


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN)
    parser.add_argument("--test-path", default=DEFAULT_TEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--few-shot-k", type=int, default=3, help="few-shot examples per class")
    parser.add_argument("--max-body-chars", type=int, default=MAX_BODY_CHARS_DEFAULT)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N test rows (smoke test)")
    args = parser.parse_args()

    load_dotenv(os.path.join(ROOT, ".env"))
    api_key = os.getenv("OPENAI_API") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API (or OPENAI_API_KEY) in .env before running this script.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    train_df = pd.read_csv(args.train_path)
    test_df  = pd.read_csv(args.test_path)
    if args.limit:
        test_df = test_df.head(args.limit)

    few_shot_messages = build_few_shot_messages(train_df, args.few_shot_k, args.seed, args.max_body_chars)
    print(f"Model: {args.model} | few-shot k/class: {args.few_shot_k} "
          f"({len(few_shot_messages)//2} shots) | test rows: {len(test_df)}")

    results = [None] * len(test_df)
    errors  = []

    def worker(i, row):
        try:
            label, raw = classify(
                client, args.model, few_shot_messages,
                row["title"], row["body"], row["comment_phrase"], args.max_body_chars,
            )
            return i, label, raw, None
        except Exception as e:
            return i, "NEUTRAL", "", str(e)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(worker, i, row) for i, row in test_df.reset_index(drop=True).iterrows()]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="classifying"):
            i, label, raw, err = fut.result()
            results[i] = (label, raw)
            if err:
                errors.append((i, err))

    if errors:
        print(f"\n{len(errors)} calls failed and fell back to NEUTRAL:")
        for i, err in errors[:10]:
            print(f"  row {i}: {err}")

    y_true = test_df["comment_sentiment"].tolist()
    y_pred = [r[0] for r in results]

    os.makedirs(args.output_dir, exist_ok=True)

    pred_path = os.path.join(args.output_dir, "predictions.csv")
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "title", "comment_phrase", "true_label", "pred_label", "raw_response"])
        for i, ((label, raw), row) in enumerate(zip(results, test_df.itertuples())):
            writer.writerow([i, row.title, row.comment_phrase, y_true[i], label, raw])

    accuracy  = accuracy_score(y_true, y_pred)
    f1_macro  = f1_score(y_true, y_pred, labels=LABELS, average="macro")
    f1_weighted = f1_score(y_true, y_pred, labels=LABELS, average="weighted")
    precision = precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    recall    = recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)

    results_path = os.path.join(args.output_dir, "results.csv")
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "model", "best_run", "checkpoint", "test_samples",
                          "accuracy", "f1_macro", "f1_weighted", "precision", "recall"])
        writer.writerow(["sentiment", args.model, "few_shot", f"k={args.few_shot_k}", len(test_df),
                          round(accuracy, 4), round(f1_macro, 4), round(f1_weighted, 4),
                          round(precision, 4), round(recall, 4)])

    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"GPT few-shot sentiment baseline\n")
        f.write(f"Model: {args.model} | few-shot k/class: {args.few_shot_k} | seed: {args.seed}\n")
        f.write(f"Test samples: {len(test_df)} | failed calls (fell back to NEUTRAL): {len(errors)}\n")
        f.write("=" * 70 + "\n")
        f.write(classification_report(y_true, y_pred, labels=LABELS, digits=4, zero_division=0))
        f.write("\nConfusion matrix (rows=true, cols=pred), label order " + str(LABELS) + "\n")
        f.write(str(confusion_matrix(y_true, y_pred, labels=LABELS)))

    print(f"\naccuracy={accuracy:.4f}  f1_macro={f1_macro:.4f}  f1_weighted={f1_weighted:.4f}  "
          f"precision={precision:.4f}  recall={recall:.4f}")
    print(f"Wrote: {results_path}\nWrote: {pred_path}\nWrote: {report_path}")


if __name__ == "__main__":
    main()
