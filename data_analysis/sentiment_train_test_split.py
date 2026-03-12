import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import entropy
import sys
import os

# ── 1. Load ────────────────────────────────────────────────────────────────────
csv_path = sys.argv[1] if len(sys.argv) > 1 else "combined_shuffled_dataset.csv"
df = pd.read_csv(csv_path)

print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution (comment_sentiment):\n{df['comment_sentiment'].value_counts()}")

# ── 2. Word-count bucket (handles body length variation) ───────────────────────
df["body_word_count"] = df["body"].fillna("").apply(lambda x: len(x.split()))

# Bin into quantile-based buckets so every bin is populated
df["wc_bucket"] = pd.qcut(df["body_word_count"], q=4, labels=False, duplicates="drop")

# Composite stratification key = class + word-count bucket
df["strat_key"] = df["comment_sentiment"].astype(str) + "_wc" + df["wc_bucket"].astype(str)

print(f"\nStratification keys:\n{df['strat_key'].value_counts()}")

# ── 3. Stratified split ────────────────────────────────────────────────────────
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Fall back to plain stratify-by-class if any stratum has < 2 samples
try:
    train_idx, test_idx = next(sss.split(df, df["strat_key"]))
except ValueError:
    print("\n⚠  Some strata too small — falling back to class-only stratification.")
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss2.split(df, df["comment_sentiment"]))

train = df.iloc[train_idx].drop(columns=["body_word_count", "wc_bucket", "strat_key"])
test  = df.iloc[test_idx ].drop(columns=["body_word_count", "wc_bucket", "strat_key"])

print(f"\nTrain size: {len(train)}  |  Test size: {len(test)}")
print(f"Train split %: {len(train)/len(df)*100:.1f}%  |  Test split %: {len(test)/len(df)*100:.1f}%")

# ── 4. Class distribution check ────────────────────────────────────────────────
classes = sorted(df["comment_sentiment"].unique())

def class_probs(subset):
    counts = subset["comment_sentiment"].value_counts().reindex(classes, fill_value=0)
    return (counts / counts.sum()).values

p_train = class_probs(train)
p_test  = class_probs(test)

print("\n── Class distribution ──────────────────────────────────────────")
print(f"{'Class':<25} {'Train %':>8} {'Test %':>8}")
for cls, pt, pte in zip(classes, p_train, p_test):
    print(f"{str(cls):<25} {pt*100:>7.1f}% {pte*100:>7.1f}%")

# ── 5. KL Divergence ──────────────────────────────────────────────────────────
# Add tiny epsilon to avoid log(0)
eps = 1e-10
p = p_train + eps;  p /= p.sum()
q = p_test  + eps;  q /= q.sum()

kl_train_test = entropy(p, q)   # KL(train || test)
kl_test_train = entropy(q, p)   # KL(test  || train)
js_div = 0.5 * kl_train_test + 0.5 * kl_test_train  # symmetric Jensen-Shannon

print("\n── KL Divergence (class distribution) ─────────────────────────")
print(f"  KL(train ∥ test)  = {kl_train_test:.6f}")
print(f"  KL(test  ∥ train) = {kl_test_train:.6f}")
print(f"  Jensen-Shannon    = {js_div:.6f}   (0 = identical, 1 = max divergence)")

# ── 6. Word-count distribution KL ─────────────────────────────────────────────
def wc_hist(subset, bins):
    wc = subset["body"].fillna("").apply(lambda x: len(x.split()))
    counts, _ = np.histogram(wc, bins=bins)
    prob = counts / counts.sum()
    return prob

all_wc = df["body"].fillna("").apply(lambda x: len(x.split()))
bins = np.histogram_bin_edges(all_wc, bins=10)

p_wc = wc_hist(train, bins) + eps;  p_wc /= p_wc.sum()
q_wc = wc_hist(test,  bins) + eps;  q_wc /= q_wc.sum()

kl_wc = entropy(p_wc, q_wc)
js_wc = 0.5 * entropy(p_wc, q_wc) + 0.5 * entropy(q_wc, p_wc)

print("\n── KL Divergence (body word-count distribution) ────────────────")
print(f"  KL(train ∥ test)  = {kl_wc:.6f}")
print(f"  Jensen-Shannon    = {js_wc:.6f}")

# ── 7. Save ────────────────────────────────────────────────────────────────────
out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)

train.to_csv(f"{out_dir}/train.csv", index=False)
test.to_csv( f"{out_dir}/test.csv",  index=False)

print(f"\nSaved → {out_dir}/train.csv  ({len(train)} rows)")
print(f"Saved → {out_dir}/test.csv   ({len(test)} rows)")