import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# ============================================================
# Load dataset
# ============================================================

file_path = "data/Writing-style-classification/writesty3.csv"
df = pd.read_csv(
    file_path,
    engine="python",
    skipinitialspace=True,
    on_bad_lines="skip"
)

# clean column names
df.columns = df.columns.str.strip()

# ============================================================
# labels distribution
# ============================================================

labels_counts = df['labels'].value_counts().sort_index()

print("Article count per labels:\n")
for labels, count in labels_counts.items():
    print(f"labels {labels}: {count}")

# ============================================================
# Text length analysis
# ============================================================

df["length"] = df["comments"].astype(str).str.len()

print("\nAverage article length per labels:")
print(df.groupby("labels")["length"].mean())

# ============================================================
# labels entropy
# ============================================================

counts = labels_counts.values
probs = counts / counts.sum()

entropy = -np.sum(probs * np.log2(probs))
print("\nlabels entropy:", entropy)

# ============================================================
# Duplicate detection
# ============================================================

duplicates_count = df["comments"].duplicated().sum()
print("\nDuplicate articles:", duplicates_count)

# show duplicate rows
duplicates = df[df.duplicated(subset="comments", keep=False)]
print("\nDuplicate samples:")
print(duplicates.head())

# ============================================================
# Top words per labels
# ============================================================

for labels in sorted(df["labels"].unique()):
    text = " ".join(df[df["labels"] == labels]["comments"].astype(str))
    words = text.split()

    most_common = Counter(words).most_common(10)

    print(f"\nlabels {labels} top words:")
    print(most_common)

# ============================================================
# Remove duplicates
# ============================================================

df_clean = df.drop_duplicates(subset="comments", keep="first")

removed = len(df) - len(df_clean)

print("\nOriginal dataset size:", len(df))
print("Clean dataset size:", len(df_clean))
print("Duplicates removed:", removed)

# save cleaned dataset
df_clean.to_csv("writing_style_clean.csv", index=False)

# ============================================================
# Stratified train/test split (80/20)
# ============================================================

train_df, test_df = train_test_split(
    df_clean,
    test_size=0.2,
    stratify=df_clean["labels"],
    random_state=42
)

train_df.to_csv("writing_style_train.csv", index=False)
test_df.to_csv("writing_style_test.csv", index=False)

print("\nTrain size:", len(train_df))
print("Test size:", len(test_df))

# ============================================================
# Verify labels distribution
# ============================================================

print("\nTrain distribution:")
print(train_df["labels"].value_counts(normalize=True))

print("\nTest distribution:")
print(test_df["labels"].value_counts(normalize=True))