import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# ============================================================
# Load dataset
# ============================================================

file_path = "data/Sinhala-News-Source-classification/sinhala-news-sources.csv"
df = pd.read_csv(
    file_path,
    engine="python",
    skipinitialspace=True,
    on_bad_lines="skip"
)

# clean column names
df.columns = df.columns.str.strip()

# ============================================================
# Label distribution
# ============================================================

label_counts = df['label'].value_counts().sort_index()

print("Article count per label:\n")
for label, count in label_counts.items():
    print(f"Label {label}: {count}")

# ============================================================
# Text length analysis
# ============================================================

df["length"] = df["comment"].astype(str).str.len()

print("\nAverage article length per label:")
print(df.groupby("label")["length"].mean())

# ============================================================
# Label entropy
# ============================================================

counts = label_counts.values
probs = counts / counts.sum()

entropy = -np.sum(probs * np.log2(probs))
print("\nLabel entropy:", entropy)

# ============================================================
# Duplicate detection
# ============================================================

duplicates_count = df["comment"].duplicated().sum()
print("\nDuplicate articles:", duplicates_count)

# show duplicate rows
duplicates = df[df.duplicated(subset="comment", keep=False)]
print("\nDuplicate samples:")
print(duplicates.head())

# ============================================================
# Top words per label
# ============================================================

for label in sorted(df["label"].unique()):
    text = " ".join(df[df["label"] == label]["comment"].astype(str))
    words = text.split()

    most_common = Counter(words).most_common(10)

    print(f"\nLabel {label} top words:")
    print(most_common)

# ============================================================
# Remove duplicates
# ============================================================

df_clean = df.drop_duplicates(subset="comment", keep="first")

removed = len(df) - len(df_clean)

print("\nOriginal dataset size:", len(df))
print("Clean dataset size:", len(df_clean))
print("Duplicates removed:", removed)

# save cleaned dataset
df_clean.to_csv("news_source_clean.csv", index=False)

# ============================================================
# Stratified train/test split (80/20)
# ============================================================

train_df, test_df = train_test_split(
    df_clean,
    test_size=0.2,
    stratify=df_clean["label"],
    random_state=42
)

train_df.to_csv("news_source_train.csv", index=False)
test_df.to_csv("news_source_test.csv", index=False)

print("\nTrain size:", len(train_df))
print("Test size:", len(test_df))

# ============================================================
# Verify label distribution
# ============================================================

print("\nTrain distribution:")
print(train_df["label"].value_counts(normalize=True))

print("\nTest distribution:")
print(test_df["label"].value_counts(normalize=True))