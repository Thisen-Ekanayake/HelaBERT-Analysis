import pandas as pd
import numpy as np

# path to dataset
file_path = "news_category_clean.csv"

# load dataset
df = pd.read_csv(file_path)

# remove extra spaces in column names (just in case)
df.columns = df.columns.str.strip()

# count articles per label
label_counts = df['labels'].value_counts().sort_index()

# print results
print("Article count per label:\n")
for label, count in label_counts.items():
    print(f"Label {label}: {count}")

# ============================================================

df["length"] = df["comments"].str.len()

print(df.groupby("labels")["length"].mean())

# ============================================================

counts = label_counts.values
probs = counts / counts.sum()

entropy = -np.sum(probs * np.log2(probs))

print("Label entropy:", entropy)

# ============================================================

duplicates = df["comments"].duplicated().sum()
print("Duplicate articles:", duplicates)

# ============================================================

from collections import Counter

for label in sorted(df["labels"].unique()):
    text = " ".join(df[df["labels"] == label]["comments"])
    words = text.split()
    most_common = Counter(words).most_common(10)

    print(f"\nLabel {label} top words:")
    print(most_common)

# =============================================================

duplicates = df[df.duplicated(subset="comments", keep=False)]
print(duplicates)

# =============================================================

# remove duplicate articles based on comment text
df_clean = df.drop_duplicates(subset="comments", keep="first")

# check how many were removed
removed = len(df) - len(df_clean)

print("Original dataset size:", len(df))
print("Clean dataset size:", len(df_clean))
print("Duplicates removed:", removed)

# save cleaned dataset
df_clean.to_csv("news_category_clean.csv", index=False)

# =============================================================

import pandas as pd
from sklearn.model_selection import train_test_split

# load cleaned dataset
df = pd.read_csv("news_category_clean.csv")

# stratified split (preserves label distribution)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["labels"],
    random_state=42
)

# save datasets
train_df.to_csv("news_train.csv", index=False)
test_df.to_csv("news_test.csv", index=False)

print("Train size:", len(train_df))
print("Test size:", len(test_df))

# =============================================================
import pandas as pd

news_train_df = pd.read_csv("news_train.csv")
news_test_df = pd.read_csv("news_test.csv")

print("\nTrain distribution:")
print(news_train_df["labels"].value_counts(normalize=True))

print("\nTest distribution:")
print(news_test_df["labels"].value_counts(normalize=True))