import pandas as pd
import re
from pathlib import Path

INPUT = "data/raw/reddit.csv"
OUTPUT = "data/processed/reddit_cleaned.csv"

Path("data/processed").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

# Keep required columns and rename them
df = df[["clean_comment", "category"]].copy()
df.columns = ["text", "label"]

# Keep valid labels
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df[df["label"].isin([-1, 0, 1])]

# Remove missing values and duplicates
df = df.dropna()
df["text"] = df["text"].astype(str)
df = df.drop_duplicates(subset=["text"])

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s.,!?'-\U0001F300-\U0001FAFF\u2600-\u27BF]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["text"] = df["text"].apply(clean_text)

# Remove empty comments
df = df[df["text"] != ""]

df["label"] = df["label"].astype(int)

# Save cleaned dataset
df.to_csv(OUTPUT, index=False)

print("Cleaning completed!")
print("Dataset shape:", df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts().sort_index())

print("\nSaved to:", OUTPUT)