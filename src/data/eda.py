import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT = "data/processed/reddit_cleaned.csv"

df = pd.read_csv(INPUT)

Path("data/processed").mkdir(parents=True, exist_ok=True)

# Calculate text length
df["text_length"] = df["text"].astype(str).apply(lambda x: len(x.split()))

# Label distribution
print("Number of comments:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts().sort_index())

print("\nText length statistics:")
print(df["text_length"].describe())

# Class distribution plot
df["label"].value_counts().sort_index().plot(kind="bar")

plt.title("Sentiment Class Distribution")
plt.xlabel("Sentiment (-1: Negative, 0: Neutral, 1: Positive)")
plt.ylabel("Number of Comments")
plt.tight_layout()
plt.savefig("data/processed/class_distribution.png")
plt.close()

# Text length distribution plot
df["text_length"].hist(bins=30)

plt.title("Comment Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("data/processed/text_length_distribution.png")
plt.close()

print("\nEDA completed!")
print("Charts saved in data/processed/")