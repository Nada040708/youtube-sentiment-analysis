import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "data/processed/reddit_cleaned.csv"

df = pd.read_csv(INPUT)

# Check minimum dataset requirements
counts = df["label"].value_counts()

if len(df) < 1000 or any(counts.get(label, 0) < 300 for label in [-1, 0, 1]):
    raise ValueError(
        "Dataset requirements not satisfied: "
        "minimum 1000 comments and 300 comments per class required."
    )

# Train/Test split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("Before balancing:")
print(train_df["label"].value_counts().sort_index())

# Handle class imbalance in training set
counts = train_df["label"].value_counts()

if counts.max() / counts.min() > 1.5:
    min_count = counts.min()

    train_df = (
        train_df.groupby("label", group_keys=False)
        .sample(n=min_count, random_state=42)
    )

    train_df = train_df.sample(frac=1, random_state=42)

    print("\nUndersampling applied.")

else:
    print("\nNo significant class imbalance detected.")

# Save datasets
train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("\nTrain size:", len(train_df))
print("Test size:", len(test_df))

print("\nFinal training distribution:")
print(train_df["label"].value_counts().sort_index())

print("\nTest distribution:")
print(test_df["label"].value_counts().sort_index())

print("\nTrain/Test split completed!")