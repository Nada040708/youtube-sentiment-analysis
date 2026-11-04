import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. PATHS
# ============================================================

print("\n========== STEP 1: INITIALIZATION ==========", flush=True)

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Training file:", TRAIN_PATH)
print("Testing file:", TEST_PATH)
print("Models directory:", MODEL_DIR)
print("Initialization completed!")


# ============================================================
# 2. LOAD DATASETS
# ============================================================

print("\n========== STEP 2: LOADING DATA ==========", flush=True)

print("Loading training dataset...")
train_df = pd.read_csv(TRAIN_PATH)

print("Loading testing dataset...")
test_df = pd.read_csv(TEST_PATH)

print("Datasets loaded successfully!")

print("\nChecking dataset structure...")
print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)

print("\nChecking required columns...")

if not {"text", "label"}.issubset(train_df.columns):
    raise ValueError("Training dataset must contain 'text' and 'label' columns.")

if not {"text", "label"}.issubset(test_df.columns):
    raise ValueError("Testing dataset must contain 'text' and 'label' columns.")

X_train = train_df["text"].fillna("").astype(str)
y_train = train_df["label"].astype(int)

X_test = test_df["text"].fillna("").astype(str)
y_test = test_df["label"].astype(int)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting class distribution:")
print(y_test.value_counts().sort_index())

print("\nData loading and verification completed!")


# ============================================================
# 3. TF-IDF + LOGISTIC REGRESSION PIPELINE
# ============================================================

print("\n========== STEP 3: BUILDING PIPELINE ==========", flush=True)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

print("Pipeline created successfully!")
print("Components: TF-IDF + Logistic Regression")


# ============================================================
# 4. HYPERPARAMETER OPTIMIZATION
# ============================================================

print("\n========== STEP 4: HYPERPARAMETER OPTIMIZATION ==========", flush=True)

param_distributions = {
    "tfidf__max_features": [5000, 10000, 20000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [2, 5],
    "tfidf__max_df": [0.9, 1.0],
    "clf__C": [0.1, 1, 5, 10],
    "clf__class_weight": [None, "balanced"]
}

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=8,
    cv=3,
    scoring="f1_macro",
    random_state=42,
    n_jobs=1,
    verbose=2
)

print("Starting RandomizedSearchCV...")
print("Number of candidates:", 8)
print("Cross-validation folds:", 3)
print("Total fits:", 24)
print("Scoring metric: F1 Macro")
print("Please wait. Optimization may take several minutes...\n", flush=True)

start_time = time.perf_counter()

search.fit(X_train, y_train)

optimization_time = time.perf_counter() - start_time

best_pipeline = search.best_estimator_

print("\nHyperparameter optimization completed!")
print("Optimization time:", round(optimization_time, 2), "seconds")

print("\nBest parameters:")
print(search.best_params_)

print("\nBest cross-validation F1-score:",
      round(search.best_score_, 4))


# ============================================================
# 5. TRANSFORM DATA USING OPTIMIZED TF-IDF
# ============================================================

print("\n========== STEP 5: TF-IDF TRANSFORMATION ==========", flush=True)

best_vectorizer = best_pipeline.named_steps["tfidf"]

print("Transforming training data...")
X_train_tfidf = best_vectorizer.transform(X_train)

print("Transforming testing data...")
X_test_tfidf = best_vectorizer.transform(X_test)

print("TF-IDF transformation completed!")
print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)


# ============================================================
# 6. COMPARE CLASSIFICATION ALGORITHMS
# ============================================================

print("\n========== STEP 6: MODEL COMPARISON ==========", flush=True)

models = {
    "Logistic Regression": best_pipeline.named_steps["clf"],
    "SVM": LinearSVC(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=1
    )
}

results = {}

for name, model in models.items():

    print(f"\n----------------------------------------")
    print(f"Model: {name}")
    print("----------------------------------------")

    if name != "Logistic Regression":
        print(f"Training {name}...", flush=True)

        start_time = time.perf_counter()
        model.fit(X_train_tfidf, y_train)
        training_time = time.perf_counter() - start_time

        print("Training completed!")
        print("Training time:", round(training_time, 2), "seconds")

    print("Generating predictions...", flush=True)

    y_pred_model = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred_model)
    f1 = f1_score(y_test, y_pred_model, average="macro")

    results[name] = {
        "accuracy": accuracy,
        "f1": f1
    }

    print("Accuracy:", round(accuracy, 4))
    print("Macro F1-score:", round(f1, 4))

print("\n========== COMPARISON SUMMARY ==========")

for name, metrics in results.items():
    print(
        f"{name}: "
        f"Accuracy = {metrics['accuracy']:.4f} | "
        f"Macro F1 = {metrics['f1']:.4f}"
    )


# ============================================================
# 7. FINAL MODEL EVALUATION
# ============================================================

print("\n========== STEP 7: FINAL MODEL EVALUATION ==========", flush=True)

# Logistic Regression is the required final model.
best_model = best_pipeline.named_steps["clf"]

print("Final model: Logistic Regression")
print("Generating final predictions...", flush=True)

y_pred = best_model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print("\nFinal Accuracy:", round(accuracy, 4))
print("Final Macro F1-score:", round(f1, 4))

print("\nClassification Report:")

report = classification_report(
    y_test,
    y_pred,
    labels=[-1, 0, 1],
    target_names=["Negative", "Neutral", "Positive"],
    output_dict=True,
    zero_division=0
)

print(classification_report(
    y_test,
    y_pred,
    labels=[-1, 0, 1],
    target_names=["Negative", "Neutral", "Positive"],
    zero_division=0
))

print("Final model evaluation completed!")


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

print("\n========== STEP 8: CONFUSION MATRIX ==========", flush=True)

print("Generating confusion matrix...")

cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Neutral", "Positive"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()

cm_path = MODEL_DIR / "confusion_matrix.png"
plt.savefig(cm_path)
plt.close()

print("Confusion matrix saved:", cm_path)


# ============================================================
# 9. SAVE MODEL AND VECTORIZER
# ============================================================

print("\n========== STEP 9: SAVING MODEL ==========", flush=True)

model_path = MODEL_DIR / "best_model.joblib"
vectorizer_path = MODEL_DIR / "tfidf_vectorizer.joblib"

print("Saving Logistic Regression model...")
joblib.dump(best_model, model_path)

print("Saving TF-IDF vectorizer...")
joblib.dump(best_vectorizer, vectorizer_path)

print("\nModel saved:", model_path)
print("Vectorizer saved:", vectorizer_path)

print("Model and vectorizer saved successfully!")


# ============================================================
# 10. INFERENCE TIME FOR 50 COMMENTS
# ============================================================

print("\n========== STEP 10: INFERENCE PERFORMANCE ==========", flush=True)

if len(X_test) >= 50:

    print("Preparing a batch of 50 comments...")

    batch = X_test.iloc[:50]

    print("Performing warm-up...")

    X_batch = best_vectorizer.transform(batch)
    best_model.predict(X_batch)

    print("Warm-up completed!")
    print("Measuring inference time over 10 executions...\n")

    times = []

    for i in range(10):

        start = time.perf_counter()

        X_batch = best_vectorizer.transform(batch)
        best_model.predict(X_batch)

        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

        print(f"Execution {i + 1}/10: {elapsed:.2f} ms")

    inference_time = sorted(times)[len(times) // 2]

    print("\nBatch size: 50 comments")
    print("Median inference time:", round(inference_time, 2), "ms")
    print("Target: < 100 ms")

    print(
        "Result:",
        "PASSED" if inference_time < 100 else "NOT PASSED"
    )

else:
    inference_time = None
    print("Not enough test samples to create a batch of 50 comments.")


# ============================================================
# 11. CHECK PERFORMANCE CRITERIA
# ============================================================

print("\n========== STEP 11: PERFORMANCE CRITERIA ==========", flush=True)

print(
    "Accuracy >= 80%:",
    "PASSED" if accuracy >= 0.80 else "NOT PASSED"
)

print("\nF1-score per class:")

for name in ["Negative", "Neutral", "Positive"]:

    score = report[name]["f1-score"]

    print(
        f"{name}: {score:.4f} -",
        "PASSED" if score > 0.75 else "NOT PASSED"
    )

if inference_time is not None:
    print(
        "\nInference time < 100 ms:",
        "PASSED" if inference_time < 100 else "NOT PASSED"
    )


# ============================================================
# 12. COMPLETION
# ============================================================

print("\n========== TRAINING PIPELINE COMPLETED ==========")
print("Final model: Logistic Regression")
print("Vectorizer: TF-IDF")
print("Saved files:")
print("-", model_path)
print("-", vectorizer_path)
print("-", cm_path)
print("\nAll steps completed successfully!")