import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "best_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"

print("[MODEL] Loading model and vectorizer...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not VECTORIZER_PATH.exists():
    raise FileNotFoundError(f"Vectorizer not found: {VECTORIZER_PATH}")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("[MODEL] Model loaded successfully!")
print("[MODEL] Vectorizer loaded successfully!")


def predict_sentiments(comments):

    if not comments:
        raise ValueError("The comments list cannot be empty.")

    X = vectorizer.transform(comments)
    predictions = model.predict(X)

    sentiment_mapping = {
        -1: "Negative",
        0: "Neutral",
        1: "Positive"
    }

    return [
        sentiment_mapping[int(prediction)]
        for prediction in predictions
    ]