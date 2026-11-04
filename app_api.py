import os
import joblib

from pathlib import Path
from collections import Counter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List


# Configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "best_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"

PORT = int(os.environ.get("PORT", 7860))


# Load model and vectorizer
print("[STARTUP] Loading model and vectorizer...", flush=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not VECTORIZER_PATH.exists():
    raise FileNotFoundError(f"Vectorizer not found: {VECTORIZER_PATH}")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("[STARTUP] Model loaded successfully.", flush=True)


# Sentiment mapping
LABELS = {
    -1: "Negative",
    0: "Neutral",
    1: "Positive"
}


# FastAPI application
app = FastAPI(
    title="YouTube Sentiment Analysis API",
    description="API for analyzing YouTube comment sentiments.",
    version="1.0.0"
)


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Request and response schemas
class PredictionRequest(BaseModel):
    comments: List[str] = Field(..., min_length=1)


class PredictionResult(BaseModel):
    comment: str
    sentiment: str
    confidence: float


class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    statistics: dict


# Health endpoint
@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# Batch prediction endpoint
@app.post("/predict_batch", response_model=PredictionResponse)
def predict_batch(request: PredictionRequest):

    try:

        comments = request.comments

        if any(not comment.strip() for comment in comments):
            raise HTTPException(
                status_code=400,
                detail="Comments cannot be empty."
            )

        print(
            f"[PREDICT] Processing {len(comments)} comments...",
            flush=True
        )

        X = vectorizer.transform(comments)

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        results = []

        for comment, prediction, probability in zip(
            comments, predictions, probabilities
        ):

            sentiment = LABELS[int(prediction)]
            confidence = float(max(probability))

            results.append({
                "comment": comment,
                "sentiment": sentiment,
                "confidence": confidence
            })

        statistics = dict(
            Counter(result["sentiment"] for result in results)
        )

        print("[PREDICT] Prediction completed.", flush=True)

        return {
            "predictions": results,
            "statistics": statistics
        }

    except HTTPException:
        raise

    except Exception as error:

        print(f"[ERROR] {str(error)}", flush=True)

        raise HTTPException(
            status_code=500,
            detail="An error occurred during sentiment prediction."
        )