from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import PredictionRequest, PredictionResponse
from src.api.model import model, predict_sentiments

from collections import Counter


print("[API] Initializing FastAPI application...")

app = FastAPI(
    title="YouTube Sentiment Analysis API",
    description="API for analyzing the sentiment of YouTube comments",
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

print("[API] CORS configured successfully!")


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
        print(f"[PREDICT] Received {len(request.comments)} comments")

        # Check for empty comments
        if any(not comment.strip() for comment in request.comments):
            raise HTTPException(
                status_code=400,
                detail="Comments cannot be empty."
            )

        # Generate predictions
        sentiments = predict_sentiments(request.comments)

        # Combine comments and predictions
        predictions = [
            {
                "comment": comment,
                "sentiment": sentiment
            }
            for comment, sentiment in zip(
                request.comments,
                sentiments
            )
        ]

        # Calculate sentiment statistics
        statistics = dict(Counter(sentiments))

        print("[PREDICT] Predictions completed successfully!")

        return {
            "predictions": predictions,
            "statistics": statistics
        }

    except HTTPException:
        raise

    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail="An error occurred during sentiment prediction."
        )