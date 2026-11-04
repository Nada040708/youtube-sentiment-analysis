FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

COPY requirements-prod.txt .

RUN pip install --no-cache-dir -r requirements-prod.txt

COPY app_api.py .

COPY models/best_model.joblib models/
COPY models/tfidf_vectorizer.joblib models/

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app_api:app --host 0.0.0.0 --port ${PORT:-7860}"]