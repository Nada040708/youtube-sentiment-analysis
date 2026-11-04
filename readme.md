# YouTube Sentiment Analysis - MLOps Project

## Project Description

This project aims to develop an end-to-end MLOps system for automatically analyzing the sentiment of YouTube video comments.

The system uses a Machine Learning model based on TF-IDF and Logistic Regression to classify comments according to their sentiment.

## Architecture

- Frontend: Chrome Extension (HTML, CSS, JavaScript)
- Backend: FastAPI REST API
- ML Model: Scikit-learn (TF-IDF + Logistic Regression)
- Deployment: Hugging Face Spaces with Docker
- Version Control: Git and GitHub

## Project Structure

- data/: Raw and processed datasets
- models/: Trained ML models
- src/: Source code
- chrome-extension/: Chrome extension
- logs/: Application logs
- tests/: Automated tests

## Installation

Clone the repository and create a virtual environment:

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt