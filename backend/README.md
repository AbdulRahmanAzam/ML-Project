# Fake Job Detector - Backend

FastAPI backend for the Fake Job Posting Detector.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API server from the `backend` directory:
```bash
cd backend
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/predict` - Predict if job posting is fake

## Files

- `api.py` - FastAPI application
- `tfidf_vectorizer.pkl` - TF-IDF vectorizer model
- `fake_job_detector_model.h5` - Keras neural network model
- `project.ipynb` - Training notebook
- `fake_job_postings.csv` - Training dataset
