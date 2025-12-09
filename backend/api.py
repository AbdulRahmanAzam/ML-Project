"""
FastAPI for Fake Job Detection Model.
Run: uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
import os
from typing import Optional

# For text preprocessing
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
try:
	nltk.data.find('corpora/stopwords')
except LookupError:
	nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Initialize FastAPI app
app = FastAPI(
    title="Fake Job Detector API",
    description="API for detecting fraudulent job postings using Machine Learning",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORIZER_PATH = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'fake_job_detector_model.h5')

# OPTIMAL THRESHOLD - Update this after training your model
# This is typically lower than 0.5 for imbalanced datasets
OPTIMAL_THRESHOLD = 0.7218  # Can be auto-updated by the training notebook

tfidf_vectorizer = None
model = None

# Load TF-IDF Vectorizer
if os.path.exists(VECTORIZER_PATH):
	tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
	print(f"✓ TF-IDF Vectorizer loaded from {VECTORIZER_PATH}")
else:
	print(f"⚠ Warning: Vectorizer file '{VECTORIZER_PATH}' not found.")

# Load Keras Model
try:
	if os.path.exists(MODEL_PATH):
		from keras.models import load_model
		model = load_model(MODEL_PATH)
		print(f"✓ Keras Model loaded from {MODEL_PATH}")
	else:
		print(f"⚠ Warning: Model file '{MODEL_PATH}' not found.")
except Exception as e:
	print(f"⚠ Warning: Failed to load model: {e}")


def clean_text(text: str, preserve_fraud_signals: bool = True) -> str:
	"""
	Clean text using the same preprocessing as training.
	Enhanced to preserve fraud indicators.
	"""
	# Parse HTML
	soup = BeautifulSoup(text, "html.parser")
	raw_text = soup.get_text()

	fraud_signals = []

	if preserve_fraud_signals:
		# Extract fraud indicators BEFORE cleaning
		fraud_signals.append('has_url' if re.search(r'http[s]?://|www\.', raw_text) else 'no_url')
		fraud_signals.append('has_email' if re.search(r'\S+@\S+\.\S+', raw_text) else 'no_email')
		fraud_signals.append('has_phone' if re.search(r'\d{10,}|\(\d{3}\)\s*\d{3}[-.\s]?\d{4}|\d{3}[-.\s]\d{3}[-.\s]\d{4}', raw_text) else 'no_phone')
		fraud_signals.append('has_money' if re.search(r'\$\d+|\d+\s*(dollar|usd|euro|£|€)|earn|income|salary|pay', raw_text, re.I) else 'no_money')

		exclamation_count = len(re.findall(r'!', raw_text))
		fraud_signals.append('high_urgency' if exclamation_count > 3 else 'low_urgency')

		scam_keywords = ['immediate', 'urgent', 'hurry', 'limited', 'guaranteed', 'no experience', 'work from home', 'easy money']
		has_scam_words = any(kw in raw_text.lower() for kw in scam_keywords)
		fraud_signals.append('has_scam_words' if has_scam_words else 'no_scam_words')

	# Remove text in square brackets
	text = re.sub(r'\[[^]]*\]', '', raw_text)

	# Remove any character that is not a letter and convert to lowercase
	text = re.sub(r'[^a-zA-Z\s]', '', text).lower()

	# Remove stopwords
	stop = set(stopwords.words('english'))
	text = ' '.join([word for word in text.split() if word not in stop])

	# Lemmatize
	lemmatizer = WordNetLemmatizer()
	text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

	# Append fraud signals as features
	if preserve_fraud_signals and fraud_signals:
		text = text + ' ' + ' '.join(fraud_signals)

	return text


# Pydantic models for request/response validation
class JobPostingRequest(BaseModel):
	title: str
	description: str
	company_profile: str
	requirements: str
	benefits: str
	location: str
	has_company_logo: int  # 1 or 0


class PredictionResponse(BaseModel):
	prediction: str
	is_fake: bool
	fraud_probability: float
	real_probability: float
	confidence: float
	confidence_level: str
	message: str


class HealthResponse(BaseModel):
	status: str
	model_loaded: bool
	vectorizer_loaded: bool
	threshold: float


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
	"""Health check endpoint."""
	return HealthResponse(
		status="healthy",
		model_loaded=model is not None,
		vectorizer_loaded=tfidf_vectorizer is not None,
		threshold=float(OPTIMAL_THRESHOLD),
	)


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(job: JobPostingRequest):
	"""
	Predict whether a job posting is fake or real.
	"""
	if model is None or tfidf_vectorizer is None:
		raise HTTPException(
			status_code=500,
			detail="Model or Vectorizer not loaded. Please run the notebook to train and save the model."
		)

	try:
		# Convert has_company_logo to text (matching training format)
		logo_text = 'company has logo yes' if job.has_company_logo == 1 else 'company has logo no'

		# Combine all text fields (matching the training data format)
		combined_text = ' '.join([
			job.title,
			job.location,
			job.company_profile,
			job.description,
			job.requirements,
			job.benefits,
			logo_text
		])

		# Clean the text using the same preprocessing as training (with fraud signals)
		cleaned_text = clean_text(combined_text, preserve_fraud_signals=True)

		# Vectorize using TF-IDF
		text_tfidf = tfidf_vectorizer.transform([cleaned_text])

		# Make prediction with Keras model (convert sparse to dense)
		prediction_proba = float(model.predict(text_tfidf.toarray(), verbose=0)[0][0])

		# Use OPTIMAL THRESHOLD instead of 0.5
		prediction = 1 if prediction_proba >= OPTIMAL_THRESHOLD else 0

		# Calculate probabilities
		fraud_probability = prediction_proba
		real_probability = 1.0 - fraud_probability

		# Determine confidence level
		confidence = max(fraud_probability, real_probability)
		if confidence >= 0.85:
			confidence_level = 'Very High'
		elif confidence >= 0.70:
			confidence_level = 'High'
		elif confidence >= 0.55:
			confidence_level = 'Medium'
		else:
			confidence_level = 'Low'

		return PredictionResponse(
			prediction='fake' if prediction == 1 else 'real',
			is_fake=bool(prediction == 1),
			fraud_probability=round(fraud_probability * 100, 2),
			real_probability=round(real_probability * 100, 2),
			confidence=round(confidence * 100, 2),
			confidence_level=confidence_level,
			message='⚠️ This job posting appears to be FRAUDULENT. Be cautious!' if prediction == 1
					else '✓ This job posting appears to be LEGITIMATE.'
		)

	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# Startup event
@app.on_event("startup")
async def startup_event():
	print("\n" + "=" * 60)
	print("FAKE JOB DETECTOR API (FastAPI)")
	print("=" * 60)
	print("Server running on http://localhost:8000")
	print("API Documentation: http://localhost:8000/docs")
	print("Model loaded:", model is not None)
	print("Vectorizer loaded:", tfidf_vectorizer is not None)
	print(f"Using threshold: {OPTIMAL_THRESHOLD}")
	print("Endpoints:")
	print("  - GET  /api/health   - Health check")
	print("  - POST /api/predict  - Make prediction")
	print("=" * 60 + "\n")

