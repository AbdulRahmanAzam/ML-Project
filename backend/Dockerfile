FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for NLTK and potential others)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (to avoid runtime downloads)
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# Copy the rest of the code and model files
COPY . .

# Expose port 8000 (matches your uvicorn command)
EXPOSE 8000

# Run the app with uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]