# Real-Time ML Inference REST API

## Project Overview
This project uses a trained deep learning neural network to classify SMS messages as spam or ham (normal message).

## Technologies Used
- Python
- TensorFlow / Keras
- Scikit-learn
- TF-IDF text vectorization
- FastAPI
- Pydantic
- Docker
- Pytest

## System Architecture
1. The user submits an SMS message as JSON.
2. FastAPI validates the input.
3. The saved TF-IDF vectorizer converts text into numerical features.
4. The trained neural network predicts the spam probability.
5. The API returns the prediction and confidence as JSON.

## API Endpoints

### GET /
Checks whether the API is running.

### POST /predict
Request:
```json
{
  "message": "Congratulations! You won a prize!"
}
```

Example response:
```json
{
  "message": "Congratulations! You won a prize!",
  "prediction": "spam",
  "confidence": 0.98,
  "spam_probability": 0.98
}
```

The values above are illustrative; actual predictions depend on the trained model.

## Run Locally
Install dependencies:
```bash
pip install -r requirements.txt
```

Start the API:
```bash
uvicorn app:app --reload
```

Open the interactive API documentation at:
http://127.0.0.1:8000/docs

## Run with Docker
Build the image:
```bash
docker build -t spam-classifier-api .
```

Run the container:
```bash
docker run -p 8000:8000 spam-classifier-api
```

## Run Unit Tests
```bash
pytest -v test_app.py
```

## Model Artifacts
- `spam_model.keras`: trained neural network
- `tfidf_vectorizer.joblib`: text vectorizer
- `label_encoder.joblib`: label encoder

## Limitations
Predictions depend on the training dataset and may be incorrect for unfamiliar messages. Confidence scores are model outputs, not guaranteed probabilities of correctness.

## Author
Mohammed Aziz Khan
