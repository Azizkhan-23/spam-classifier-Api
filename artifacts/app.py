import joblib
import numpy as np
import tensorflow as tf

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Spam Message Classifier API",
    description="Predicts whether a text message is spam or ham.",
    version="1.0.0"
)

model = tf.keras.models.load_model("artifacts/spam_model.keras")
vectorizer = joblib.load("artifacts/tfidf_vectorizer.joblib")
encoder = joblib.load("artifacts/label_encoder.joblib")


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)


class PredictionResponse(BaseModel):
    message: str
    prediction: str
    confidence: float
    spam_probability: float


@app.get("/")
def home():
    return {"message": "Spam Classifier API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: MessageRequest):
    features = vectorizer.transform([request.message]).toarray()
    spam_probability = float(model.predict(features, verbose=0)[0][0])

    spam_label = next(
        (i for i, label in enumerate(encoder.classes_)
         if label.lower() == "spam"),
        None
    )

    if spam_label is None:
        raise ValueError("Spam label not found in encoder classes.")

    # The model's sigmoid output corresponds to encoder class 1.
    # Confirm that the encoder maps class 1 to spam.
    if len(encoder.classes_) != 2 or encoder.classes_[1].lower() != "spam":
        raise ValueError("Unexpected label encoding for this model.")

    prediction = "spam" if spam_probability >= 0.5 else "ham"
    confidence = spam_probability if prediction == "spam" else 1 - spam_probability

    return {
        "message": request.message,
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "spam_probability": round(spam_probability, 4)
    }
