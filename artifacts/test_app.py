from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_spam_prediction():
    response = client.post(
        "/predict",
        json={"message": "Congratulations! You won a free prize!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["spam", "ham"]
    assert 0 <= data["confidence"] <= 1
    assert 0 <= data["spam_probability"] <= 1


def test_normal_message():
    response = client.post(
        "/predict",
        json={"message": "Hi, see you tomorrow."}
    )
    assert response.status_code == 200
    assert response.json()["prediction"] in ["spam", "ham"]


def test_empty_message():
    response = client.post("/predict", json={"message": ""})
    assert response.status_code == 422


def test_missing_message():
    response = client.post("/predict", json={})
    assert response.status_code == 422
