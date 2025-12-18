import pytest
from fastapi.testclient import TestClient


from app.main import app

client = TestClient(app)

def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200

    data = resp.json()
    assert "status" in data
    assert "model loaded" in data


def test_similarity_basic():
    payload = {
        "sentence1":"I love Machine Learning",
        "sentence2":"I enjoy studing machine learning"
    }


    resp = client.post("/similarity",json=payload)
    assert resp.status_code == 200

    data = resp.json()

    assert "cosine similarity" in data
    assert "sentence1" in data
    assert "sentence2" in data
    assert len(data["sentence1"]["embedding"]) > 0
    assert len(data["sentence2"]["embedding"]) > 0


def test_validation_empty_sentence():
    payload = {
        "sentence1": "",
        "sentence2": "Valid sentence",
    }
    resp = client.post("/similarity", json=payload)
    assert resp.status_code == 422
    