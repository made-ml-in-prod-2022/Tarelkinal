import pytest

from fastapi.testclient import TestClient
from dotenv import find_dotenv, load_dotenv

from online_inference.app import app


dotenv_path = find_dotenv()
load_dotenv(dotenv_path)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_healz(client):
    response = client.get('/healz')
    assert response.ok


def test_predict(client):
    request_dict = {
        "data": [
            [69, 1, 0, 160, 234, 1, 2, 131, 0, 0.1, 1, 1, 0]
        ],
        "features": [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        ]
    }
    response = client.get('/predict/', json=request_dict)
    assert pytest.approx(response.json()['prob']) == 0.438011
