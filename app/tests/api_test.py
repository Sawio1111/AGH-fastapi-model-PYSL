from fastapi.testclient import TestClient
from app.api.run import app

client = TestClient(app)

def test_get_model_params():
    response = client.get("iris/params")
    assert response.status_code == 200
    response_json = response.json()
    assert "n_estimators" in response_json
    assert "criterion" in response_json
    assert "max_depth" in response_json
    assert "min_samples_split" in response_json
    assert "min_samples_leaf" in response_json
    assert "max_features" in response_json
    assert "bootstrap" in response_json
    assert "random_state" in response_json

def test_get_features():
    response = client.get("iris/features")
    assert response.status_code == 200
    assert "features" in response.json()
    assert sorted(response.json()["features"]) == sorted([
        "sepal_length", "sepal_width", "petal_length", "petal_width"])

def test_predict_species():
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("iris/predict", json=test_data)
    assert response.status_code == 200
    assert "result" in response.json()

def test_get_species():
    response = client.get("iris/species")
    assert response.status_code == 200
    assert "species" in response.json()

def test_get_model_metrics():
    response = client.get("iris/model-info")
    assert response.status_code == 200
    assert "accuracy" in response.json()

def test_explain_prediction():
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("iris/predict/explain", json=test_data)
    assert response.status_code == 200
    assert "shap_explanation" in response.json()
    
def test_predict_species_invalid_data():
    invalid_data = {
        "sepal_length": -1,
        "sepal_width": -1,
        "petal_length": -1,
        "petal_width": -1
    }
    response = client.post("iris/predict", json=invalid_data)
    assert response.status_code == 422

def test_predict_species_missing_data():
    incomplete_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5
    }
    response = client.post("iris/predict", json=incomplete_data)
    assert response.status_code == 422