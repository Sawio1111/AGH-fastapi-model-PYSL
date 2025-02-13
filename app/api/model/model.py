import joblib
"""
This module loads a pre-trained Iris model and its associated metrics.

Modules:
    joblib: A library for serializing and deserializing Python objects.
    json: A library for parsing JSON data.

Constants:
    PATH_MODEL (str): The file path to the serialized Iris model.
    PATH_METRICS (str): The file path to the JSON file containing model metrics.

Variables:
    iris_model: The deserialized Iris model loaded from PATH_MODEL.
    iris_metrics: A dictionary containing the model metrics loaded from PATH_METRICS.
"""
import json

PATH_MODEL = "./app/api/model/iris_model_fastapi.pkl"
PATH_METRICS = "./app/api/model/model_metrics.json"

iris_model = joblib.load(PATH_MODEL)

with open(PATH_METRICS, "r") as f:
    iris_metrics = json.load(f)
