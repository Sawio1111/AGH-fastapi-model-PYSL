import shap
import numpy as np
from typing import List
from sklearn.base import BaseEstimator

from app.api.model.schema import IrisSpecies, IrisPredict
from app.api.model.model import iris_model, iris_metrics

class IrisProvider:
    
    def __init__(self):
        self._model: BaseEstimator = iris_model
        self._metrics: dict = iris_metrics
    
    async def predict_species(self, data: IrisPredict) -> str:
        input = self._transform_to_model_data(data)
        result = self._model.predict(input)
        return IrisSpecies(result).name

    async def get_model_params(self) -> dict:
        return self._model.get_params()
    
    async def get_species(self) -> List[str]:
        return [IrisSpecies(species).name for species in self._model.classes_]

    async def get_metrics(self) -> dict:
        return iris_metrics
    
    async def get_explain(self, data: IrisPredict) -> List[List[List[float]]]:
        explainer = shap.Explainer(self._model)
        
        input_data = self._transform_to_model_data(data)
        shap_values = explainer(input_data)
        return shap_values.values.tolist()
        
    def _transform_to_model_data(self, data) -> List[List[int]]:
        return np.array(list(data.model_dump().values())).reshape(1, -1)

iris_provider = IrisProvider()
