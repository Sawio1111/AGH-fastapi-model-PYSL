from pydantic import BaseModel, field_validator
from typing import Optional, List

class IrisPredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
    @field_validator('sepal_length', 'sepal_width', 'petal_length', 'petal_width')
    def values_must_be_positive(cls, value):
        if value <= 0:
            raise ValueError('All measurements must be positive')
        return value
    
class IrisPredictResponse(IrisPredictRequest):
    result: str

class IrisParamsResponse(BaseModel):
    n_estimators: int
    criterion: str
    max_depth: Optional[int]
    min_samples_split: int
    min_samples_leaf: int
    max_features: str
    bootstrap: bool
    random_state: Optional[int]

class IrisFeatureResponse(BaseModel):
    features: List[str]

class IrisSpeciesResponse(BaseModel):
    species: List[str]
    
class IrisModelMetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]

class IrisExplanationResponse(BaseModel):
    species: str
    shap_explanation: List[List[List[float]]]
