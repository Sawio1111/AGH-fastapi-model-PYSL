from enum import Enum
from pydantic import BaseModel

class IrisSpecies(Enum):
    SETOSA = 0
    VERSICOLOR = 1
    VIRGINICA = 2

class IrisPredict(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float