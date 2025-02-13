from fastapi import APIRouter, HTTPException

from app.api.view.schema_request import (
    IrisPredictRequest,
    IrisPredictResponse,
    IrisParamsResponse,
    IrisFeatureResponse,
    IrisSpeciesResponse,
    IrisModelMetricsResponse,
    IrisExplanationResponse,
)
from app.api.model.provider import iris_provider

iris_router = APIRouter()

@iris_router.get("/params", response_model=IrisParamsResponse)
async def get_model_params():
    try:
        params = await iris_provider.get_model_params()
        return IrisParamsResponse(**params)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Something was going wrong due to: {err}")

@iris_router.post("/predict", response_model=IrisPredictResponse)
async def predict_species(data: IrisPredictRequest):
    try:
        species = await iris_provider.predict_species(data=data)
        return IrisPredictResponse(**data.model_dump(), result=f"For your data we guess it is {species}")
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Prediction failed due to: {str(err)}")
    
@iris_router.get("/features", response_model=IrisFeatureResponse)
async def get_features():
    features = {
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        }
    return IrisFeatureResponse(features=features)

@iris_router.get("/species", response_model=IrisSpeciesResponse)
async def get_species():
    species = await iris_provider.get_species()
    return IrisSpeciesResponse(
        species=species
    )

@iris_router.get("/model-info", response_model=IrisModelMetricsResponse)
async def get_metrics():
    metrics = await iris_provider.get_metrics()
    print(metrics)
    return IrisModelMetricsResponse(**metrics)

@iris_router.post("/predict/explain")
async def explain(data: IrisPredictRequest):
    try:
        species = await iris_provider.predict_species(data=data)
        explanation = await iris_provider.get_explain(data=data)
        return IrisExplanationResponse(
            species=species,
            shap_explanation=explanation,
        )
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Prediction and explanation failed due to: {str(err)}")
