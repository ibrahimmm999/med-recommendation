from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.prediction_service import predict_drug

router = APIRouter()

class PredictionRequest(BaseModel):
    complaint: str

class PredictionResponse(BaseModel):
    category: str
    drugs: list

@router.post("/predict", response_model=PredictionResponse)
def get_prediction(request: PredictionRequest):
    try:
        category, drugs = predict_drug(request.complaint)
        return PredictionResponse(category=category, drugs=drugs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
