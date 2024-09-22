# app/routers.py
from fastapi import APIRouter
from app.models import LungCancerInput
from app.services.cancer_pred_services import predict_lung_cancer

# Initialize router
router = APIRouter()

# Define root endpount
@router.get('/')
async def root():
    return "Welcome to Lung Cancer Prediction App"

# Define the prediction endpoint
@router.post("/predict/")
async def predict(data: LungCancerInput):
    # Call the prediction service
    result = predict_lung_cancer(data)
    return {"lung_cancer_prediction": result}
