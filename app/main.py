# app/main.py
from fastapi import FastAPI
from app.routers.cancer_pred_routers import router as lung_cancer_router

# Initialize FastAPI app
app = FastAPI()

# Include the router from routers.py
app.include_router(lung_cancer_router)

