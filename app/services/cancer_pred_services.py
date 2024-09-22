# app/services/cancer_pred_services.py
import numpy as np
import pandas as pd
import joblib

# Load model, scaler, and feature names
model = joblib.load('app/artifacts/lung_cancer_model.pkl')
scaler = joblib.load('app/artifacts/scaler.pkl')
feature_names = joblib.load('app/artifacts/feature_names.pkl')

def predict_lung_cancer(data):
    # Extract features from the input data
    features = np.array([data.gender, data.age, data.smoking, data.yellow_fingers, data.anxiety,
                         data.peer_pressure, data.chronic_disease, data.fatigue, 
                         data.allergy, data.wheezing, data.alcohol_consuming, 
                         data.coughing, data.shortness_of_breath, 
                         data.swallowing_difficulty, data.chest_pain])
    
    # Convert features to a pandas DataFrame with proper column names
    features_df = pd.DataFrame([features], columns=feature_names)
    
    # Apply scaling
    features_scaled = scaler.transform(features_df)
    
    # Predict lung cancer risk
    prediction = model.predict(features_scaled)
    
    return int(prediction[0])
