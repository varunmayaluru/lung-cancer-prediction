# streamlit_app.py
import streamlit as st
import numpy as np
import joblib
from app.services.cancer_pred_services import predict_lung_cancer
from app.models import LungCancerInput

# Load the trained model and scaler
model = joblib.load('app/artifacts/lung_cancer_model.pkl')
scaler = joblib.load('app/artifacts/scaler.pkl')

# Function to get user inputs from the Streamlit interface
def get_user_input():
    gender = st.selectbox('Gender', [0, 1])
    age = st.slider('Age', 20, 100, 60)
    smoking = st.selectbox('Smoking', [0, 1])
    yellow_fingers = st.selectbox('Yellow Fingers', [0, 1])
    anxiety = st.selectbox('Anxiety', [0, 1])
    peer_pressure = st.selectbox('Peer Pressure', [0, 1])
    chronic_disease = st.selectbox('Chronic Disease', [0, 1])
    fatigue = st.selectbox('Fatigue', [0, 1])
    allergy = st.selectbox('Allergy', [0, 1])
    wheezing = st.selectbox('Wheezing', [0, 1])
    alcohol_consuming = st.selectbox('Alcohol Consuming', [0, 1])
    coughing = st.selectbox('Coughing', [0, 1])
    shortness_of_breath = st.selectbox('Shortness of Breath', [0, 1])
    swallowing_difficulty = st.selectbox('Swallowing Difficulty', [0, 1])
    chest_pain = st.selectbox('Chest Pain', [0, 1])

    # Create a dictionary with all the inputs
    user_data = {
        'gender' : gender,
        'age': age,
        'smoking': smoking,
        'yellow_fingers': yellow_fingers,
        'anxiety': anxiety,
        'peer_pressure': peer_pressure,
        'chronic_disease': chronic_disease,
        'fatigue': fatigue,
        'allergy': allergy,
        'wheezing': wheezing,
        'alcohol_consuming': alcohol_consuming,
        'coughing': coughing,
        'shortness_of_breath': shortness_of_breath,
        'swallowing_difficulty': swallowing_difficulty,
        'chest_pain': chest_pain
    }
    return user_data

# Main Streamlit app
def main():
    st.title("Lung Cancer Prediction App")

    # Get user input
    user_data = get_user_input()

    # Button to predict lung cancer
    if st.button("Predict"):
        # Convert user input into a LungCancerInput model
        input_data = LungCancerInput(**user_data)

        # Call prediction service
        prediction = predict_lung_cancer(input_data)
        
        # Display prediction result
        if prediction == 1:
            st.error("High Risk of Lung Cancer!")
        else:
            st.success("Low Risk of Lung Cancer")

if __name__ == "__main__":
    main()
