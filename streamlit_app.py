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
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.slider('Age', 20, 100, 60)
    smoking = st.selectbox('Smoking', ['Yes', 'No'])
    yellow_fingers = st.selectbox('Yellow Fingers', ['Yes', 'No'])
    anxiety = st.selectbox('Anxiety', ['Yes', 'No'])
    peer_pressure = st.selectbox('Peer Pressure', ['Yes', 'No'])
    chronic_disease = st.selectbox('Chronic Disease', ['Yes', 'No'])
    fatigue = st.selectbox('Fatigue', ['Yes', 'No'])
    allergy = st.selectbox('Allergy', ['Yes', 'No'])
    wheezing = st.selectbox('Wheezing', ['Yes', 'No'])
    alcohol_consuming = st.selectbox('Alcohol Consuming', ['Yes', 'No'])
    coughing = st.selectbox('Coughing', ['Yes', 'No'])
    shortness_of_breath = st.selectbox('Shortness of Breath', ['Yes', 'No'])
    swallowing_difficulty = st.selectbox('Swallowing Difficulty', ['Yes', 'No'])
    chest_pain = st.selectbox('Chest Pain', ['Yes', 'No'])

    # Map user-friendly input back to the original values (0 or 1)
    gender_mapping = {'Male': 1, 'Female': 0}
    yes_no_mapping = {'Yes': 1, 'No': 0}

    # Create a dictionary with all the inputs mapped to numerical values
    user_data = {
        'gender': gender_mapping[gender],
        'age': age,
        'smoking': yes_no_mapping[smoking],
        'yellow_fingers': yes_no_mapping[yellow_fingers],
        'anxiety': yes_no_mapping[anxiety],
        'peer_pressure': yes_no_mapping[peer_pressure],
        'chronic_disease': yes_no_mapping[chronic_disease],
        'fatigue': yes_no_mapping[fatigue],
        'allergy': yes_no_mapping[allergy],
        'wheezing': yes_no_mapping[wheezing],
        'alcohol_consuming': yes_no_mapping[alcohol_consuming],
        'coughing': yes_no_mapping[coughing],
        'shortness_of_breath': yes_no_mapping[shortness_of_breath],
        'swallowing_difficulty': yes_no_mapping[swallowing_difficulty],
        'chest_pain': yes_no_mapping[chest_pain]
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
