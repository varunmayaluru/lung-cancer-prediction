# tests/test_services.py
import unittest
import numpy as np
from app.services.cancer_pred_services import predict_lung_cancer
from app.models import LungCancerInput

class TestLungCancerPrediction(unittest.TestCase):

    def setUp(self):
        # Example input data
        self.data = LungCancerInput(
            gender=1,
            age=60,
            smoking=1,
            yellow_fingers=0,
            anxiety=1,
            peer_pressure=1,
            chronic_disease=0,
            fatigue=1,
            allergy=0,
            wheezing=1,
            alcohol_consuming=0,
            coughing=1,
            shortness_of_breath=1,
            swallowing_difficulty=0,
            chest_pain=1
        )

    def test_prediction_high_risk(self):
        # Test the high-risk prediction scenario
        prediction = predict_lung_cancer(self.data)
        self.assertEqual(prediction, 1)  # Expect high risk (1)

    def test_prediction_low_risk(self):
        # Modify input for a low-risk prediction scenario
        self.data = LungCancerInput(
            gender=0,
            age=30,
            smoking=0,
            yellow_fingers=0,
            anxiety=0,
            peer_pressure=0,
            chronic_disease=0,
            fatigue=0,
            allergy=0,
            wheezing=0,
            alcohol_consuming=0,
            coughing=0,
            shortness_of_breath=0,
            swallowing_difficulty=0,
            chest_pain=0
        )
        prediction = predict_lung_cancer(self.data)
        self.assertEqual(prediction, 0)  # Expect low risk (0)

if __name__ == "__main__":
    unittest.main()
