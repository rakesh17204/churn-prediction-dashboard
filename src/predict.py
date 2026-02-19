"""
Prediction Service: Load pipeline and model, transform input, predict.
"""
import joblib
import pandas as pd
from typing import Dict, Any, List


class Predictor:
    """
    End-to-end predictor for customer churn.
    Loads preprocessor and model, transforms input, returns predictions.
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.preprocessor = joblib.load(f"{model_dir}/preprocessor.pkl")
        self.label_encoder = joblib.load(f"{model_dir}/label_encoder.pkl")
        self.model = joblib.load(f"{model_dir}/model.pkl")
        self.feature_names = joblib.load(f"{model_dir}/feature_metadata.pkl")['feature_names']

        print(f"✅ Predictor loaded from {model_dir}/")

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict churn probability for a single customer.
        
        Args:
            input_data: Dictionary of raw feature values
            
        Returns:
            Dictionary with prediction, probability, and recommendation
        """
        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Transform using saved preprocessor
        try:
            processed = self.preprocessor.transform(df)
        except Exception as e:
            raise ValueError(f"Preprocessing failed: {e}")

        # Make prediction
        pred = self.model.predict(processed)[0]
        prob = self.model.predict_proba(processed)[0][1]

        # Decode label
        churn_label = self.label_encoder.inverse_transform([pred])[0]

        # Recommendation
        threshold = 0.5
        recommendation = "Offer retention discount" if prob > threshold else "No action needed"

        return {
            "churn_risk": churn_label,
            "probability": round(prob, 4),
            "recommendation": recommendation,
            "threshold": threshold
        }


def test_predictor():
    """Test the predictor."""
    print("🧪 Testing Predictor...")
    
    # Sample input
    new_customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.85,
        "TotalCharges": 958.10
    }

    # Initialize predictor
    predictor = Predictor()

    # Predict
    result = predictor.predict(new_customer)

    print("\n🎯 PREDICTION RESULT:")
    for k, v in result.items():
        print(f"• {k}: {v}")

    return result


if __name__ == "__main__":
    test_predictor()
