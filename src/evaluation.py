"""
SHAP Explainability & Evaluation Module
"""
import shap
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os
from typing import List, Tuple


class ModelEvaluator:
    """
    Generate SHAP explanations and visualizations for model interpretability.
    """

    def __init__(self, model_path: str = "models/model.pkl", feature_names: List[str] = None):
        self.model = joblib.load(model_path)
        self.feature_names = feature_names or []
        self.explainer = None
        
        print(f"✅ ModelEvaluator loaded model from {model_path}")

    def explain_global(self, X_train: pd.DataFrame, top_features: int = 20):
        """
        Global SHAP summary plot.
        """
        print("\n🔍 Generating global SHAP explanation...")

        # Create explainer
        self.explainer = shap.TreeExplainer(self.model)

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_train)

        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_train, feature_names=self.feature_names, max_display=top_features)
        plt.title(f"Global Feature Importance (Top {top_features})")
        plt.tight_layout()

        plot_path = "reports/shap_summary.png"
        os.makedirs("reports", exist_ok=True)
        plt.savefig(plot_path)
        print(f"📊 SHAP summary saved to {plot_path}")
        plt.close()

    def explain_local(self, X_test: pd.DataFrame, instance_idx: int = 0):
        """
        Local SHAP explanation for a single instance.
        """
        print(f"\n🔍 Explaining prediction for instance {instance_idx}...")

        # Get single row
        instance = X_test.iloc[instance_idx:instance_idx+1]

        # Compute SHAP values
        shap_values = self.explainer.shap_values(instance)

        # Plot
        plt.figure(figsize=(8, 6))
        shap.plots.beeswarm(self.explainer.shap_values(instance), feature_names=self.feature_names)
        plt.title(f"Local SHAP Explanation (Instance {instance_idx})")
        plt.tight_layout()

        plot_path = f"reports/shap_local_{instance_idx}.png"
        plt.savefig(plot_path)
        print(f"📊 Local SHAP saved to {plot_path}")
        plt.close()

    def save_explanation(self, output_path: str = "reports/explanation.html"):
        """Save HTML report."""
        import shap
        if self.explainer is None:
            raise ValueError("Explainer not initialized.")

        # This requires full dataset
        # Example: shap.summary_plot(...) → export to HTML
        print(f"📌 Note: Full SHAP HTML report requires more setup. Use `shap.initjs()` + `shap.force_plot()`.")
        print(f"💡 For now, use PNGs. Later: integrate with Streamlit or Jupyter.")


def test_evaluator():
    """Test evaluator with sample data."""
    print("🧪 Testing ModelEvaluator...")
    
    from src.feature_engineering import FeatureEngineer
    
    # Load data
    loader = FeatureEngineer()
    df = loader.load_data()
    if df is None:
        print("❌ Data not loaded.")
        return
    
    # Preprocess
    X_train, X_test, y_train, y_test = loader.fit_transform(df)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_path="models/model.pkl", feature_names=loader.feature_names)
    
    # Generate global explanation
    evaluator.explain_global(X_train, top_features=15)
    
    # Generate local explanation
    evaluator.explain_local(X_test, instance_idx=0)
    
    print("\n✅ SHAP explanations generated!")


if __name__ == "__main__":
    test_evaluator()
