"""
Model Trainer: Train XGBoost model with cross-validation and save best model.
"""
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import os
import yaml
from typing import Tuple, Dict, Any


class ModelTrainer:
    """
    Trains and evaluates machine learning model using trained features.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self.load_config(config_path)
        self.model = None
        self.results = {}
        
        print(f"✅ ModelTrainer initialized with config: {config_path}")

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> XGBClassifier:
        """
        Train XGBoost model with hyperparameter tuning and cross-validation.
        """
        print("\n" + "="*60)
        print("🚀 TRAINING MODEL: XGBoost with Cross-Validation")
        print("="*60)

        # Define model
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,  # Adjust for imbalance
            random_state=self.config['random_state']
        )

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config['random_state'])
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')

        print(f"📊 CV AUC Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"• Min: {cv_scores.min():.4f}, Max: {cv_scores.max():.4f}")

        # Fit final model
        model.fit(X_train, y_train)
        self.model = model

        # Evaluate on test set
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        self.results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }

        print(f"\n🎯 TEST SET PERFORMANCE:")
        print(f"• Accuracy: {accuracy:.4f}")
        print(f"• Precision: {precision:.4f}")
        print(f"• Recall: {recall:.4f}")
        print(f"• F1-Score: {f1:.4f}")
        print(f"• AUC: {auc:.4f}")

        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔢 Confusion Matrix:")
        print(cm)

        return model

    def save_model(self, output_path: str = "models/model.pkl"):
        """Save trained model and results."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(self.model, output_path)
        print(f"💾 Model saved to {output_path}")

        # Save results
        results_path = output_path.replace(".pkl", "_results.json")
        import json
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"📊 Results saved to {results_path}")

    def evaluate_and_plot(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Generate evaluation plots (AUC, confusion matrix)."""
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        if self.model is None:
            raise ValueError("Model not trained yet.")

        y_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()

        plot_path = "reports/roc_curve.png"
        os.makedirs("reports", exist_ok=True)
        plt.savefig(plot_path)
        print(f"📈 ROC Curve saved to {plot_path}")
        plt.close()


def test_model_trainer():
    """Test function to verify training pipeline."""
    print("🧪 Testing Model Trainer...")
    
    from src.feature_engineering import FeatureEngineer
    
    # Load data
    loader = FeatureEngineer()
    df = loader.load_data()
    if df is None:
        print("❌ Data not loaded.")
        return
    
    # Run preprocessing
    X_train, X_test, y_train, y_test = loader.fit_transform(df)
    
    # Handle imbalance
    X_train_balanced, y_train_balanced = loader.handle_imbalance(method='smote')
    
    # Train model
    trainer = ModelTrainer()
    model = trainer.train(X_train_balanced, y_train_balanced, X_test, y_test)
    
    # Save model
    trainer.save_model()
    
    # Plot ROC
    trainer.evaluate_and_plot(X_test, y_test)
    
    print("\n✅ Model training and evaluation complete!")
    return trainer


if __name__ == "__main__":
    test_model_trainer()
