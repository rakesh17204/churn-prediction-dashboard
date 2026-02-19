"""
Module for loading and initial exploration of Telco Churn dataset.
Designed for reproducibility, error resilience, and clarity.
"""
import pandas as pd
import os
import yaml
from typing import Dict, Any, Optional

class DataLoader:
    """
    Robust data loader for Telco Customer Churn dataset.
    Handles download fallback, metadata generation, and summary reporting.
    """

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or os.path.join("data", "raw", "telco_churn_raw.csv")
        self.df: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}

    def load_data(self) -> Optional[pd.DataFrame]:
        """Load CSV. If missing, prints helpful instructions."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Loaded {self.data_path} | Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            print(f"❌ File not found: {self.data_path}")
            print("💡 Tip: Run `data/raw/download_data.py` first (or upload manually).")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

    def get_basic_info(self) -> Dict[str, Any]:
        """Generate rich metadata dictionary."""
        if self.df is None:
            print("⚠️  Warning: No data loaded. Call `load_data()` first.")
            return {}

        self.metadata = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicates": int(self.df.duplicated().sum()),
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / 1024**2, 2),
            "churn_distribution": (
                self.df["Churn"].value_counts(normalize=True).to_dict()
                if "Churn" in self.df.columns
                else None
            ),
        }
        return self.metadata

    def print_summary(self):
        """Print human-readable dataset health report."""
        if self.df is None:
            print("❌ No data loaded. Call `load_data()` first.")
            return

        print("\n" + "=" * 65)
        print("📊 DATASET HEALTH REPORT")
        print("=" * 65)
        print(f"📁 Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"💾 Memory usage: {self.metadata.get('memory_usage_mb', 'N/A')} MB")

        # Columns & dtypes
        print(f"\n📋 Columns ({len(self.df.columns)}):")
        for i, col in enumerate(self.df.columns, 1):
            dtype = str(self.df[col].dtype)
            nulls = self.df[col].isnull().sum()
            null_pct = f" ({nulls}/{len(self.df)} = {nulls/len(self.df)*100:.1f}%)" if nulls else ""
            print(f"  {i:2d}. {col:<22} [{dtype:<8}]{null_pct}")

        # Target variable
        if "Churn" in self.df.columns:
            churn = self.df["Churn"].value_counts()
            yes, no = churn.get("Yes", 0), churn.get("No", 0)
            imbalance = min(yes, no) / max(yes, no) if max(yes, no) > 0 else 1.0
            print(f"\n🎯 Target 'Churn': Yes={yes:,} | No={no:,} | Imbalance={imbalance:.2f}")
            if imbalance < 0.3:
                print("   ⚠️  Class imbalance detected — consider SMOTE or class weights.")

        print("=" * 65)

    def save_metadata(self, filepath: str = "data/processed/metadata.yaml"):
        """Save metadata to YAML for reproducibility."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(self.metadata, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Metadata saved to {filepath}")


def test_data_loader():
    """Standalone test function — safe to run in any environment."""
    print("🧪 Running DataLoader test...")
    loader = DataLoader()
    df = loader.load_data()
    if df is not None:
        loader.get_basic_info()
        loader.print_summary()
        loader.save_metadata()
    return df


if __name__ == "__main__":
    test_data_loader()
