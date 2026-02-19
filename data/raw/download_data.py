"""
Downloads the official Telco Customer Churn dataset directly into data/raw/
Source: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
"""
import pandas as pd
import os

def download_dataset():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    output_path = os.path.join("data", "raw", "telco_churn_raw.csv")

    print("⬇️  Downloading Telco Churn dataset...")
    try:
        df = pd.read_csv(url)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ Success! Saved {len(df):,} rows to {output_path}")
        print(f"   First 3 rows:")
        print(df.head(3))
        return df
    except Exception as e:
        print(f"❌ Failed to download: {e}")
        print("💡 Try uploading the CSV manually to `data/raw/` instead.")
        return None

if __name__ == "__main__":
    download_dataset()
