"""
Feature Engineering Pipeline for Customer Churn Prediction
Transforms raw data into model-ready features with proper preprocessing.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib
import os
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Complete feature engineering pipeline with:
    - Missing value imputation
    - Categorical encoding (one-hot & label)
    - Numerical scaling
    - Class imbalance handling (SMOTE)
    - Train/test splitting
    """
    
    def __init__(self, target_col: str = "Churn", test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            target_col: Name of the target variable
            test_size: Proportion for test split (default: 0.2)
            random_state: Random seed for reproducibility
        """
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        
        # Store feature names and transformers
        self.numerical_features = None
        self.categorical_features = None
        self.binary_features = None
        self.preprocessor = None
        self.label_encoder = LabelEncoder()
        
        # Store processed data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_resampled = None
        self.y_train_resampled = None
        
        # Metadata
        self.feature_names = None
        self.class_distribution = {}
        
        print(f"✅ FeatureEngineer initialized (target: {target_col}, test_size: {test_size})")
    
    def identify_feature_types(self, df: pd.DataFrame) -> Dict[str, list]:
        """
        Automatically identify numerical, categorical, and binary features.
        """
        # Exclude target column
        features = [col for col in df.columns if col != self.target_col]
        
        # Identify numerical features
        numerical = []
        categorical = []
        binary = []
        
        for col in features:
            if df[col].dtype in ['int64', 'float64']:
                numerical.append(col)
            else:
                # Check if it's binary (2 unique values)
                unique_vals = df[col].nunique()
                if unique_vals <= 2:
                    binary.append(col)
                else:
                    categorical.append(col)
        
        # Special handling for SeniorCitizen (binary but numeric)
        if 'SeniorCitizen' in numerical:
            numerical.remove('SeniorCitizen')
            binary.append('SeniorCitizen')
        
        return {
            'numerical': numerical,
            'categorical': categorical,
            'binary': binary
        }
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Initial data preparation: handle missing values and convert data types.
        """
        print("🔧 Preparing data...")
        
        # Create a copy
        data = df.copy()
        
        # Convert TotalCharges to numeric (handling empty strings)
        if 'TotalCharges' in data.columns:
            data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        
        # Handle missing values
        missing_before = data.isnull().sum().sum()
        if missing_before > 0:
            print(f"   Found {missing_before} missing values")
            
            # Impute numerical columns with median
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if data[col].isnull().sum() > 0:
                    median_val = data[col].median()
                    data[col].fillna(median_val, inplace=True)
                    print(f"   • Imputed {col} with median: {median_val:.2f}")
            
            # Impute categorical columns with mode
            categorical_cols = data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if data[col].isnull().sum() > 0:
                    mode_val = data[col].mode()[0]
                    data[col].fillna(mode_val, inplace=True)
                    print(f"   • Imputed {col} with mode: {mode_val}")
        
        # Convert binary columns to 0/1
        binary_mapping = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
        for col in data.columns:
            if data[col].dtype == 'object' and data[col].nunique() == 2:
                unique_vals = data[col].unique()
                if set(unique_vals).issubset(set(binary_mapping.keys())):
                    data[col] = data[col].map(binary_mapping)
                    print(f"   • Encoded binary column: {col} → {unique_vals}")
        
        print("✅ Data preparation complete")
        return data
    
    def build_preprocessor(self, feature_types: Dict[str, list]) -> ColumnTransformer:
        """
        Build sklearn ColumnTransformer for different feature types.
        """
        print("🔨 Building preprocessing pipeline...")
        
        # Numerical pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline (one-hot encoding)
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Binary pipeline (already encoded as 0/1, just impute if needed)
        binary_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])
        
        # Combine pipelines
        preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, feature_types['numerical']),
            ('cat', categorical_pipeline, feature_types['categorical']),
            ('bin', binary_pipeline, feature_types['binary'])
        ])
        
        print(f"   • Numerical features: {len(feature_types['numerical'])}")
        print(f"   • Categorical features: {len(feature_types['categorical'])}")
        print(f"   • Binary features: {len(feature_types['binary'])}")
        
        return preprocessor
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Main method: fit and transform the entire dataset.
        Returns processed features and target.
        """
        print("\n" + "="*60)
        print("🚀 STARTING FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        # Step 1: Prepare data
        data = self.prepare_data(df)
        
        # Step 2: Identify feature types
        feature_types = self.identify_feature_types(data)
        self.numerical_features = feature_types['numerical']
        self.categorical_features = feature_types['categorical']
        self.binary_features = feature_types['binary']
        
        # Step 3: Separate features and target
        X = data.drop(columns=[self.target_col])
        y = data[self.target_col]
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_distribution = dict(zip(self.label_encoder.classes_, 
                                         np.bincount(y_encoded)))
        
        print(f"🎯 Target distribution: {self.class_distribution}")
        
        # Step 4: Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y_encoded
        )
        
        print(f"📊 Train size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")
        
        # Step 5: Build and fit preprocessor
        self.preprocessor = self.build_preprocessor(feature_types)
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Step 6: Get feature names
        self.feature_names = self._get_feature_names()
        print(f"🔤 Total features after encoding: {len(self.feature_names)}")
        
        # Convert to DataFrames
        X_train_df = pd.DataFrame(X_train_processed, columns=self.feature_names)
        X_test_df = pd.DataFrame(X_test_processed, columns=self.feature_names)
        
        # Store processed data
        self.X_train = X_train_df
        self.X_test = X_test_df
        self.y_train = y_train
        self.y_test = y_test
        
        print("\n✅ Feature engineering complete!")
        print(f"   • Original shape: {df.shape}")
        print(f"   • Processed train shape: {X_train_df.shape}")
        print(f"   • Processed test shape: {X_test_df.shape}")
        print("="*60)
        
        return X_train_df, X_test_df, y_train, y_test
    
    def handle_imbalance(self, method: str = 'smote') -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Handle class imbalance using SMOTE.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Must call fit_transform() before handling imbalance")
        
        print(f"\n⚖️  Handling class imbalance using {method.upper()}...")
        
        original_counts = np.bincount(self.y_train)
        print(f"   Before: Class 0 = {original_counts[0]:,} | Class 1 = {original_counts[1]:,}")
        print(f"   Imbalance ratio: {original_counts[1]/original_counts[0]:.2f}")
        
        if method.lower() == 'smote':
            smote = SMOTE(random_state=self.random_state)
            X_resampled, y_resampled = smote.fit_resample(self.X_train, self.y_train)
        else:
            # Default to no resampling
            X_resampled, y_resampled = self.X_train, self.y_train
        
        resampled_counts = np.bincount(y_resampled)
        print(f"   After:  Class 0 = {resampled_counts[0]:,} | Class 1 = {resampled_counts[1]:,}")
        print(f"   Imbalance ratio: {resampled_counts[1]/resampled_counts[0]:.2f}")
        
        self.X_train_resampled = X_resampled
        self.y_train_resampled = y_resampled
        
        return X_resampled, y_resampled
    
    def _get_feature_names(self) -> list:
        """Extract feature names after preprocessing."""
        if self.preprocessor is None:
            return []
        
        feature_names = []
        
        # Numerical features
        feature_names.extend(self.numerical_features)
        
        # Categorical features (one-hot encoded)
        for i, col in enumerate(self.categorical_features):
            # Get categories from one-hot encoder
            cat_pipeline = self.preprocessor.named_transformers_['cat']
            onehot = cat_pipeline.named_steps['onehot']
            
            if hasattr(onehot, 'get_feature_names_out'):
                cat_features = onehot.get_feature_names_out([col])
                feature_names.extend(cat_features)
            else:
                # Fallback: create generic names
                unique_vals = self.preprocessor.transformers_[1][2][i]
                for val in unique_vals:
                    feature_names.append(f"{col}_{val}")
        
        # Binary features
        feature_names.extend(self.binary_features)
        
        return feature_names
    
    def save_pipeline(self, output_dir: str = "models"):
        """
        Save the preprocessing pipeline and metadata.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save preprocessor
        preprocessor_path = os.path.join(output_dir, "preprocessor.pkl")
        joblib.dump(self.preprocessor, preprocessor_path)
        
        # Save label encoder
        encoder_path = os.path.join(output_dir, "label_encoder.pkl")
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'binary_features': self.binary_features,
            'class_distribution': self.class_distribution,
            'target_col': self.target_col
        }
        
        metadata_path = os.path.join(output_dir, "feature_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        print(f"💾 Saved pipeline to {output_dir}/")
        print(f"   • preprocessor.pkl")
        print(f"   • label_encoder.pkl")
        print(f"   • feature_metadata.pkl")
    
    def load_pipeline(self, input_dir: str = "models"):
        """
        Load a saved preprocessing pipeline.
        """
        preprocessor_path = os.path.join(input_dir, "preprocessor.pkl")
        encoder_path = os.path.join(input_dir, "label_encoder.pkl")
        metadata_path = os.path.join(input_dir, "feature_metadata.pkl")
        
        if not all(os.path.exists(p) for p in [preprocessor_path, encoder_path, metadata_path]):
            raise FileNotFoundError(f"Pipeline files not found in {input_dir}")
        
        self.preprocessor = joblib.load(preprocessor_path)
        self.label_encoder = joblib.load(encoder_path)
        metadata = joblib.load(metadata_path)
        
        self.feature_names = metadata['feature_names']
        self.numerical_features = metadata['numerical_features']
        self.categorical_features = metadata['categorical_features']
        self.binary_features = metadata['binary_features']
        self.class_distribution = metadata['class_distribution']
        self.target_col = metadata['target_col']
        
        print(f"📂 Loaded pipeline from {input_dir}/")


def test_feature_engineering():
    """
    Test function to demonstrate the feature engineering pipeline.
    """
    print("🧪 Testing Feature Engineering Pipeline...")
    
    # Load sample data
    from data_loader import DataLoader
    
    loader = DataLoader()
    df = loader.load_data()
    
    if df is None:
        print("❌ Could not load data. Skipping test.")
        return None
    
    # Initialize feature engineer
    engineer = FeatureEngineer(target_col="Churn", test_size=0.2, random_state=42)
    
    # Run full pipeline
    X_train, X_test, y_train, y_test = engineer.fit_transform(df)
    
    # Handle imbalance
    X_train_balanced, y_train_balanced = engineer.handle_imbalance(method='smote')
    
    # Save pipeline
    engineer.save_pipeline()
    
    # Print sample of processed data
    print("\n📋 Sample of processed features (first 5 rows):")
    print(X_train.head())
    
    print("\n📊 Feature summary:")
    print(f"• Total features: {len(engineer.feature_names)}")
    print(f"• Numerical: {len(engineer.numerical_features)}")
    print(f"• Categorical (encoded): {len(engineer.categorical_features)}")
    print(f"• Binary: {len(engineer.binary_features)}")
    
    return engineer, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    test_feature_engineering()
