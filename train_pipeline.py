# train_pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import inspect
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os

# === Update path to your dataset ===
CSV_PATH = "/Users/suyog/Downloads/ML_Projects/Insurance_Risk_Analysis/Insurance_claims_data.csv"

def load_data(path):
    df = pd.read_csv(path)
    return df

def select_features(df, target_col="claim_status"):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    # Build a version-compatible OneHotEncoder
    encoder_kwargs = {"handle_unknown": "ignore"}
    try:
        # sklearn >= 1.2 uses sparse_output
        if "sparse_output" in inspect.signature(OneHotEncoder.__init__).parameters:
            encoder_kwargs["sparse_output"] = False
        else:
            # older versions use sparse
            encoder_kwargs["sparse"] = False
    except (ValueError, TypeError):
        # very old versions fallback
        encoder_kwargs["sparse"] = False

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
        ('onehot', OneHotEncoder(**encoder_kwargs))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', cat_transformer, cat_cols)
    ], remainder='drop')

    return preprocessor

def train_model(csv_path=CSV_PATH, model_out='models/fraud_model.joblib'):
    df = load_data(csv_path)
    X, y = select_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(X_train)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', clf)
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, model_out)
    print(f"Saved full pipeline to {model_out}")

    return pipeline, X_test, y_test

if __name__ == '__main__':
    pipeline, X_test, y_test = train_model()
