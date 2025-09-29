# utils_detect_target.py
import pandas as pd

COMMON_TARGET_NAMES = ['is_fraud','fraud','fraud_flag','label','target','is_fraudulent','fraudulent']

def detect_target(df: pd.DataFrame):
    for name in COMMON_TARGET_NAMES:
        if name in df.columns:
            return name
    # heuristics: binary columns with 0/1 values
    for col in df.columns:
        unique = df[col].dropna().unique()
        if set(unique).issubset({0,1}) and df[col].nunique() <= 3:
            return col
    return None
