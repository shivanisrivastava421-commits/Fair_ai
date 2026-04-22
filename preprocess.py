import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    df = df.copy()

    # -----------------------------
    # 1. HANDLE MISSING VALUES
    # -----------------------------
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # numeric column → mean fill
            df[col] = df[col].fillna(df[col].mean())
        else:
            # categorical column → most frequent value
            df[col] = df[col].fillna(df[col].mode()[0])

    # -----------------------------
    # 2. CONVERT CATEGORICAL → NUMERIC
    # -----------------------------
    le = LabelEncoder()

    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    return df