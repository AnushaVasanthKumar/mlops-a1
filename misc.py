# misc.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

def load_boston_manual() -> pd.DataFrame:
    """
    Recreate Boston Housing dataset from CMU source (as per assignment note).
    Returns a DataFrame with features and target column 'MEDV'.
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)

    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    feature_names = [
        "CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD",
        "TAX","PTRATIO","B","LSTAT"
    ]
    df = pd.DataFrame(data, columns=feature_names)
    df["MEDV"] = target
    return df

def make_xy(df: pd.DataFrame, target_col: str = "MEDV") -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def build_pipeline(estimator, scale: bool = True) -> Pipeline:
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)

def evaluate_cv(estimator, X, y, cv_splits: int = 5) -> float:
    """
    Returns average CV MSE (positive number).
    """
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scores = cross_val_score(
        build_pipeline(estimator, scale=True),
        X, y,
        cv=cv,
        scoring="neg_mean_squared_error"
    )
    return float(np.mean(-scores))

def train_test_holdout_mse(estimator, X, y, test_size: float = 0.2, random_state: int = 42, scale: bool = True) -> float:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)
    pipe = build_pipeline(estimator, scale=scale)
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    return float(mean_squared_error(y_te, preds))
