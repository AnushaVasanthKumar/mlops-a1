# misc.py
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

def load_boston() -> pd.DataFrame:
    """Load Boston Housing from a local CSV (created by CI), with HTTPS fallback."""
    local = Path("data/BostonHousing.csv")
    if local.exists():
        df = pd.read_csv(local)
    else:
        # Fallback only for local runs; CI will provide the CSV
        url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
        df = pd.read_csv(url)

    # Normalize column names and ensure target is 'MEDV'
    cols = {c.lower(): c for c in df.columns}
    lower_map = {k.lower(): k for k in df.columns}
    if "medv" in lower_map and "MEDV" not in df.columns:
        df.rename(columns={lower_map["medv"]: "MEDV"}, inplace=True)
    if "chas" in lower_map and "CHAS" not in df.columns:
        df.rename(columns={lower_map["chas"]: "CHAS"}, inplace=True)
    if "ptratio" in lower_map and "PTRATIO" not in df.columns:
        df.rename(columns={lower_map["ptratio"]: "PTRATIO"}, inplace=True)
    return df

def make_xy(df: pd.DataFrame, target_col: str = "MEDV"):
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
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scores = cross_val_score(build_pipeline(estimator, True), X, y,
                             cv=cv, scoring="neg_mean_squared_error")
    return float(np.mean(-scores))

def train_test_holdout_mse(estimator, X, y, test_size: float = 0.2, random_state: int = 42, scale: bool = True) -> float:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)
    pipe = build_pipeline(estimator, scale=scale)
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    return float(mean_squared_error(y_te, preds))
