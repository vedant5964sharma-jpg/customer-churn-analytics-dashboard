from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://raw.githubusercontent.com/Giskard-AI/examples/main/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv"
FINAL_THRESHOLD = 0.40

BINARY_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]
BINARY_MAP = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
CATEGORICAL_COLS = ["InternetService", "Contract", "PaymentMethod"]
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges"]


@dataclass
class ChurnArtifacts:
    model: LogisticRegression
    scaler: StandardScaler
    threshold: float
    feature_names: list[str]
    metrics: dict


def load_data(url: str = DATA_URL) -> pd.DataFrame:
    with urlopen(url, timeout=30) as response:
        raw = response.read()
    return pd.read_csv(BytesIO(raw))


def prepare_features(raw: pd.DataFrame, keep_target: bool = False) -> pd.DataFrame:
    df = raw.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Preserve the notebook's service-category cleanup.
    df = df.replace({"No internet service": "No", "No phone service": "No"})

    # Preserve the notebook's final modeling choice: TotalCharges was dropped.
    if "TotalCharges" in df.columns:
        df = df.drop(columns=["TotalCharges"])

    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map(BINARY_MAP)

    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True, dtype=int)
    if not keep_target and "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
    return df


def train_model(raw: pd.DataFrame) -> ChurnArtifacts:
    prepared = prepare_features(raw, keep_target=True)
    X = prepared.drop(columns=["Churn"])
    y = prepared["Churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        {"C": [0.05, 0.1, 0.5, 1, 2]},
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X_train_scaled, y_train)
    model = grid.best_estimator_

    proba = model.predict_proba(X_test_scaled)[:, 1]
    pred = (proba >= FINAL_THRESHOLD).astype(int)
    cm = confusion_matrix(y_test, pred)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "recall": float(recall_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "cv_roc_auc": float(grid.best_score_),
        "best_c": float(grid.best_params_["C"]),
        "confusion_matrix": cm.tolist(),
        "test_size": int(len(y_test)),
        "test_churn_rate": float(y_test.mean()),
    }
    return ChurnArtifacts(model, scaler, FINAL_THRESHOLD, list(X.columns), metrics)


def save_artifacts(artifacts: ChurnArtifacts, directory: str = "artifacts") -> None:
    import os
    os.makedirs(directory, exist_ok=True)
    joblib.dump(artifacts.model, f"{directory}/churn_model.pkl")
    joblib.dump(artifacts.scaler, f"{directory}/scaler.pkl")
    joblib.dump(artifacts.threshold, f"{directory}/threshold.pkl")
    joblib.dump(artifacts.feature_names, f"{directory}/feature_names.pkl")
    joblib.dump(artifacts.metrics, f"{directory}/metrics.pkl")


def load_artifacts(directory: str = "artifacts") -> ChurnArtifacts:
    return ChurnArtifacts(
        joblib.load(f"{directory}/churn_model.pkl"),
        joblib.load(f"{directory}/scaler.pkl"),
        joblib.load(f"{directory}/threshold.pkl"),
        joblib.load(f"{directory}/feature_names.pkl"),
        joblib.load(f"{directory}/metrics.pkl"),
    )


def align_for_model(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in feature_names:
        if col not in out.columns:
            out[col] = 0
    return out[feature_names]


def predict_customer(raw_customer: pd.DataFrame, artifacts: ChurnArtifacts) -> tuple[float, int]:
    X = prepare_features(raw_customer, keep_target=False)
    X = align_for_model(X, artifacts.feature_names)
    scaled = artifacts.scaler.transform(X)
    probability = float(artifacts.model.predict_proba(scaled)[0, 1])
    return probability, int(probability >= artifacts.threshold)


def coefficient_drivers(artifacts: ChurnArtifacts, top_n: int = 8) -> pd.DataFrame:
    coef = artifacts.model.coef_[0]
    out = pd.DataFrame({"feature": artifacts.feature_names, "coefficient": coef})
    out["direction"] = np.where(out["coefficient"] >= 0, "Higher churn probability", "Lower churn probability")
    out["magnitude"] = out["coefficient"].abs()
    return out.sort_values("magnitude", ascending=False).head(top_n)
