import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

DATA_PATH = "data/churn.csv"
MODEL_PATH = "models/best_model.joblib"


def make_fake_data(n=5000, seed=42):
    """Used only if data/churn.csv is missing, so the project still runs."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "tenure": rng.integers(1, 72, n),
        "MonthlyCharges": rng.uniform(20, 120, n).round(2),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2]),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2]),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n),
        "TechSupport": rng.choice(["Yes", "No"], n),
    })
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2)
    score = (
        -0.04 * df["tenure"]
        + 0.02 * df["MonthlyCharges"]
        + 1.2 * (df["Contract"] == "Month-to-month")
        + 0.6 * (df["InternetService"] == "Fiber optic")
        + 0.5 * (df["PaymentMethod"] == "Electronic check")
        - 0.5 * (df["TechSupport"] == "Yes")
        - 0.5
    )
    prob = 1 / (1 + np.exp(-score))
    df["Churn"] = np.where(rng.random(n) < prob, "Yes", "No")
    return df


def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print("Loaded", DATA_PATH)
    else:
        print("No data/churn.csv found, using generated data")
        df = make_fake_data()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])
    return X, y


def build_preprocessor(X):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42,
        ),
    }

    best_name, best_auc, best_pipe = None, 0, None

    for name, model in models.items():
        pipe = Pipeline([("prep", build_preprocessor(X)), ("model", model)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, proba)
        print(f"\n===== {name} | ROC-AUC: {auc:.4f} =====")
        print(classification_report(y_test, preds))
        if auc > best_auc:
            best_name, best_auc, best_pipe = name, auc, pipe

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipe, MODEL_PATH)
    print(f"\nBest model: {best_name} (AUC {best_auc:.4f}) saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
