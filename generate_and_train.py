#!/usr/bin/env python3
"""Generate synthetic student data and train Logistic Regression + XGBoost models."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import json
import os

np.random.seed(42)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(OUT_DIR, "data")
MODEL_DIR = os.path.join(OUT_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

N_STUDENTS = 2000

def generate_dataset(n=N_STUDENTS):
    """Generate realistic synthetic student dropout data."""
    # Features
    gpa = np.clip(np.random.normal(2.8, 0.7, n), 0.5, 4.0)
    attendance = np.clip(np.random.normal(82, 15, n), 20, 100)
    failed_courses = np.random.poisson(0.8, n).clip(0, 8)
    assignment_rate = np.clip(np.random.normal(0.78, 0.18, n), 0.1, 1.0)
    participation = np.clip(np.random.normal(0.65, 0.22, n), 0.05, 1.0)
    late_submissions = np.random.poisson(2.5, n).clip(0, 20)
    system_activity = np.clip(np.random.normal(0.55, 0.25, n), 0.0, 1.0)
    previous_warnings = np.random.poisson(0.6, n).clip(0, 6)
    absences = np.clip((100 - attendance) * 0.4 + np.random.normal(0, 3, n), 0, 60).astype(int)
    credits_completed = np.clip(np.random.normal(45, 25, n), 0, 120).astype(int)
    age = np.random.randint(18, 28, n)
    financial_aid = np.random.binomial(1, 0.4, n)
    first_gen = np.random.binomial(1, 0.35, n)
    online_ratio = np.clip(np.random.beta(2, 3, n), 0, 1)

    # Latent risk score (higher = more dropout risk)
    risk_latent = (
        -1.8 * (gpa - 2.0) / 2.0
        - 1.5 * (attendance - 50) / 50
        + 0.9 * failed_courses
        - 1.2 * assignment_rate
        - 0.8 * participation
        + 0.35 * late_submissions
        - 0.7 * system_activity
        + 0.8 * previous_warnings
        + 0.25 * absences / 10
        - 0.15 * credits_completed / 30
        + 0.3 * first_gen
        - 0.2 * financial_aid
        + 0.4 * online_ratio
        + np.random.normal(0, 0.6, n)
    )

    # Probabilities via sigmoid
    dropout_prob = 1 / (1 + np.exp(-risk_latent))
    dropout = (np.random.rand(n) < dropout_prob).astype(int)

    names_first = ["أحمد", "محمد", "فاطمة", "نورة", "عبدالله", "سارة", "خالد", "ريم", "يوسف", "لينا",
                   "عمر", "هدى", "سعيد", "مها", "تركي", "دانة", "فهد", "جواهر", "بندر", "العنود"]
    names_last = ["العتيبي", "القحطاني", "الحربي", "الشمري", "الدوسري", "المطيري", "الغامدي", "الزهراني",
                  "السبيعي", "البقمي", "الرشيدي", "العنزي", "الخالدي", "الفهيد", "النجدي"]

    student_ids = [f"S{2024000 + i}" for i in range(n)]
    names = [f"{np.random.choice(names_first)} {np.random.choice(names_last)}" for _ in range(n)]

    df = pd.DataFrame({
        "student_id": student_ids,
        "name": names,
        "gpa": np.round(gpa, 2),
        "attendance": np.round(attendance, 1),
        "failed_courses": failed_courses,
        "assignment_rate": np.round(assignment_rate, 2),
        "participation": np.round(participation, 2),
        "late_submissions": late_submissions,
        "system_activity": np.round(system_activity, 2),
        "previous_warnings": previous_warnings,
        "absences": absences,
        "credits_completed": credits_completed,
        "age": age,
        "financial_aid": financial_aid,
        "first_generation": first_gen,
        "online_ratio": np.round(online_ratio, 2),
        "dropout": dropout,
        "true_prob": np.round(dropout_prob, 3),
    })
    return df

def train_models(df):
    feature_cols = [
        "gpa", "attendance", "failed_courses", "assignment_rate", "participation",
        "late_submissions", "system_activity", "previous_warnings", "absences",
        "credits_completed", "age", "financial_aid", "first_generation", "online_ratio"
    ]
    X = df[feature_cols]
    y = df["dropout"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Logistic Regression pipeline
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)
    lr_proba = lr_pipe.predict_proba(X_test)[:, 1]

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

    def metrics(y_true, y_pred, y_proba):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
        }

    lr_metrics = metrics(y_test, lr_pred, lr_proba)
    xgb_metrics = metrics(y_test, xgb_pred, xgb_proba)

    # Feature importance (XGBoost)
    importance = dict(zip(feature_cols, xgb_model.feature_importances_.tolist()))
    importance = {k: float(v) for k, v in sorted(importance.items(), key=lambda x: -x[1])}

    # LR coefficients (absolute for ranking)
    lr_coefs = dict(zip(feature_cols, np.abs(lr_pipe.named_steps["clf"].coef_[0]).tolist()))
    lr_coefs = {k: float(v) for k, v in sorted(lr_coefs.items(), key=lambda x: -x[1])}

    results = {
        "logistic_regression": lr_metrics,
        "xgboost": xgb_metrics,
        "feature_importance_xgb": importance,
        "feature_importance_lr": lr_coefs,
        "feature_names": feature_cols,
        "best_model": "xgboost" if xgb_metrics["f1"] >= lr_metrics["f1"] else "logistic_regression",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "dropout_rate": float(y.mean()),
    }

    # Save models
    joblib.dump(lr_pipe, os.path.join(MODEL_DIR, "logistic_regression.joblib"))
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgboost.joblib"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.joblib"))

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Also save full dataset with predictions from best model
    best = xgb_model if results["best_model"] == "xgboost" else lr_pipe
    if results["best_model"] == "xgboost":
        proba = best.predict_proba(X)[:, 1]
    else:
        proba = best.predict_proba(X)[:, 1]

    df["dropout_prob"] = np.round(proba, 3)
    df["risk_score"] = (df["dropout_prob"] * 100).round(1)

    def risk_level(p):
        if p < 0.25:
            return "منخفض"
        elif p < 0.50:
            return "متوسط"
        elif p < 0.75:
            return "مرتفع"
        else:
            return "حرج"

    df["risk_level"] = df["dropout_prob"].apply(risk_level)
    df["status"] = df["dropout"].map({0: "مستمر", 1: "تسرب"})

    df.to_csv(os.path.join(DATA_DIR, "students.csv"), index=False, encoding="utf-8-sig")
    print("Dataset saved:", len(df), "students")
    print("Metrics:", json.dumps(results, indent=2, ensure_ascii=False))
    print("Best model:", results["best_model"])
    return df, results

if __name__ == "__main__":
    print("Generating dataset...")
    df = generate_dataset()
    print("Training models...")
    train_models(df)
    print("Done.")
