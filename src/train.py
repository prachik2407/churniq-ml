"""Train, cross-validate, select, evaluate, and save the ChurnIQ model."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from src.data import load_data
from src.preprocessing import make_models

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
RANDOM_STATE = 42


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = load_data()
    df = df.dropna(subset=["Churn"]).copy()
    y = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = make_models(RANDOM_STATE)
    rows = []
    fitted = {}
    for name, pipeline in candidates.items():
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring={"f1": "f1", "roc_auc": "roc_auc", "precision": "precision", "recall": "recall"},
            n_jobs=1, error_score="raise"
        )
        mean_f1 = float(scores["test_f1"].mean())
        rows.append({
            "model": name,
            "cv_f1_mean": mean_f1,
            "cv_f1_std": float(scores["test_f1"].std()),
            "cv_precision_mean": float(scores["test_precision"].mean()),
            "cv_recall_mean": float(scores["test_recall"].mean()),
            "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
        })
        fitted[name] = pipeline

    comparison = pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    best_name = str(comparison.iloc[0]["model"])
    best_model = fitted[best_name]
    best_model.fit(X_train, y_train)

    probabilities = best_model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "selected_model": best_name,
        "selection_method": "Highest mean F1 across 5-fold stratified cross-validation on training data",
        "test_size": int(len(y_test)),
        "test_metrics": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        },
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, target_names=["No Churn", "Churn"], output_dict=True, zero_division=0
        ),
        "random_state": RANDOM_STATE,
        "threshold": 0.5,
        "feature_columns": list(X.columns),
    }
    joblib.dump(
        {"model": best_model, "feature_columns": list(X.columns), "selected_model": best_name},
        MODELS_DIR / "best_model.joblib"
    )
    with open(REPORTS_DIR / "best_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Global model importance on the fitted pipeline, mapped to transformed features.
    preprocessor = best_model.named_steps["preprocessor"]
    classifier = best_model.named_steps["classifier"]
    try:
        feature_names = preprocessor.get_feature_names_out()
        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            importances = abs(classifier.coef_[0])
        else:
            importances = None
        if importances is not None and len(feature_names) == len(importances):
            pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
                "importance", ascending=False
            ).to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    except Exception as exc:
        print(f"Feature importance export skipped: {exc}")

    print(f"\nSelected model (by training CV F1): {best_name}")
    print("\nCross-validation comparison:")
    print(comparison.to_string(index=False))
    print("\nHeld-out test metrics:")
    for metric, value in metrics["test_metrics"].items():
        print(f"{metric}: {value:.4f}")
    print(f"\nSaved model: {MODELS_DIR / 'best_model.joblib'}")
    print(f"Saved reports in: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
