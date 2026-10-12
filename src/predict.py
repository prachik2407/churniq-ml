"""Inference helpers."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"


def load_artifact(path=MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError("Model artifact not found. Run `python -m src.train` first.")
    return joblib.load(path)


def predict_customer(customer: dict, artifact=None):
    artifact = artifact or load_artifact()
    model = artifact["model"]
    columns = artifact["feature_columns"]
    row = pd.DataFrame([{col: customer.get(col, None) for col in columns}], columns=columns)
    probability = float(model.predict_proba(row)[0, 1])
    return {"probability": probability, "prediction": int(probability >= 0.5)}
