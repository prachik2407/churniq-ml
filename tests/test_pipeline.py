import pandas as pd
from src.preprocessing import make_models


def test_pipeline_fits_and_predicts():
    X = pd.DataFrame({
        "tenure": [1, 2, 3, 10, 20, 30, 5, 40, 60, 12, 15, 25],
        "MonthlyCharges": [20, 30, 35, 70, 80, 90, 25, 100, 110, 50, 55, 75],
        "Contract": ["Month-to-month", "Month-to-month", "One year", "Two year",
                     "Two year", "One year", "Month-to-month", "Two year",
                     "Two year", "One year", "Month-to-month", "One year"],
    })
    y = pd.Series([1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0])
    for model in make_models().values():
        model.fit(X, y)
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        assert len(predictions) == len(y)
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
