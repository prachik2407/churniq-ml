import pandas as pd
import pytest
from src.data import load_data


def test_load_data_converts_total_charges_and_excludes_no_columns(tmp_path):
    path = tmp_path / "tiny.csv"
    pd.DataFrame({
        "customerID": ["A", "B"],
        "Churn": ["No", "Yes"],
        "tenure": [1, 2],
        "MonthlyCharges": [20.0, 30.0],
        "TotalCharges": ["", "60.0"],
    }).to_csv(path, index=False)
    df = load_data(path, download=False)
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert pd.isna(df.loc[0, "TotalCharges"])
    assert len(df) == 2


def test_missing_required_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Churn": ["No"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_data(path, download=False)
