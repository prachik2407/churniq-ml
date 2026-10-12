"""Dataset loading and validation utilities for ChurnIQ."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "Telco-Customer-Churn.csv"
DATA_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/main/data/Telco-Customer-Churn.csv",
]
REQUIRED_COLUMNS = {"Churn", "customerID", "tenure", "MonthlyCharges", "TotalCharges"}


def load_data(path: str | Path = DATA_PATH, download: bool = True) -> pd.DataFrame:
    """Load local CSV, downloading the public sample if needed."""
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
    elif download:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        last_error = None
        for url in DATA_URLS:
            try:
                df = pd.read_csv(url)
                df.to_csv(path, index=False)
                break
            except Exception as exc:
                last_error = exc
        else:
            raise FileNotFoundError(
                f"Could not download the dataset. Download the CSV manually and save it to {path}. "
                f"Last error: {last_error}"
            ) from last_error
    else:
        raise FileNotFoundError(f"Dataset not found: {path}")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Remove exact duplicate records, reporting is handled by the caller/UI.
    df = df.drop_duplicates().reset_index(drop=True)
    if not set(df["Churn"].dropna().astype(str).unique()).issubset({"Yes", "No"}):
        raise ValueError("Expected Churn values to be 'Yes' or 'No'.")
    return df


def main():
    df = load_data()
    print(f"Dataset saved/loaded: {DATA_PATH}")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    print("\nChurn distribution:")
    print(df["Churn"].value_counts(dropna=False))
    print("\nMissing values (top 10):")
    print(df.isna().sum().sort_values(ascending=False).head(10))


if __name__ == "__main__":
    main()
