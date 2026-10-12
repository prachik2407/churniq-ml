# ChurnIQ — Customer Churn Prediction & Explainable ML

ChurnIQ is an end-to-end binary classification project using the IBM Telco Customer Churn sample dataset. It includes data validation, preprocessing, cross-validated model selection, held-out test evaluation, a saved scikit-learn pipeline, and a Streamlit dashboard.

## Requirements
- Python 3.10–3.12 recommended
- Internet connection for the first dataset download
- VS Code recommended for development

## Quick start (Windows PowerShell)

```powershell
# 1. Extract the ZIP, then open the churniq-ml folder in VS Code.
# 2. In the VS Code terminal:
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Download the public dataset
python -m src.data

# 4. Train models, compare them using cross-validation, and evaluate the selected model on a held-out test set
python -m src.train

# 5. Run tests
python -m pytest -q

# 6. Start the dashboard
streamlit run app.py
```

If `py` is not recognized, use `python` instead. If PowerShell blocks activation, use:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.data
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Dataset
The app uses IBM's public Telco Customer Churn sample CSV. `python -m src.data` attempts to download it from the source URL configured in `src/data.py` and saves it to `data/Telco-Customer-Churn.csv`. If that URL is temporarily unavailable, download the CSV manually from the IBM repository or its linked sample dataset and save it under that exact filename.

Source URL used by this project:
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

The dataset is a public sample, not live customer data. Check the source repository's license and attribution requirements before redistributing the raw CSV.

## Project structure

```text
churniq-ml/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── README.md
├── models/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
└── tests/
    ├── test_data.py
    └── test_pipeline.py
```

## What it does
- Validates the expected target column and converts `TotalCharges` to numeric.
- Excludes `customerID` from predictive features.
- Uses a stratified train/test split and a preprocessing `Pipeline` to reduce leakage risk.
- Compares Logistic Regression and Random Forest using stratified cross-validation on the training set.
- Reports precision, recall, F1, ROC-AUC, confusion matrix, and classification report on the held-out test set.
- Saves the chosen pipeline and evaluation artifacts for the Streamlit app.
- Displays dataset exploration, evaluation metrics, customer predictions, and global model feature importance.

## Important limitations
- Churn scores are estimates, not guarantees or causal explanations.
- The sample dataset may not represent other companies or current customers.
- Model performance depends on the dataset and the selected evaluation setup.
- Feature importance describes model behavior; it does not prove causation.
- This project is a portfolio MVP, not a production customer-retention system.

## Troubleshooting
- **Dataset download fails:** confirm internet access; manually save the CSV to `data/Telco-Customer-Churn.csv`.
- **Dashboard says model is missing:** run `python -m src.train` successfully before starting Streamlit.
- **Dependency errors:** use Python 3.10–3.12 in a fresh virtual environment and reinstall `requirements.txt`.
- **Deployment:** push the code and required artifacts to GitHub, then deploy `app.py` through Streamlit Community Cloud. The app needs access to the dataset for the Overview tab and to `models/best_model.joblib` for predictions.
