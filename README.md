# ChurnIQ — Customer Churn Prediction & Explainable ML

ChurnIQ is an end-to-end machine learning project that predicts customer churn using the IBM Telco Customer Churn sample dataset. It includes data preprocessing, model comparison, classification evaluation, customer-level predictions, and an interactive Streamlit dashboard.

## 🚀 Live Demo

**[Open ChurnIQ — Customer Churn Prediction & Explainable AI](https://churniq-ml-j427d9waeeimw95atpshqq.streamlit.app/)**

Explore customer churn patterns, review model performance, predict churn probabilities, and view feature importance through the interactive dashboard.

## ✨ Features

- Data validation and preprocessing
- Logistic Regression and Random Forest model comparison
- Stratified cross-validation for model selection
- Precision, recall, F1-score, ROC-AUC, and confusion matrix evaluation
- Interactive customer churn prediction
- Global feature-importance visualization
- Saved scikit-learn pipeline for consistent inference
- Streamlit web application

## 🛠️ Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Streamlit, Matplotlib
- **Model Persistence:** Joblib
- **Testing:** Pytest
- **Deployment:** Streamlit Community Cloud

## 📊 Dataset

The project uses IBM's public Telco Customer Churn sample dataset.

**Source:** [IBM Telco Customer Churn CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

The dataset contains customer demographics, subscribed services, contract information, charges, tenure, and a binary churn label.

The dataset is a public sample and does not represent live customer data. Check the source repository's license and attribution requirements before redistributing it.

## ⚙️ Run Locally

### Requirements

- Python 3.10–3.12 recommended
- Git
- VS Code recommended
- Internet connection for the initial dataset download

### Installation

Clone the repository:

```bash
git clone https://github.com/prachik2407/churniq-ml.git
cd churniq-ml
```

Create and activate a virtual environment on Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Download the dataset

```bash
python -m src.data
```

### Train and evaluate the models

```bash
python -m src.train
```

This command compares Logistic Regression and Random Forest using stratified cross-validation on the training set, selects a model based on mean cross-validation F1-score, and evaluates it on a held-out test set.

### Run tests

```bash
python -m pytest -q
```

### Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## 📁 Project Structure

```text
churniq-ml/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── README.md
├── models/
│   └── best_model.joblib
├── reports/
│   ├── best_model_metrics.json
│   ├── model_comparison.csv
│   └── feature_importance.csv
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

*Note: The model and report files are generated after training. Their presence in the repository depends on whether you commit the generated artifacts.*

## 🧠 Machine Learning Workflow

1. Load and validate the customer dataset.
2. Convert `TotalCharges` to numeric and handle missing values within the preprocessing pipeline.
3. Exclude `customerID` from predictive features.
4. Perform a stratified train/test split.
5. Preprocess numerical and categorical features using a scikit-learn `ColumnTransformer`.
6. Compare Logistic Regression and Random Forest using stratified cross-validation on training data.
7. Select the model based on mean cross-validation F1-score.
8. Evaluate the selected model on the held-out test set.
9. Save the fitted pipeline and evaluation artifacts.
10. Serve predictions and model insights through Streamlit.

## 📈 Evaluation Metrics

The project evaluates models using:

- **Precision:** Proportion of predicted churners who actually churned.
- **Recall:** Proportion of actual churners correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Ability to rank churners above non-churners across thresholds.
- **Confusion Matrix:** Breakdown of correct and incorrect classifications.

The selected model and its measured results are generated during training and saved in the `reports/` directory.

## ⚠️ Limitations

- Churn probabilities are estimates, not guarantees.
- The sample dataset may not generalize to other businesses or customer populations.
- Feature importance describes model behavior and does not establish causation.
- Classification performance depends on the data, evaluation methodology, and decision threshold.
- This project is a portfolio MVP, not a production customer-retention system.

## 🚀 Deployment

The application is deployed using Streamlit Community Cloud.

**Live application:** [ChurnIQ](https://churniq-ml-j427d9waeeimw95atpshqq.streamlit.app/)

To deploy your own instance, connect the GitHub repository to Streamlit Community Cloud and select `app.py` as the application entry point. Ensure the required dependencies, dataset, trained model, and evaluation artifacts are available to the deployed application.

## 🔮 Future Improvements

- Add XGBoost and additional model comparisons.
- Implement validation-based threshold tuning.
- Add SHAP-based local prediction explanations.
- Improve model monitoring and drift detection.
- Add automated retraining and experiment tracking.

---

Developed as an applied machine learning portfolio project using Python and Streamlit.
