from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.data import load_data
from src.predict import MODEL_PATH, predict_customer

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

st.set_page_config(page_title="ChurnIQ | Customer Churn ML", page_icon="📊", layout="wide")

st.title("📊 ChurnIQ")
st.caption("Customer churn prediction and explainable machine learning")
st.info("This portfolio demo uses a public sample dataset. Predictions are estimates, not guarantees.")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Model Performance", "Customer Prediction", "Model Insights"]
)

@st.cache_data
def get_data():
    return load_data()

def get_metrics():
    path = REPORTS / "best_model_metrics.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def get_artifact():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

try:
    df = get_data()
except Exception as exc:
    df = None
    st.error(f"Dataset unavailable: {exc}")
    st.markdown("Place `Telco-Customer-Churn.csv` inside the `data/` folder or run `python -m src.data`.")

metrics = get_metrics()
artifact = get_artifact()

if page == "Overview":
    st.header("Dataset overview")
    if df is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Customer records", f"{len(df):,}")
        c2.metric("Features (excluding ID/target)", str(len(df.columns) - 2))
        churn_rate = (df["Churn"].astype(str) == "Yes").mean() * 100
        c3.metric("Churn rate", f"{churn_rate:.1f}%")
        left, right = st.columns(2)
        with left:
            st.subheader("Churn distribution")
            counts = df["Churn"].value_counts().reindex(["No", "Yes"]).fillna(0)
            st.bar_chart(counts)
        with right:
            st.subheader("Monthly charges by churn")
            if "MonthlyCharges" in df and "Churn" in df:
                st.bar_chart(df.groupby("Churn")["MonthlyCharges"].mean())
        st.subheader("Preview")
        st.dataframe(df.head(20), use_container_width=True)
        st.subheader("Missing values")
        missing = df.isna().sum().sort_values(ascending=False)
        st.dataframe(missing[missing > 0].rename("missing_count").to_frame(), use_container_width=True)
    else:
        st.warning("Load the dataset to view the overview.")

elif page == "Model Performance":
    st.header("Model performance")
    if metrics is None:
        st.warning("No saved evaluation report found. Train the model first: `python -m src.train`.")
    else:
        st.caption(f"Selected model: **{metrics['selected_model']}** · Selection based on training-set cross-validation F1.")
        m = metrics["test_metrics"]
        cols = st.columns(5)
        for col, key, label in zip(cols, ["precision", "recall", "f1", "roc_auc", "accuracy"],
                                   ["Precision", "Recall", "F1-score", "ROC-AUC", "Accuracy"]):
            col.metric(label, f"{m[key]:.3f}")
        cm = metrics["confusion_matrix"]
        st.subheader("Confusion matrix")
        st.dataframe(pd.DataFrame(cm, index=["Actual: No Churn", "Actual: Churn"],
                                  columns=["Predicted: No Churn", "Predicted: Churn"]),
                     use_container_width=True)
        comparison_path = REPORTS / "model_comparison.csv"
        if comparison_path.exists():
            st.subheader("Training cross-validation comparison")
            st.dataframe(pd.read_csv(comparison_path), use_container_width=True)
        st.caption("The metrics above are from a held-out test split. Cross-validation results are shown separately.")

elif page == "Customer Prediction":
    st.header("Predict churn for a customer")
    if artifact is None:
        st.warning("The trained model is missing. In the terminal, run `python -m src.train` first.")
    else:
        with st.form("customer_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
                partner = st.selectbox("Has partner?", ["Yes", "No"])
                dependents = st.selectbox("Has dependents?", ["Yes", "No"])
                tenure = st.slider("Tenure (months)", 0, 72, 12)
            with c2:
                phone = st.selectbox("Phone service", ["Yes", "No"])
                multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
                internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
                security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
                backup = st.selectbox("Online backup", ["Yes", "No", "No internet service"])
                protection = st.selectbox("Device protection", ["Yes", "No", "No internet service"])
            with c3:
                tech = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
                tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless = st.selectbox("Paperless billing", ["Yes", "No"])
                payment = st.selectbox("Payment method", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                monthly = st.number_input("Monthly charges", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
                total = st.number_input("Total charges", min_value=0.0, max_value=15000.0, value=840.0, step=10.0)
            submitted = st.form_submit_button("Predict churn", type="primary")
        if submitted:
            customer = {
                "gender": gender, "SeniorCitizen": senior, "Partner": partner,
                "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
                "MultipleLines": multiple, "InternetService": internet, "OnlineSecurity": security,
                "OnlineBackup": backup, "DeviceProtection": protection, "TechSupport": tech,
                "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
                "PaperlessBilling": paperless, "PaymentMethod": payment,
                "MonthlyCharges": monthly, "TotalCharges": total,
            }
            try:
                result = predict_customer(customer, artifact=artifact)
                risk = result["probability"] * 100
                st.metric("Estimated churn probability", f"{risk:.1f}%")
                if result["prediction"] == 1:
                    st.warning("Model prediction: likely to churn at the default 0.50 threshold.")
                else:
                    st.success("Model prediction: not flagged as likely to churn at the default 0.50 threshold.")
                st.caption("The probability is model-specific and should not be interpreted as certainty.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

elif page == "Model Insights":
    st.header("Model insights")
    importance_path = REPORTS / "feature_importance.csv"
    if importance_path.exists():
        imp = pd.read_csv(importance_path).head(15).sort_values("importance")
        st.subheader("Top global feature importances")
        st.bar_chart(imp.set_index("feature")["importance"])
        st.caption("Importance is model-specific and describes predictive behavior; it does not establish causation.")
    else:
        st.info("Feature importance is not available yet. Run training and check the saved reports.")
    if metrics:
        st.subheader("Classification report")
        report = metrics.get("classification_report", {})
        rows = []
        for label in ["No Churn", "Churn"]:
            key = "No Churn" if label == "No Churn" else "Churn"
            if key in report:
                rows.append({"class": label, **report[key]})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.sidebar.divider()
st.sidebar.caption("ChurnIQ · Educational portfolio project")
