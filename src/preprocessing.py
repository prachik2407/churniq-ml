"""Reusable preprocessing and estimator construction."""
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def make_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, make_column_selector(dtype_include="number")),
        ("categorical", categorical, make_column_selector(dtype_exclude="number")),
    ])


def make_models(random_state: int = 42):
    """Return named, unfitted end-to-end pipelines."""
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("classifier", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=random_state
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=250, min_samples_leaf=2,
                class_weight="balanced", random_state=random_state, n_jobs=-1
            )),
        ]),
    }
