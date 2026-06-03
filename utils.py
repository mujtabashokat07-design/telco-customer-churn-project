import os
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw Telco churn dataset from CSV."""
    df = pd.read_csv(path)
    return df


def clean_churn_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the Telco churn dataset and return a ready-to-use DataFrame."""
    cleaned = df.copy()
    if "customerID" in cleaned.columns:
        cleaned = cleaned.drop(columns=["customerID"])

    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")

    cleaned = cleaned.dropna().reset_index(drop=True)
    return cleaned


def get_feature_target_sets(
    df: pd.DataFrame, target_column: str = "Churn"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the dataset into features and a binary target vector."""
    X = df.drop(columns=[target_column])
    y = df[target_column].map({"No": 0, "Yes": 1})
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing stage for numeric and categorical features."""
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
            ("numeric", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def get_classifier(name: str = "Random Forest"):
    """Return a classifier instance by name."""
    mapping = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    return mapping.get(name, mapping["Random Forest"])


def build_pipeline(model_name: str, X: pd.DataFrame) -> Pipeline:
    """Construct a full sklearn pipeline using preprocessing and a classifier."""
    preprocessor = build_preprocessor(X)
    classifier = get_classifier(model_name)
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    return pipeline


def train_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, float], pd.Series, pd.Series, pd.Series]:
    """Train the pipeline and return evaluation metrics and test predictions."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    return pipeline, metrics, y_test, y_pred, y_prob


def evaluate_classification(y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> str:
    """Return a classification report for model predictions."""
    report = classification_report(y_true, y_pred, digits=4)
    return report


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    """Serialize a trained pipeline to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: str) -> Pipeline:
    """Load a serialized pipeline from disk."""
    return joblib.load(path)


def build_input_dataframe(pipeline: Pipeline, user_input: Dict[str, object]) -> pd.DataFrame:
    """Build a DataFrame from user input for a trained pipeline."""
    feature_columns = list(pipeline.feature_names_in_)
    base_data = {feature: 0 for feature in feature_columns}
    base_data.update(user_input)
    return pd.DataFrame([base_data], columns=feature_columns)
