import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df


def build_pipeline(X: pd.DataFrame) -> Pipeline:
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

    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    return pipeline


def main() -> None:
    data_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = load_data(data_path)

    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    pipeline = build_pipeline(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("Training features:", list(X.columns))
    print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 4))
    print("Classification report:\n", classification_report(y_test, y_pred, digits=4))

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, os.path.join("models", "churn_pipeline.pkl"))
    print("\n✅ Saved trained pipeline to models/churn_pipeline.pkl")


if __name__ == "__main__":
    main()