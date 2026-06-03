import os
from typing import Dict, Tuple

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = os.path.join("WA_Fn-UseC_-Telco-Customer-Churn.csv")
PIPELINE_PATH = os.path.join("models", "churn_pipeline.pkl")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df


@st.cache_data
def load_pipeline(path: str):
    return joblib.load(path)


def build_pipeline(model_name: str, X: pd.DataFrame) -> Pipeline:
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

    model_map = {
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

    classifier = model_map.get(model_name, model_map["Random Forest"])
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    return pipeline


def train_pipeline(
    model_name: str, X: pd.DataFrame, y: pd.Series
) -> Tuple[Pipeline, Dict[str, float], pd.Series, pd.Series, pd.Series]:
    pipeline = build_pipeline(model_name, X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_prob),
    }
    return pipeline, metrics, y_test, y_pred, y_prob


def get_model_comparison(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    model_names = ["Random Forest", "Logistic Regression", "Gradient Boosting"]
    records = []
    for model_name in model_names:
        _, metrics, y_test, y_pred, y_prob = train_pipeline(model_name, X, y)
        records.append(
            {
                "Model": model_name,
                "Accuracy": metrics["Accuracy"],
                "ROC AUC": metrics["ROC AUC"],
            }
        )
    return pd.DataFrame(records)


def build_input_dataframe(pipeline: Pipeline, user_input: Dict[str, object]) -> pd.DataFrame:
    feature_columns = list(pipeline.feature_names_in_)
    base_data = {feature: 0 for feature in feature_columns}
    base_data.update(user_input)
    return pd.DataFrame([base_data], columns=feature_columns)


def render_home(df: pd.DataFrame) -> None:
    st.title("Telco Customer Churn Prediction Suite")
    st.markdown(
        """
        ### Business context

        Telco providers need a reliable, end-to-end churn prediction system to protect revenue, retain customers, and allocate marketing spend more effectively.
        """
    )
    st.markdown(
        "This dashboard includes dataset exploration, interactive EDA, model training, model comparison, and a production-ready customer churn prediction form."
    )

    churn_rate = round((df["Churn"].value_counts().get("Yes", 0) / df.shape[0]) * 100, 2)
    avg_tenure = round(df["tenure"].median(), 1)
    avg_monthly = round(df["MonthlyCharges"].mean(), 2)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customer records", df.shape[0])
    c2.metric("Predictor features", df.shape[1] - 1)
    c3.metric("Churn rate", f"{churn_rate}%")
    c4.metric("Median tenure", f"{avg_tenure} months")

    st.markdown("### Asset index")
    st.markdown(
        """
        1. Home overview
        2. Dataset Explorer
        3. EDA Dashboard
        4. Model Training Section
        5. Model Comparison Dashboard
        6. Prediction System Form
        """
    )

    churn_counts = df["Churn"].value_counts().rename({"No": "Retained", "Yes": "Churn"})
    fig_churn = px.pie(
        names=churn_counts.index,
        values=churn_counts.values,
        title="Churn / Retained Split",
        hole=0.35,
        color_discrete_sequence=px.colors.qualitative.T10,
    )
    st.plotly_chart(fig_churn, use_container_width=True)


def render_dataset(df: pd.DataFrame) -> None:
    st.title("Dataset Explorer")
    st.write(
        "Explore the cleaned Telco churn dataset with search, profile summaries, and null reports."
    )

    search_text = st.text_input("Search text in categorical columns")
    selectable_columns = st.multiselect(
        "Select columns to display",
        options=list(df.columns),
        default=df.columns.tolist(),
    )

    filtered_df = df.copy()
    if search_text:
        mask = pd.Series(False, index=df.index)
        for column in df.select_dtypes(include=["object"]).columns:
            mask |= df[column].astype(str).str.contains(search_text, case=False, na=False)
        filtered_df = filtered_df[mask]

    st.write("### Data preview")
    st.dataframe(filtered_df[selectable_columns].head(200), use_container_width=True)

    null_summary = df.isna().sum().sort_values(ascending=False)
    st.write("### Missing values")
    st.bar_chart(null_summary[null_summary > 0])

    st.write("### Summary statistics")
    st.dataframe(df.describe(include="all"), use_container_width=True)

    with st.expander("Column profiles"):
        for column in df.columns:
            st.write(f"**{column}**")
            if df[column].dtype == "object":
                st.write(df[column].value_counts(dropna=False).head(10))
            else:
                st.write(df[column].describe())


def render_eda(df: pd.DataFrame) -> None:
    st.title("EDA Dashboard")
    st.write("Interactive visual analysis of churn behavior and feature relationships.")

    numeric_options = ["tenure", "MonthlyCharges", "TotalCharges"]
    category_options = [
        "gender",
        "Partner",
        "Dependents",
        "InternetService",
        "Contract",
        "PaymentMethod",
    ]

    selected_numeric = st.selectbox("Choose numeric feature", numeric_options, index=0)
    selected_category = st.selectbox("Choose category segment", category_options, index=0)

    row1, row2 = st.columns(2)
    churn_counts = df["Churn"].value_counts().rename({"No": "Retained", "Yes": "Churn"})
    fig_churn = px.pie(
        names=churn_counts.index,
        values=churn_counts.values,
        title="Churn Distribution",
        hole=0.35,
    )
    row1.plotly_chart(fig_churn, use_container_width=True)

    fig_box = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Monthly Charges by Churn",
        color_discrete_map={"No": "#2ca02c", "Yes": "#d62728"},
    )
    row2.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    row3, row4 = st.columns(2)
    fig_tenure = px.histogram(
        df,
        x="tenure",
        color="Churn",
        nbins=30,
        title="Tenure Distribution by Churn",
        barmode="overlay",
        opacity=0.7,
    )
    row3.plotly_chart(fig_tenure, use_container_width=True)

    fig_segment = px.box(
        df,
        x=selected_category,
        y=selected_numeric,
        color="Churn",
        title=f"{selected_numeric} by {selected_category} and Churn",
        points="all",
    )
    row4.plotly_chart(fig_segment, use_container_width=True)

    st.markdown("---")
    st.write("### Correlation heatmap")
    correlation_matrix = df[numeric_options].corr()
    fig_corr = px.imshow(
        correlation_matrix,
        text_auto=True,
        title="Numeric Feature Correlation",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig_corr, use_container_width=True)


def render_training(df: pd.DataFrame) -> None:
    st.title("Model Training Section")
    st.write(
        "Train a candidate pipeline live in the browser and display immediate performance summaries."
    )

    if "trained_pipeline" not in st.session_state:
        st.session_state.trained_pipeline = None
        st.session_state.training_metrics = None
        st.session_state.training_results = None
        st.session_state.training_model = None

    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    model_name = st.selectbox(
        "Select model to train",
        ["Random Forest", "Logistic Regression", "Gradient Boosting"],
    )
    train_button = st.button("Train selected model")

    if train_button:
        pipeline, metrics, y_test, y_pred, y_prob = train_pipeline(model_name, X, y)
        st.session_state.trained_pipeline = pipeline
        st.session_state.training_metrics = metrics
        st.session_state.training_results = (y_test, y_pred, y_prob)
        st.session_state.training_model = model_name

    if st.session_state.trained_pipeline is not None:
        st.success(f"{st.session_state.training_model} trained successfully.")
        st.subheader("Performance summary")
        st.metric("Accuracy", f"{st.session_state.training_metrics['Accuracy']:.4f}")
        st.metric("ROC AUC", f"{st.session_state.training_metrics['ROC AUC']:.4f}")

        y_test, y_pred, _ = st.session_state.training_results
        st.write("#### Classification report")
        report = classification_report(y_test, y_pred, output_dict=True, digits=4)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)

        cm = confusion_matrix(y_test, y_pred)
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Predicted Retained", "Predicted Churn"],
                y=["Actual Retained", "Actual Churn"],
                colorscale="Blues",
                showscale=False,
            )
        )
        fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig_cm, use_container_width=True)

        save_pipeline = st.button("Save this model as production pipeline")
        if save_pipeline:
            os.makedirs("models", exist_ok=True)
            joblib.dump(st.session_state.trained_pipeline, PIPELINE_PATH)
            st.success(f"Saved production pipeline to {PIPELINE_PATH}")


def render_comparison(df: pd.DataFrame) -> None:
    st.title("Model Comparison Dashboard")
    st.write(
        "Compare key candidates and identify the highest-performing production model."
    )

    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    metrics_df = get_model_comparison(X, y)
    metrics_df["Accuracy"] = metrics_df["Accuracy"].round(4)
    metrics_df["ROC AUC"] = metrics_df["ROC AUC"].round(4)

    champion_row = metrics_df.loc[metrics_df["ROC AUC"].idxmax()]
    st.write(
        f"**Champion model:** {champion_row['Model']} with ROC AUC = {champion_row['ROC AUC']:.4f}"
    )

    fig_bar = px.bar(
        metrics_df,
        x="Model",
        y=["Accuracy", "ROC AUC"],
        barmode="group",
        title="Model Performance Comparison",
        labels={"value": "Score", "variable": "Metric"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.write("### Candidate leaderboard")
    st.dataframe(metrics_df.sort_values(by="ROC AUC", ascending=False), use_container_width=True)

    if os.path.exists(PIPELINE_PATH):
        prod_pipeline = load_pipeline(PIPELINE_PATH)
        st.write("### Current production artifact")
        st.write(f"Saved production pipeline: `{PIPELINE_PATH}`")
        st.write("Pipeline input features:")
        st.write(list(prod_pipeline.feature_names_in_))
    else:
        st.warning("No production pipeline artifact found yet. Train and save a model in the Model Training section.")


def render_prediction(df: pd.DataFrame, pipeline: Pipeline) -> None:
    st.title("Prediction System Form")
    st.write(
        "Submit customer profile inputs and receive a churn risk score using the serialized pipeline artifact."
    )

    if pipeline is None:
        st.error(
            "Production pipeline is unavailable. Run training and save a pipeline first, then reload the app."
        )
        return

    with st.form(key="prediction_form"):
        left, right = st.columns(2)

        gender = left.selectbox("Gender", ["Female", "Male"])
        senior = left.selectbox("Senior Citizen", ["No", "Yes"])
        partner = left.selectbox("Partner", ["Yes", "No"])
        dependents = left.selectbox("Dependents", ["Yes", "No"])
        tenure = left.slider("Tenure (months)", min_value=0, max_value=72, value=12)
        monthly_charges = left.number_input(
            "Monthly Charges", min_value=0.0, value=70.0, step=0.1
        )
        total_charges = left.number_input(
            "Total Charges", min_value=0.0, value=250.0, step=0.1
        )

        phone_service = right.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = right.selectbox(
            "Multiple Lines", ["No phone service", "No", "Yes"]
        )
        internet_service = right.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = right.selectbox(
            "Online Security", ["No internet service", "No", "Yes"])
        online_backup = right.selectbox("Online Backup", ["No internet service", "No", "Yes"])
        device_protection = right.selectbox(
            "Device Protection", ["No internet service", "No", "Yes"])
        tech_support = right.selectbox("Tech Support", ["No internet service", "No", "Yes"])
        streaming_tv = right.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
        streaming_movies = right.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
        contract = right.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = right.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = right.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

        submit_button = st.form_submit_button(label="Run prediction")

    if submit_button:
        user_input = {
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        input_df = build_input_dataframe(pipeline, user_input)
        prediction = pipeline.predict(input_df)[0]
        churn_probability = pipeline.predict_proba(input_df)[0, 1]

        st.markdown("## Prediction Result")
        if prediction == 1:
            st.error("⚠️ This customer is likely to churn.")
        else:
            st.success("✅ This customer is likely to stay.")

        st.write(f"**Churn probability:** {churn_probability:.2%}")
        st.write("### Input review")
        st.dataframe(input_df, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Telco Churn Predictor",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    df = load_data(DATA_PATH)
    pipeline = None
    if os.path.exists(PIPELINE_PATH):
        pipeline = load_pipeline(PIPELINE_PATH)

    page = st.sidebar.radio(
        "Navigation",
        [
            "Home",
            "Dataset Explorer",
            "EDA Dashboard",
            "Model Training",
            "Model Comparison",
            "Prediction System",
        ],
        index=0,
    )

    if page == "Home":
        render_home(df)
    elif page == "Dataset Explorer":
        render_dataset(df)
    elif page == "EDA Dashboard":
        render_eda(df)
    elif page == "Model Training":
        render_training(df)
    elif page == "Model Comparison":
        render_comparison(df)
    elif page == "Prediction System":
        render_prediction(df, pipeline)


if __name__ == "__main__":
    main()
