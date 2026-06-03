# Customer Churn Prediction Project

A data science and machine learning project for predicting customer churn in a telecom business using the Telco Customer Churn dataset.

## Project Overview

This repository includes a full churn prediction workflow:
- Data ingestion and cleaning
- Exploratory data analysis (EDA)
- Model training and evaluation
- A Streamlit dashboard for dataset exploration and interactive churn prediction

The target outcome is a production-ready pipeline that predicts whether a customer is likely to churn, enabling business stakeholders to take retention action early.

## Key Components

- `app.py`
  - Streamlit application with:
    - Home overview and business context
    - Dataset explorer
    - EDA dashboard
    - Model training and comparison
    - Interactive churn prediction form
- `train.py`
  - Training script for building a Random Forest churn model
  - Saves a serialized pipeline to `models/churn_pipeline.pkl`
- `data/`
  - `WA_Fn-UseC_-Telco-Customer-Churn.csv`: original Telco churn dataset
  - `cleaned_data.csv`: cleaned dataset ready for modeling
- `models/`
  - `churn_pipeline.pkl`: saved trained scikit-learn pipeline
- `notebooks/`
  - `eda.ipynb`: exploratory data analysis notebook
  - `model_testing.ipynb`: model evaluation and testing notebook
- `requirments.txt`
  - Project dependencies for Python environment setup

## Dataset

The dataset includes customer account details, service usage, billing, and churn status. It is commonly used for classification tasks to predict whether a customer will churn (`Yes`) or stay (`No`).

## Installation

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirments.txt
   ```

## Usage

### Train the model

Run the training pipeline to build and save the churn prediction model:

```powershell
python train.py
```

This produces `models/churn_pipeline.pkl` and prints model performance metrics.

### Launch the Streamlit app

Start the interactive dashboard:

```powershell
streamlit run app.py
```

The app provides dataset visualization, model comparison, and a prediction interface for new customer records.

## What’s Included

- Cleaned dataset handling and missing value processing
- One-hot encoding for categorical features
- Standard scaling for numeric features
- Random Forest classification with balanced class weights
- Train/test split with stratified sampling
- Accuracy and ROC AUC evaluation metrics
- Streamlit dashboard for exploratory analytics and predictions

## Project Structure

```text
.
+-- app.py
+-- train.py
+-- requirments.txt
+-- data/
¦   +-- WA_Fn-UseC_-Telco-Customer-Churn.csv
¦   +-- cleaned_data.csv
+-- models/
¦   +-- churn_pipeline.pkl
+-- notebooks/
    +-- eda.ipynb
    +-- model_testing.ipynb
```

## Notes

- `app.py` uses Streamlit caching to speed up data and model loading.
- The current training script uses a Random Forest by default, but `app.py` supports multiple classifiers in the dashboard code.
- Update the dataset path if you move the CSV file.

## Contact

For questions or enhancements, review the notebooks and inspect `app.py` / `train.py` for implementation details.
