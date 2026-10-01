# Predictive Customer Churn Model for ConnectTel

## Project Overview

This project builds an end-to-end machine learning system to predict customer churn for a telecom company, ConnectTel.

The workflow covers:
- Data cleaning
- Exploratory Data Analysis
- Feature engineering
- Preprocessing
- Logistic Regression, Random Forest and XGBoost
- Stratified 5-fold cross-validation
- GridSearchCV hyperparameter tuning
- Model evaluation
- SHAP model interpretation
- Business retention recommendations

## Business Problem

ConnectTel wants to identify customers who are more likely to churn and understand the factors associated with churn so that the retention team can take proactive action.

## Dataset

The project uses the Telco Customer Churn dataset.

Target:
- `No` → customer stayed
- `Yes` → customer churned

## Feature Engineering

Three features were created:
- `TotalChargesPerTenure`
- `ServiceCount`
- `TenureGroup`

`TenureGroup` is used for analysis, while the executed model results use the numeric engineered features.

## Models

Three classification models were trained and tuned:
1. Logistic Regression
2. Random Forest
3. XGBoost

ROC-AUC was used as the main optimization metric because the target is imbalanced. Accuracy, precision, recall and F1-score were also reported.

## Final Test Results

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.7999 | 0.6597 | 0.5080 | 0.5740 | 0.8472 |
| Random Forest | 0.7551 | 0.5266 | 0.7674 | 0.6246 | 0.8414 |
| Logistic Regression | 0.8041 | 0.6531 | 0.5588 | 0.6023 | 0.8413 |

XGBoost has the highest held-out test ROC-AUC in this experiment and is used for SHAP interpretation.

## Key EDA Findings

- Month-to-month customers: 42.71% observed churn
- One-year customers: 11.27% observed churn
- Two-year customers: 2.83% observed churn
- Fiber optic customers: 41.89% observed churn
- DSL customers: 18.96% observed churn
- Mean tenure of churned customers: about 18 months
- Mean tenure of retained customers: about 38 months

These are observed relationships in the dataset and should not be interpreted as causal effects.

## Business Recommendations

- Target high-risk month-to-month customers with suitable contract incentives.
- Strengthen engagement during the early customer lifecycle.
- Review high-charge, high-risk customers for plan/value issues.
- Investigate the higher churn observed among fiber optic customers.
- Test relevant service bundles and support offers.

The model should be used as a decision-support tool and combined with customer value, service history and business context.

## Project Structure

```text
predictive-customer-churn/
├── data/
├── notebooks/
│   └── 01_predictive_customer_churn_FINAL.ipynb
├── src/
├── outputs/
│   ├── figures/
│   ├── models/
│   └── reports/
├── requirements.txt
├── README.md
└── LICENSE
```

## Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, SHAP and Jupyter Notebook.
