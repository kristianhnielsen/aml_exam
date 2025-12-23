# Project 1: Customer Churn Prediction in a Subscription-Based Service

**Task:**

Tabular Data (Classification)

**Dataset:**

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle) WA*Fn-UseC*-Telco-Customer-Churn.csv

**Short Description:**
Telecommunication and subscription-based companies rely heavily on churn prediction
models to retain customers. YOU will build an end-to-end ML pipeline to classify customers
as “likely to churn” or “likely to stay,” using tabular data that combines demographic,
behavioral, and usage features.

## Task 1: EDA

- Analyze dataset structure: distributions, outliers, skewness
- Examine correlations among numerical features
- Explore categorical feature cardinality
- Study churn vs. non-churn population imbalance
- Identify missing values and propose imputation strategies

## Task 2: Shallow Learning Approaches

**Train and evaluate at least three classical models:**

- Random Forest, XGBoost / Gradient Boosting
  (others: Logistic Regression, LightGBM allowed as extras)
- Use Ensemble Learner

**Expected steps:**

- Tune hyperparameters through cross-validation
- Use stratification for splits
- Evaluate using accuracy, confusion matrix, F1

## Task 3: Neural Network Models

**Feed Forward Neural Networks**

- Apply regularization: dropout, batch normalization, L1, L2, Early Stopping, ..etc
- Experiment with different optimizers, learning rate, activation functions,...etc
- Analyze training curves (loss/accuracy)

**Autoencoder**

- Train an Autoencoder for feature compression
- Feed the latent embedding into shallow ML / FFNN (use the AE for feature extraction)
- Compare performance vs raw features

## Task 4: Model Comparison & Interpretability

- Compare performance across shallow models, FFNN, and AE
- Provide a detailed error analysis (misclassified users)

**Use SHAP values to explain**

- Important features
- Differences between models

**Conclude with**

- What model is best for deployment?
- Which features contribute most to churn?
- How could performance be improved?

# MLFlow

Start the MLFlow UI server:

```bash
uvx mlflow ui
```

# Marimo

Run the Marimo notebooks:

```bash
uv run marimo edit
```

Or open a specific notebook:

```bash
uv run marimo edit file.py
```
