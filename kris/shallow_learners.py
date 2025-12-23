import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score, accuracy_score
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LogisticRegression,SGDClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    import marimo as mo

    data = pd.read_csv("./data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return (
        LogisticRegression,
        OneHotEncoder,
        accuracy_score,
        data,
        np,
        pd,
        plt,
        sns,
        train_test_split,
    )


@app.cell
def _(data):
    data.columns = data.columns.str.lower()
    data.drop(columns=["customerid", "gender"], inplace=True)
    data.head()
    return


@app.cell
def _(data):
    data.shape
    return


@app.cell
def _(data):
    data.drop(labels=data[data["totalcharges"] == " "].index, inplace=True)
    data["totalcharges"] = data["totalcharges"].astype(float)
    return


@app.cell
def _(data):
    data.shape
    return


@app.cell
def _():
    service_cols = [
        "onlinesecurity",
        "onlinebackup",
        "deviceprotection",
        "techsupport",
        "streamingtv",
        "streamingmovies",
    ]
    return (service_cols,)


@app.cell
def _(data):
    data.select_dtypes(include=["object"]).columns.to_list()
    return


@app.cell
def _(OneHotEncoder, data):
    for col in data.select_dtypes(include=["object"]).columns.to_list():
        le = OneHotEncoder()
        data[col] = le.fit_transform(data[col])
    data.head()
    return


@app.cell
def _(
    LogisticRegression,
    accuracy_score,
    data,
    np,
    pd,
    plt,
    service_cols,
    sns,
    train_test_split,
    y,
):
    _acc_list = []
    for _ in range(20):
        X_no_service = data.copy().drop(columns=["churn"])
        X_no_service.drop(columns=[*service_cols, 'totalcharges'], inplace=True)
        y_no_service = data["churn"]
        X_train_no_service, X_test_no_service, y_train_no_service, y_test_no_service = train_test_split(
            X_no_service, y_no_service, test_size=0.2, stratify=y
        )
    
        _model = LogisticRegression(max_iter=100000)
        _model.fit(X_train_no_service, y_train_no_service)
        _churn_pred = _model.predict(X_test_no_service)
        _accuracy = accuracy_score(y_test_no_service, _churn_pred)
        _acc_list.append(_accuracy)
    print(f"Mean accuracy without service features: {np.mean(_acc_list):.4f}")
    print(f"Highest accuracy without service features: {np.max(_acc_list):.4f}")

    # feature importance
    _importance = _model.coef_[0]
    _feature_names = X_no_service.columns
    _feature_importance = pd.Series(_importance, index=_feature_names).sort_values(
        ascending=False
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(x=_feature_importance.values, y=_feature_importance.index)
    plt.title("Feature Importance from Logistic Regression")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

    return


@app.cell
def _(
    LogisticRegression,
    accuracy_score,
    data,
    np,
    pd,
    plt,
    sns,
    train_test_split,
):
    X = data.copy().drop(columns=["churn"])
    y = data["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    _acc_list = []
    for _ in range(20):
        model = LogisticRegression(max_iter=100000)
        model.fit(X_train, y_train)
        _churn_pred = model.predict(X_test)
        _accuracy = accuracy_score(y_test, _churn_pred)
        _acc_list.append(_accuracy)

    print(f"Mean accuracy with all features: {np.mean(_acc_list):.4f}")
    print(f"Highest accuracy with all features: {np.max(_acc_list):.4f}")

    # feature importance
    importance = model.coef_[0]
    feature_names = X.columns
    feature_importance = pd.Series(importance, index=feature_names).sort_values(
        ascending=False
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance.values, y=feature_importance.index)
    plt.title("Feature Importance from Logistic Regression")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()
    return (y,)


if __name__ == "__main__":
    app.run()
