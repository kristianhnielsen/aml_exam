import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    from preprocessor import preprocessor
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
    from xgboost import XGBClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.metrics import classification_report
    from experiment_tracking import ExperimentTracker
    from sklearn.linear_model import LogisticRegression
    return (
        ExperimentTracker,
        GradientBoostingClassifier,
        LogisticRegression,
        RandomForestClassifier,
        RandomizedSearchCV,
        VotingClassifier,
        XGBClassifier,
        accuracy_score,
        classification_report,
        preprocessor,
        roc_auc_score,
    )


@app.cell
def _(preprocessor):
    X_train, X_test, y_train, y_test = preprocessor()
    X_train.shape, y_train.shape, X_test.shape, y_test.shape
    return X_test, X_train, y_test, y_train


@app.cell
def _(ExperimentTracker):
    mlflow = ExperimentTracker(experiment_name="AML Task 2")

    return (mlflow,)


@app.cell
def _(
    LogisticRegression,
    RandomizedSearchCV,
    X_test,
    X_train,
    accuracy_score,
    classification_report,
    mlflow,
    roc_auc_score,
    y_test,
    y_train,
):
    lr = LogisticRegression()

    param_grid_lr = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga', 'lbfgs']
    }

    with mlflow.start_run(run_name="Logistic Regression"):
        lr_random = RandomizedSearchCV(estimator=lr, param_distributions=param_grid_lr, n_iter=10, cv=5, verbose=2, n_jobs=-1, random_state=42)
        lr_random.fit(X_train, y_train)
        best_lr = lr_random.best_estimator_
        y_pred_lr = best_lr.predict(X_test)
    
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        roc_auc_lr = roc_auc_score(y_test, best_lr.predict_proba(X_test)[:, 1])
        report_lr = classification_report(y_test, y_pred_lr)
    
        mlflow.log_params(lr_random.best_params_)
        mlflow.log_metrics({
            "test_accuracy": accuracy_lr,
            "test_roc_auc": roc_auc_lr
        })
    
        print(f"Test Accuracy: {accuracy_lr}")
        print(f"Test ROC AUC: {roc_auc_lr}")
        print(report_lr)
    return (best_lr,)


@app.cell
def _(
    RandomForestClassifier,
    RandomizedSearchCV,
    X_test,
    X_train,
    accuracy_score,
    classification_report,
    mlflow,
    roc_auc_score,
    y_test,
    y_train,
):
    rfc = RandomForestClassifier()

    param_grid_rfc = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    with mlflow.start_run(run_name="Random Forest Classifier"):
        random_search = RandomizedSearchCV(estimator=rfc, param_distributions=param_grid_rfc,
                                        n_iter=50, cv=3, verbose=2, random_state=42, n_jobs=-1)
        random_search.fit(X_train, y_train)
        best_rfc = random_search.best_estimator_
        best_rfc
        y_pred_rfc = best_rfc.predict(X_test)
        y_proba_rfc = best_rfc.predict_proba(X_test)[:, 1]
        roc_auc_rfc = roc_auc_score(y_test, y_proba_rfc)
        accuracy_rfc = accuracy_score(y_test, y_pred_rfc)

        mlflow.log_metrics({
            "test_accuracy": accuracy_rfc,
            "test_roc_auc": roc_auc_rfc
        })

        print(f"ROC AUC: {roc_auc_rfc}")
        print(f"Accuracy: {accuracy_rfc}")
        print(classification_report(y_test, y_pred_rfc))
    return (best_rfc,)


@app.cell
def _(
    GradientBoostingClassifier,
    RandomizedSearchCV,
    X_test,
    X_train,
    accuracy_score,
    classification_report,
    mlflow,
    roc_auc_score,
    y_test,
    y_train,
):
    gbc = GradientBoostingClassifier()

    param_grid_gbc = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.6, 0.8, 1.0],
        'min_samples_split': [2, 5, 10]
    }

    with mlflow.start_run(run_name="Gradient Boosting Classifier"):
        random_search_gbc = RandomizedSearchCV(estimator=gbc, param_distributions=param_grid_gbc,
                                        n_iter=50, cv=3, verbose=2, random_state=42, n_jobs=-1)
        random_search_gbc.fit(X_train, y_train)
        best_gbc = random_search_gbc.best_estimator_
        best_gbc
        y_pred_gbc = best_gbc.predict(X_test)
        y_proba_gbc = best_gbc.predict_proba(X_test)[:, 1]
        roc_auc_gbc = roc_auc_score(y_test, y_proba_gbc)
        accuracy_gbc = accuracy_score(y_test, y_pred_gbc)

        mlflow.log_metrics({
            "test_accuracy": accuracy_gbc,
            "test_roc_auc": roc_auc_gbc
        })

        print(f"ROC AUC: {roc_auc_gbc}")
        print(f"Accuracy: {accuracy_gbc}")
        print(classification_report(y_test, y_pred_gbc))
    return (best_gbc,)


@app.cell
def _(
    RandomizedSearchCV,
    XGBClassifier,
    X_test,
    X_train,
    accuracy_score,
    classification_report,
    mlflow,
    roc_auc_score,
    y_test,
    y_train,
):
    xgb = XGBClassifier(eval_metric='logloss')
    param_grid_xgb = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    with mlflow.start_run(run_name="XGBoost Classifier"):
        random_search_xgb = RandomizedSearchCV(estimator=xgb, param_distributions=param_grid_xgb,
                                        n_iter=50, cv=3, verbose=2, random_state=42, n_jobs=-1)
        random_search_xgb.fit(X_train, y_train)
        best_xgb = random_search_xgb.best_estimator_
        best_xgb
        y_pred_xgb = best_xgb.predict(X_test)
        y_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]
        roc_auc_xgb = roc_auc_score(y_test, y_proba_xgb)
        accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

        mlflow.log_metrics({
            "test_accuracy": accuracy_xgb,
            "test_roc_auc": roc_auc_xgb
        })

        print(f"ROC AUC: {roc_auc_xgb}")
        print(f"Accuracy: {accuracy_xgb}")
        print(classification_report(y_test, y_pred_xgb))
    return (best_xgb,)


@app.cell
def _(
    VotingClassifier,
    X_test,
    X_train,
    accuracy_score,
    best_gbc,
    best_lr,
    best_rfc,
    best_xgb,
    classification_report,
    mlflow,
    roc_auc_score,
    y_test,
    y_train,
):

    voting_clf = VotingClassifier(estimators=[
        ('rfc', best_rfc),
        ('gbc', best_gbc),
        ('xgb', best_xgb),
        ('lr', best_lr)
    ], voting='soft')


    with mlflow.start_run(run_name="Voting Classifier"):

        voting_clf.fit(X_train, y_train)
        y_pred_voting = voting_clf.predict(X_test)
    
        accuracy_voting = accuracy_score(y_test, y_pred_voting)
        roc_auc_voting = roc_auc_score(y_test, voting_clf.predict_proba(X_test)[:, 1])
        report_voting = classification_report(y_test, y_pred_voting)
    
        mlflow.log_metrics({
            "test_accuracy": accuracy_voting,
            "test_roc_auc": roc_auc_voting
        })
    
        print(f"Test Accuracy: {accuracy_voting}")
        print(f"Test ROC AUC: {roc_auc_voting}")
        print(report_voting)
    return


@app.cell
def _(
    X_test,
    X_train,
    accuracy_score,
    best_gbc,
    best_lr,
    best_rfc,
    best_xgb,
    classification_report,
    mlflow,
    roc_auc_score,
    y_test,
    y_train,
):
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression


    stacking_clf = StackingClassifier(
        estimators=[
            ('rfc', best_rfc),
            ('gbc', best_gbc),
            ('xgb', best_xgb),
            ('lr', best_lr)
        ],
        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1
    )

    with mlflow.start_run(run_name="Stacking Classifier"):
        stacking_clf.fit(X_train, y_train)
        y_pred_stacking = stacking_clf.predict(X_test)
    
        accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
        roc_auc_stacking = roc_auc_score(y_test, stacking_clf.predict_proba(X_test)[:, 1])
        report_stacking = classification_report(y_test, y_pred_stacking)
    
        mlflow.log_metrics({
            "test_accuracy": accuracy_stacking,
            "test_roc_auc": roc_auc_stacking
        })
    
        print(f"Test Accuracy: {accuracy_stacking}")
        print(f"Test ROC AUC: {roc_auc_stacking}")
        print(report_stacking)
    return (LogisticRegression,)


if __name__ == "__main__":
    app.run()
