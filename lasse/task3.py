import marimo

__generated_with = "0.18.4"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    from preprocessor import preprocessor
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ReduceLROnPlateau,
    )
    from tensorflow.keras.optimizers import AdamW, SGD
    from experiment_tracking import ExperimentTracker
    import tensorflow as tf
    import xgboost as xgb
    from sklearn.metrics import (
        roc_auc_score,
        accuracy_score,
        classification_report,
    )
    import marimo as mo
    import pandas as pd
    import altair as alt
    import numpy as np
    return (
        AdamW,
        BatchNormalization,
        Dense,
        Dropout,
        EarlyStopping,
        ExperimentTracker,
        ReduceLROnPlateau,
        Sequential,
        accuracy_score,
        alt,
        classification_report,
        mo,
        np,
        pd,
        preprocessor,
        roc_auc_score,
        tf,
        xgb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Starting training
    """)
    return


@app.cell
def _(preprocessor):
    # Load and preprocess data
    X_train, X_test, y_train, y_test = preprocessor()
    input_dim = X_train.shape[1]
    return X_test, X_train, input_dim, y_test, y_train


@app.cell
def _(BatchNormalization, Dense, Dropout, Sequential):
    # Define Feed Forward Neural Network model
    def create_ffnn(input_dim, dropout_rate=0.2):
        model = Sequential()
        model.add(
            Dense(
                512,
                activation="relu",
                input_dim=input_dim,
            )
        )
        model.add(
            Dense(
                512,
                activation="relu",
            )
        )
        model.add(
            Dense(
                512,
                activation="relu",
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(
            Dense(
                256,
                activation="relu",
            )
        )
        model.add(
            Dense(
                256,
                activation="relu",
            )
        )
        model.add(
            Dense(
                256,
                activation="relu",
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(
            Dense(
                128,
                activation="relu",
            )
        )
        model.add(
            Dense(
                128,
                activation="relu",
            )
        )
        model.add(
            Dense(
                128,
                activation="relu",
            )
        )
        model.add(BatchNormalization())
        model.add(Dense(1, activation="sigmoid"))
        return model
    return (create_ffnn,)


@app.cell
def _(ExperimentTracker):
    mlflow = ExperimentTracker(experiment_name="AML Task 3")
    return (mlflow,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Creating FFNN
    """)
    return


@app.cell
def _(
    AdamW,
    EarlyStopping,
    ReduceLROnPlateau,
    X_test,
    X_train,
    classification_report,
    create_ffnn,
    input_dim,
    mlflow,
    roc_auc_score,
    y_test,
    y_train,
):
    # Compile and train the model
    ffnn_model = create_ffnn(input_dim)
    ffnn_model.compile(
        optimizer=AdamW(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=15,
        restore_best_weights=True,
        min_delta=0.0001,
    )

    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=10)

    with mlflow.start_run(
        run_name="Feed Forward Neural Network - deep with - SGD"
    ):
        history = ffnn_model.fit(
            X_train,
            y_train,
            validation_split=0.3,
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, lr_scheduler],
        )

        # Evaluate the model
        test_loss, test_accuracy = ffnn_model.evaluate(X_test, y_test)
        y_pred_ffnn = ffnn_model.predict(X_test)
        _roc_auc_score = roc_auc_score(y_test, y_pred_ffnn)
        class_report = classification_report(
            y_test, (y_pred_ffnn >= 0.5).astype(int)
        )
        mlflow.log_metrics(
            {
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "roc_auc": _roc_auc_score,
            }
        )
        print("Classification Report:\n", class_report)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Creating AE
    """)
    return


@app.cell
def _(Dense, tf):
    # Create Autoencoder model
    def create_autoencoder(input_dim, encoding_dim=64):
        input_layer = tf.keras.Input(shape=(input_dim,))
        encoded = Dense(512, activation="relu")(input_layer)
        encoded = Dense(256, activation="relu")(encoded)
        encoded = Dense(128, activation="relu")(encoded)
        encoded = Dense(encoding_dim, activation="relu")(encoded)

        decoded = Dense(128, activation="relu")(encoded)
        decoded = Dense(256, activation="relu")(decoded)
        decoded = Dense(512, activation="relu")(decoded)
        decoded = Dense(input_dim, activation="sigmoid")(decoded)

        autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoded)
        encoder = tf.keras.Model(inputs=input_layer, outputs=encoded)
        return autoencoder, encoder
    return (create_autoencoder,)


@app.cell
def _(
    EarlyStopping,
    ReduceLROnPlateau,
    X_test,
    X_train,
    create_autoencoder,
    input_dim,
    mlflow,
):
    # Compile and train the Autoencoder
    autoencoder, encoder = create_autoencoder(input_dim)
    autoencoder.compile(optimizer="adamW", loss="mse")
    early_stopping_ae = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        min_delta=0.0001,
    )

    lr_scheduler_ae = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=10
    )

    with mlflow.start_run(run_name="Autoencoder - run 1"):
        history_ae = autoencoder.fit(
            X_train,
            X_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping_ae, lr_scheduler_ae],
        )

        # Evaluate the autoencoder
        val_loss = autoencoder.evaluate(X_test, X_test)
        mlflow.log_metrics({"val_loss": val_loss})
        print(f"Validation Loss: {val_loss}")
    return (encoder,)


@app.cell
def _(X_test, X_train, encoder):
    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test)
    return X_test_encoded, X_train_encoded


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training FFNN with AE
    """)
    return


@app.cell
def _(
    AdamW,
    EarlyStopping,
    ReduceLROnPlateau,
    X_test_encoded,
    X_train_encoded,
    classification_report,
    create_ffnn,
    mlflow,
    roc_auc_score,
    y_test,
    y_train,
):
    # Define and train a FFNN on the encoded features
    ffnn_encoded_model = create_ffnn(X_train_encoded.shape[1])
    ffnn_encoded_model.compile(
        optimizer=AdamW(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    early_stopping_enc = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        min_delta=0.0001,
    )

    lr_scheduler_enc = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5
    )

    with mlflow.start_run(run_name="FFNN on Autoencoder Features - run 1"):
        history_enc = ffnn_encoded_model.fit(
            X_train_encoded,
            y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping_enc, lr_scheduler_enc],
        )

        # Evaluate the model
        y_pred_ffnn_ae = ffnn_encoded_model.predict(X_test_encoded)
        _roc_auc_score_enc = roc_auc_score(y_test, y_pred_ffnn_ae)
        test_loss_enc, test_accuracy_enc = ffnn_encoded_model.evaluate(
            X_test_encoded, y_test
        )
        class_report_enc = classification_report(
            y_test, (y_pred_ffnn_ae >= 0.5).astype(int)
        )
        mlflow.log_metrics(
            {
                "test_loss_encoded": test_loss_enc,
                "test_accuracy_encoded": test_accuracy_enc,
                "roc_auc": _roc_auc_score_enc,
            }
        )
        print(
            f"Test Loss (Encoded): {test_loss_enc}, Test Accuracy (Encoded): {test_accuracy_enc}"
        )
        print("Classification Report (Encoded):\n", class_report_enc)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Trying XGBoost with Encoded
    """)
    return


@app.cell
def _(
    X_test_encoded,
    X_train_encoded,
    accuracy_score,
    classification_report,
    mlflow,
    roc_auc_score,
    xgb,
    y_test,
    y_train,
):
    # Use the encoder for feature extraction and train XGBoost
    with mlflow.start_run(run_name="XGBoost on Autoencoder Features"):
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        xgb_model.fit(X_train_encoded, y_train)

        # Evaluate the model
        y_pred_proba = xgb_model.predict_proba(X_test_encoded)[:, 1]
        test_auc_xgb = roc_auc_score(y_test, y_pred_proba)
        y_pred_xgb = (y_pred_proba >= 0.5).astype(int)
        test_accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
        class_report_xgb = classification_report(y_test, y_pred_xgb)
        mlflow.log_metrics(
            {"roc_auc": test_auc_xgb, "test_accuracy_xgb": test_accuracy_xgb}
        )
        print(f"Test AUC (XGBoost on Encoded): {test_auc_xgb}")
        print(f"Test Accuracy (XGBoost on Encoded): {test_accuracy_xgb}")
        print("Classification Report (XGBoost on Encoded):\n", class_report_xgb)
    return (y_pred_xgb,)


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    # Creating Visualizations
    """)
    return


@app.cell
def _(pd, y_pred_xgb, y_test):
    y_pred_xgb_series = pd.DataFrame(y_pred_xgb).set_index(y_test.index)
    return (y_pred_xgb_series,)


@app.cell
def _(X_test, pd, y_pred_xgb_series, y_test):
    _df = pd.merge(
        left=X_test,
        right=y_test,
        left_index=True,
        right_on=y_test.index,
    )

    XGB_df = pd.merge(
        left=_df,
        right=y_pred_xgb_series,
        left_index=True,
        right_index=True,
    )
    XGB_df.rename(columns={0: "XGB_Predictions"}, inplace=True)
    XGB_df.drop(columns=["key_0"], inplace=True)
    XGB_df
    return (XGB_df,)


@app.cell
def _(XGB_df):
    XGB_with_predictions = XGB_df.copy()
    XGB_with_predictions["Guessed_right"] = XGB_with_predictions.apply(
        lambda row: "Correct"
        if row["Churn"] == row["XGB_Predictions"]
        else "Incorrect",
        axis=1,
    )
    return (XGB_with_predictions,)


@app.cell
def _(XGB_with_predictions, mo):
    pie_chart_dropdown = mo.ui.dropdown(
        options=XGB_with_predictions.columns.tolist(),
        label="Select Feature for Pie Chart",
    )
    return (pie_chart_dropdown,)


@app.cell(hide_code=True)
def _(XGB_with_predictions, alt, mo, pie_chart_dropdown):
    # replace _df with your data source
    _chart = (
        alt.Chart(XGB_with_predictions)
        .mark_arc()
        .encode(
            color=alt.Color(field="Guessed_right", type="nominal"),
            theta=alt.Theta(field=pie_chart_dropdown.value, type="quantitative"),
            tooltip=[
                alt.Tooltip(field=pie_chart_dropdown.value),
                alt.Tooltip(field=pie_chart_dropdown.value),
                alt.Tooltip(field="Guessed_right"),
            ],
        )
        .properties(
            height=290,
            width="container",
            config={"axis": {"grid": False}},
            title="Distribution of Predictions by Selected Feature",
        )
    )
    mo.vstack([pie_chart_dropdown, _chart])
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    # Trying other pretrained models
    """)
    return


@app.cell
def _(
    X_test,
    X_train,
    accuracy_score,
    mlflow,
    notify_completion,
    np,
    roc_auc_score,
    y_test,
    y_train,
):
    from tabpfn import TabPFNClassifier
    from sklearn.utils import resample

    MAX_SAMPLES = 1000

    if len(X_train) > MAX_SAMPLES:
        X_train_sub, y_train_sub = resample(
            X_train,
            y_train,
            n_samples=MAX_SAMPLES,
            random_state=42,
            stratify=y_train,
        )
    else:
        X_train_sub, y_train_sub = X_train, y_train


    with mlflow.start_run(run_name="tabPFN Classifier"):
        clf = TabPFNClassifier(device="cpu", n_estimators=2)
        clf.fit(X_train_sub, y_train_sub)

        prediction_probabilities = clf.predict_proba(X_test)
        _y_pred_tabpfn_proba = (
            prediction_probabilities[:, 1]
            if getattr(prediction_probabilities, "ndim", 1) > 1
            else prediction_probabilities
        )
        _y_pred_tabpfn = (_y_pred_tabpfn_proba >= 0.5).astype(int)
        _y_true = y_test.values if hasattr(y_test, "values") else np.array(y_test)

        _accuracy_tabpfn = accuracy_score(_y_true, _y_pred_tabpfn)
        _roc_auc_tabpfn = roc_auc_score(_y_true, _y_pred_tabpfn_proba)

        notify_completion(f"TabPFN Accuracy: {_accuracy_tabpfn:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Using BERT
    """)
    return


@app.cell
def _():
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import BertTokenizer, BertForSequenceClassification


    class TabularDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            _text = str(self.texts.iloc[idx])
            _label = self.labels.iloc[idx]

            _encoding = self.tokenizer.encode_plus(
                _text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            return {
                "input_ids": _encoding["input_ids"].flatten(),
                "attention_mask": _encoding["attention_mask"].flatten(),
                "labels": torch.tensor(_label, dtype=torch.long),
            }
    return (
        BertForSequenceClassification,
        BertTokenizer,
        DataLoader,
        TabularDataset,
        torch,
    )


@app.cell
def _(
    BertForSequenceClassification,
    BertTokenizer,
    DataLoader,
    TabularDataset,
    X_test,
    X_train,
    torch,
    y_test,
    y_train,
):
    device = torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    model.to(device)

    params = {
        "learning_rate": 2e-5,
        "epochs": 3,
        "batch_size": 16,
        "max_length": 128,
        "model_name": "bert-base-uncased",
    }

    train_dataset = TabularDataset(
        X_train, y_train, tokenizer, max_len=params["max_length"]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )

    test_dataset = TabularDataset(
        X_test, y_test, tokenizer, max_len=params["max_length"]
    )
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"])
    return device, model, optimizer, params, test_loader, train_loader


@app.cell
def _(
    device,
    mlflow,
    model,
    notify_completion,
    optimizer,
    params,
    test_loader,
    torch,
    train_loader,
):
    with mlflow.start_run(run_name="BERT Text Classification"):
        mlflow.log_params(params)

        for _epoch in range(params["epochs"]):
            model.train()
            _total_train_loss = 0
            for _batch in train_loader:
                _input_ids = _batch["input_ids"].to(device)
                _attention_mask = _batch["attention_mask"].to(device)
                _labels = _batch["labels"].to(device)

                optimizer.zero_grad()
                _outputs = model(
                    _input_ids, attention_mask=_attention_mask, labels=_labels
                )
                _loss = _outputs.loss
                _loss.backward()
                optimizer.step()
                _total_train_loss += _loss.item()

            _avg_train_loss = _total_train_loss / len(train_loader)

            model.eval()
            _correct_preds = 0
            _total_examples = 0
            with torch.no_grad():
                for _batch in test_loader:
                    _input_ids = _batch["input_ids"].to(device)
                    _attention_mask = _batch["attention_mask"].to(device)
                    _labels = _batch["labels"].to(device)

                    _outputs = model(_input_ids, attention_mask=_attention_mask)
                    _preds = torch.argmax(_outputs.logits, dim=1)
                    _correct_preds += (_preds == _labels).sum().item()
                    _total_examples += _labels.size(0)

            _val_accuracy = _correct_preds / _total_examples
            mlflow.log_metrics(
                {
                    f"train_loss_epoch_{_epoch + 1}": _avg_train_loss,
                    f"val_accuracy_epoch_{_epoch + 1}": _val_accuracy,
                }
            )

            print(
                f"Epoch {_epoch + 1} | Loss: {_avg_train_loss:.4f} | Val Acc: {_val_accuracy:.4f}"
            )
            notify_completion(f"Epoch {_epoch + 1} Done. Acc: {_val_accuracy:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Trying tabNet
    """)
    return


@app.cell
def _(
    X_test,
    X_train,
    accuracy_score,
    mlflow,
    notify_completion,
    y_test,
    y_train,
):
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier

    _X_train_np = X_train.values if hasattr(X_train, "values") else X_train
    _y_train_np = y_train.values if hasattr(y_train, "values") else y_train
    _X_test_np = X_test.values if hasattr(X_test, "values") else X_test
    _y_test_np = y_test.values if hasattr(y_test, "values") else y_test

    _tabnet_model = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="entmax",
    )

    with mlflow.start_run(run_name="TabNet_Baseline"):
        _tabnet_model.fit(
            X_train=_X_train_np,
            y_train=_y_train_np,
            eval_set=[(_X_train_np, _y_train_np), (_X_test_np, _y_test_np)],
            eval_name=["train", "valid"],
            eval_metric=["auc", "accuracy"],
            max_epochs=100,
            patience=15,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
        )

        _preds = _tabnet_model.predict(_X_test_np)
        _acc = accuracy_score(_y_test_np, _preds)
        mlflow.log_metrics({"Accuracy": _acc})
        notify_completion(f"TabNet Accuracy: {_acc:.4f}")
    return TabNetClassifier, torch


@app.cell
def _(
    TabNetClassifier,
    X_test,
    X_train,
    accuracy_score,
    classification_report,
    mlflow,
    notify_completion,
    roc_auc_score,
    torch,
    y_test,
    y_train,
):
    _X_train_np = X_train.values if hasattr(X_train, "values") else X_train
    _y_train_np = y_train.values if hasattr(y_train, "values") else y_train
    _X_test_np = X_test.values if hasattr(X_test, "values") else X_test
    _y_test_np = y_test.values if hasattr(y_test, "values") else y_test

    _params = {
        "n_d": 16,
        "n_a": 16,
        "n_steps": 5,
        "gamma": 1.5,  # Relaxation factor for feature reuse
        "n_independent": 2,
        "n_shared": 2,
        "lambda_sparse": 1e-4,  # Sparsity regulator
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": dict(lr=2e-2, weight_decay=1e-5),
        "mask_type": "entmax",
        "scheduler_params": {"step_size": 10, "gamma": 0.9},
        "scheduler_fn": torch.optim.lr_scheduler.StepLR,
        "verbose": 0,
    }

    _tabnet_tuned = TabNetClassifier(**_params)

    with mlflow.start_run(run_name="TabNet_Tuned_Balanced"):
        mlflow.log_params(_params)

        _tabnet_tuned.fit(
            X_train=_X_train_np,
            y_train=_y_train_np,
            eval_set=[(_X_test_np, _y_test_np)],
            eval_name=["valid"],
            eval_metric=["auc", "accuracy"],
            max_epochs=200,
            patience=20,
            batch_size=512,
            virtual_batch_size=64,
            weights=1,
            drop_last=False,
        )

        _probs = _tabnet_tuned.predict_proba(_X_test_np)[:, 1]
        _auc_score = roc_auc_score(_y_test_np, _probs)
        _accuracy = accuracy_score(_y_test_np, (_probs >= 0.5).astype(int))
        _class_report = classification_report(
            _y_test_np, (_probs >= 0.5).astype(int)
        )
        mlflow.log_metrics({"final_auc": _auc_score, "Accuracy": _accuracy})
        mlflow.log_metrics(
            {"best_val_auc": max(_tabnet_tuned.history["valid_auc"])}
        )
        notify_completion(
            f"TabNet Tuned AUC: {_auc_score:.4f} and Accuracy: {_accuracy:.4f}"
        )
        print(
            "Feature importances from tuned TabNet:",
            _tabnet_tuned.feature_importances_,
        )
        print("Classification Report:\n", _class_report)
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""
    # Teams Notifications
    """)
    return


@app.cell
def _():
    import requests
    import json


    def notify_completion(message: str):
        # Your URL from the first box in your image
        webhook_url = "https://default9a97c27db83e4694b35354bdbf18ab.5b.environment.api.powerplatform.com:443/powerautomate/automations/direct/workflows/d4ce1fe7f1994da5b5686894a14ddb00/triggers/manual/paths/invoke?api-version=1&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=JMMci_gLQcBkz_II3xwcfe9rKhferl9qk4jsz6GKuTc"

        # This is the exact format knockknock sends
        payload = {"text": message}

        response = requests.post(
            webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        if response.status_code == 200:
            print("Notification sent successfully!")
        else:
            print(
                f"Failed to send notification. Status code: {response.status_code}, Response: {response.text}"
            )
    return (notify_completion,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
