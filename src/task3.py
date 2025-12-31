import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    from preprocessor import preprocessor
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import AdamW
    from experiment_tracking import ExperimentTracker
    import tensorflow as tf
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, accuracy_score
    return (
        AdamW,
        BatchNormalization,
        Dense,
        Dropout,
        EarlyStopping,
        ExperimentTracker,
        ModelCheckpoint,
        ReduceLROnPlateau,
        Sequential,
        accuracy_score,
        preprocessor,
        roc_auc_score,
        tf,
        xgb,
    )


@app.cell
def _(preprocessor):
    # Load and preprocess data
    X_train, X_test, y_train, y_test = preprocessor()
    input_dim = X_train.shape[1]
    return X_test, X_train, input_dim, y_test, y_train


@app.cell
def _(BatchNormalization, Dense, Dropout, Sequential, tf):
    # Define Feed Forward Neural Network model
    def create_ffnn(input_dim, dropout_rate=0.2, l2_reg=0.0001):
        model = Sequential()
        model.add(
            Dense(
                512,
                activation="relu",
                input_dim=input_dim,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(
            Dense(
                256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(
            Dense(
                128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation="sigmoid"))
        return model
    return (create_ffnn,)


@app.cell
def _(ExperimentTracker):
    mlflow = ExperimentTracker(experiment_name="AML Task 3")
    return (mlflow,)


@app.cell
def _(
    AdamW,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    X_test,
    X_train,
    create_ffnn,
    input_dim,
    mlflow,
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
        monitor="val_loss", patience=10, restore_best_weights=True, min_delta=0.0001
    )
    checkpoint = ModelCheckpoint(
        "best_ffnn_model.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
    )
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)

    with mlflow.start_run(run_name="Feed Forward Neural Network - run 3"):
        history = ffnn_model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, lr_scheduler],
        )

        # Evaluate the model
        test_loss, test_accuracy = ffnn_model.evaluate(X_test, y_test)
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_accuracy})
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    return


@app.cell
def _(Dense, tf):
    # Create Autoencoder model
    def create_autoencoder(input_dim, encoding_dim=64):
        input_layer = tf.keras.Input(shape=(input_dim,))
        encoded = Dense(256, activation="relu")(input_layer)
        encoded = Dense(128, activation="relu")(encoded)
        encoded = Dense(encoding_dim, activation="relu")(encoded)

        decoded = Dense(128, activation="relu")(encoded)
        decoded = Dense(256, activation="relu")(decoded)
        decoded = Dense(input_dim, activation="sigmoid")(decoded)

        autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoded)
        encoder = tf.keras.Model(inputs=input_layer, outputs=encoded)
        return autoencoder, encoder
    return (create_autoencoder,)


@app.cell
def _(
    EarlyStopping,
    ModelCheckpoint,
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
        monitor="val_loss", patience=10, restore_best_weights=True, min_delta=0.0001
    )
    checkpoint_ae = ModelCheckpoint(
        "best_autoencoder_model.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
    )
    lr_scheduler_ae = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)

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


@app.cell
def _(
    AdamW,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    X_test_encoded,
    X_train_encoded,
    create_ffnn,
    mlflow,
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
        monitor="val_loss", patience=10, restore_best_weights=True, min_delta=0.0001
    )
    checkpoint_enc = ModelCheckpoint(
        "best_ffnn_encoded_model.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
    )
    lr_scheduler_enc = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)

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
        test_loss_enc, test_accuracy_enc = ffnn_encoded_model.evaluate(
            X_test_encoded, y_test
        )
        mlflow.log_metrics(
            {"test_loss_encoded": test_loss_enc, "test_accuracy_encoded": test_accuracy_enc}
        )
        print(
            f"Test Loss (Encoded): {test_loss_enc}, Test Accuracy (Encoded): {test_accuracy_enc}"
        )
    return


@app.cell
def _(
    X_test_encoded,
    X_train_encoded,
    accuracy_score,
    mlflow,
    roc_auc_score,
    xgb,
    y_test,
    y_train,
):
    # Use the encoder for feature extraction and train XGBoost
    with mlflow.start_run(run_name="XGBoost on Autoencoder Features - run 1"):
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
        y_pred = (y_pred_proba >= 0.5).astype(int)
        test_accuracy_xgb = accuracy_score(y_test, y_pred)
        mlflow.log_metrics(
            {"test_auc_xgb": test_auc_xgb, "test_accuracy_xgb": test_accuracy_xgb}
        )
        print(f"Test AUC (XGBoost on Encoded): {test_auc_xgb}")
        print(f"Test Accuracy (XGBoost on Encoded): {test_accuracy_xgb}")
    return


if __name__ == "__main__":
    app.run()
