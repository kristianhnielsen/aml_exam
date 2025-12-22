## Task 3: Neural Network Models

# **Feed Forward Neural Networks**

# - Apply regularization: dropout, batch normalization, L1, L2, Early Stopping, ..etc
# - Experiment with different optimizers, learning rate, activation functions,...etc
# - Analyze training curves (loss/accuracy)

# **Autoencoder**
# - Train an Autoencoder for feature compression
# - Feed the latent embedding into shallow ML / FFNN (use the AE for feature extraction)
# - Compare performance vs raw features

# %%
from preprocessor import preprocessor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, AdamW
import mlflow
from experiment_tracking import ExperimentTracker
import tensorflow as tf

# %%
# Load and preprocess data
X_train, X_test, y_train, y_test = preprocessor()
input_dim = X_train.shape[1]


# %%
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


# %%
mlflow = ExperimentTracker(experiment_name="AML Exam Experiment - Task 3")

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

# %%
