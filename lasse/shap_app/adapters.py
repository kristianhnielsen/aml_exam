from __future__ import annotations
import numpy as np

def load_model(model_path: str, model_type: str):
    """
    loads a model given its type and path
    sklearn expects a joblib or pickle file
    xgboost expects a saved xgboost model
    tensorflow expects a saved keras model
    pytorch expects a torch.save()'d model
    """
    model_type = model_type.lower()

    if model_type in ["sklearn", "tree", "linear"]:
        import joblib
        return joblib.load(model_path)

    if model_type == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return model

    if model_type == "tensorflow":
        import tensorflow as tf
        return tf.keras.models.load_model(model_path)

    if model_type in ["pytorch", "torch", "mytorch"]:
        import torch
        model = torch.load(model_path, map_location="cpu")
        model.eval()
        return model

    raise ValueError(f"unsupported model_type, do you need to add one?: {model_type}")

def get_predict_fn(model, model_type: str, task: str = "classification"):
    """
    wraps the model into a callable f(X: np.ndarray) -> np.ndarray (1D or 2D)
    """
    model_type = model_type.lower()

    if model_type in ["sklearn", "tree", "linear"]:
        def predict_fn(X: np.ndarray):
            if hasattr(model, "predict_proba") and task == "classification":
                proba = model.predict_proba(X)
                # assume churn is the positive class at index 1
                return proba[:, 1]
            return model.predict(X)
        return predict_fn

    if model_type == "xgboost":
        def predict_fn(X: np.ndarray):
            # xgboost returns probabilities for binary classification by default
            pred = model.predict(X)
            # ensure one dimension
            return np.array(pred).reshape(-1)
        return predict_fn

    if model_type == "tensorflow":
        def predict_fn(X: np.ndarray):
            import numpy as np
            out = model.predict(np.array(X), verbose=0)
            if out.ndim == 2 and out.shape[1] == 1:
                out = out[:, 0]
            return out
        return predict_fn

    if model_type in ["pytorch", "torch", "mytorch"]:
        import torch

        def predict_fn(X: np.ndarray):
            X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
            with torch.no_grad():
                out = model(X_tensor)
            out = out.cpu().numpy()
            if out.ndim == 2 and out.shape[1] == 1:
                out = out[:, 0]
            return out
        return predict_fn

    raise ValueError(f"unsupported model_type for predict_fn: {model_type}")
