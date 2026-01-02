from __future__ import annotations
import numpy as np
import shap
from .config import DEFAULT_BACKGROUND_SIZE
from .adapters import get_predict_fn

def compute_shap_values(
    model,
    X,
    model_type: str,
    task: str = "classification",
    background_size: int = DEFAULT_BACKGROUND_SIZE,
):
    if X.shape[0] > background_size:
        background = shap.sample(X, background_size, random_state=0)
    else:
        background = X

    predict_fn = get_predict_fn(model, model_type, task=task)
    explainer = shap.Explainer(predict_fn, background)
    shap_values = explainer(X)

    values = np.array(shap_values.values)
    base_values = np.array(shap_values.base_values)

    return values, base_values, explainer
