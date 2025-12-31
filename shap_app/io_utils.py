import os
import json
import numpy as np
import pandas as pd
from .config import DEFAULT_OUTPUT_DIR

def ensure_output_dir(output_dir: str | None = None) -> str:
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_tabular_data(path: str):
    """
    loads data for SHAP
    supports csv and npy for now
    returns (X, feature_names)
    """
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        return df.values, list(df.columns)
    elif path.endswith(".npy"):
        X = np.load(path)
        return X, None
    else:
        raise ValueError(f"unsupported data format, do you need to add a filetype?: {path}")

def save_shap_arrays(shap_values, expected_values, feature_names, output_dir: str):
    np.save(os.path.join(output_dir, "shap_values.npy"), shap_values)
    np.save(os.path.join(output_dir, "expected_values.npy"), expected_values)

    if feature_names is not None:
        with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
            json.dump(feature_names, f, indent=2)
