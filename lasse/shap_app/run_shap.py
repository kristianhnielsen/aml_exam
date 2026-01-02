from __future__ import annotations
import argparse
import numpy as np
from .adapters import load_model
from .explainers import compute_shap_values
from .io_utils import ensure_output_dir, load_tabular_data, save_shap_arrays
from .plotting import save_summary_plot, save_single_force_plot_html

def run_shap(
    model_path: str,
    data_path: str,
    model_type: str,
    task: str = "classification",
    output_dir: str | None = None,
    make_plots: bool = True,
):
    """
    - load model
    - load data
    - run SHAP
    - save outputs (arrays, plots)
    """

    output_dir = ensure_output_dir(output_dir)

    model = load_model(model_path, model_type)

    X, feature_names = load_tabular_data(data_path)

    shap_values, expected_values, explainer = compute_shap_values(
        model=model,
        X=X,
        model_type=model_type,
        task=task,
    )

    save_shap_arrays(shap_values, expected_values, feature_names, output_dir)

    # plotting
    if make_plots:
        save_summary_plot(
            shap_values=shap_values,
            X=X,
            feature_names=feature_names,
            output_dir=output_dir,
            file_name="summary_plot.png",
        )

        # one example force plot for first instance
        if X.shape[0] > 0:
            x0 = X[0]
            shap0 = shap_values[0]
            save_single_force_plot_html(
                explainer,
                shap0,
                x0,
                feature_names,
                output_dir=output_dir,
                file_name="force_plot_example.html",
            )

    print(f"outputs saved in '{output_dir}'.")

def main():
    parser = argparse.ArgumentParser(description="SHAP")
    parser.add_argument("--model_path", type=str, required=True, help="saved model path")
    parser.add_argument("--data_path", type=str, required=True, help="path to data (csv or npy).")
    parser.add_argument("--model_type", type=str, required=True,
                        help="model type: sklearn, xgboost, tensorflow, pytorch, mytorch, tree, linear, add others in adapters.py")
    parser.add_argument("--task", type=str, default="classification",
                        help="task type: classification or regression")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="directory to save SHAP outputs")
    parser.add_argument("--no_plots", action="store_true",
                        help="if set, do not generate plots")
    args = parser.parse_args()

    run_shap(
        model_path=args.model_path,
        data_path=args.data_path,
        model_type=args.model_type,
        task=args.task,
        output_dir=args.output_dir,
        make_plots=not args.no_plots,
    )

if __name__ == "__main__":
    main()
