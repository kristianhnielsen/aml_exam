import os
import shap
import matplotlib.pyplot as plt

def save_summary_plot(
    shap_values,
    X,
    feature_names,
    output_dir: str,
    file_name: str = "summary_plot.png",
    max_display: int = 20,
):
    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    path = os.path.join(output_dir, file_name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()

def save_single_force_plot_html(
    explainer,
    shap_values_row,
    x_row,
    feature_names,
    output_dir: str,
    file_name: str = "force_plot.html",
):
    path = os.path.join(output_dir, file_name)

    # Handle new SHAP API (Explanation object) vs Legacy
    if hasattr(shap_values_row, "base_values"):
        base_value = shap_values_row.base_values
        values = shap_values_row.values
    else:
        base_value = explainer.expected_value
        values = shap_values_row

    shap_html = shap.force_plot(
        base_value,
        values,
        x_row,
        feature_names=feature_names,
        matplotlib=False
    )
    shap.save_html(path, shap_html)
