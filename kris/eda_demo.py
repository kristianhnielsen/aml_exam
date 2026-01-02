import marimo

__generated_with = "0.18.4"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _():
    import marimo as mo
    import math
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import altair as alt
    from sklearn.preprocessing import (
        StandardScaler,
        OneHotEncoder,
        LabelEncoder,
        MinMaxScaler,
    )
    from sklearn.decomposition import PCA, KernelPCA
    from sklearn.model_selection import (
        train_test_split,
        RandomizedSearchCV,
        GridSearchCV,
        cross_val_score,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_curve,
        auc,
        roc_auc_score,
        accuracy_score,
        RocCurveDisplay,
        silhouette_score,
        ConfusionMatrixDisplay,
        f1_score,
        recall_score,
    )

    from sklearn.cluster import KMeans
    return (
        ConfusionMatrixDisplay,
        LogisticRegression,
        OneHotEncoder,
        PCA,
        StandardScaler,
        accuracy_score,
        alt,
        confusion_matrix,
        f1_score,
        math,
        mo,
        np,
        pd,
        plt,
        recall_score,
        roc_auc_score,
        sns,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(pd):
    raw_data = pd.read_csv("./data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    raw_data.columns = raw_data.columns.str.lower()
    raw_data.drop(columns=["customerid", "gender"], inplace=True)
    return (raw_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TotalCharges missing values
    """)
    return


@app.cell(hide_code=True)
def _(raw_data):
    _total_charges_missing = raw_data[raw_data["totalcharges"] == " "]
    _total_charges_missing
    return


@app.cell(hide_code=True)
def _(raw_data):
    raw_data.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cleaned Data
    """)
    return


@app.cell(hide_code=True)
def _(raw_data):
    _prepped_data = raw_data.drop(
        labels=raw_data[raw_data["totalcharges"] == " "].index
    )
    _prepped_data["totalcharges"] = _prepped_data["totalcharges"].astype(float)
    # make seniorcitizen categorical
    _prepped_data["seniorcitizen"] = _prepped_data["seniorcitizen"].apply(
        lambda x: "Yes" if x == 1 else "No"
    )
    clean_data = _prepped_data.copy()
    clean_data
    return (clean_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data sets
    - OHE
    - OHE with customer segments
    - OHE with PCA
    """)
    return


@app.cell(hide_code=True)
def _(df_pca_combined, mo, ohe_data, ohe_data_segment, ohe_data_segment_str):
    datasets = {
        "OHE": ohe_data,
        "OHE with Segments (labeled)": ohe_data_segment_str,
        "OHE with Segments (OHE)": ohe_data_segment,
        "PCA": df_pca_combined,
    }

    dataset_selector = mo.ui.radio(
        options=list(datasets.keys()),
        label="Select Dataset:",
        value=list(datasets.keys())[0],
    )
    return dataset_selector, datasets


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### OHE Data
    """)
    return


@app.cell(hide_code=True)
def _(OneHotEncoder, clean_data, np, pd):
    _one_hot_encoder = OneHotEncoder(sparse_output=False)

    binary_cols = [
        col
        for col in clean_data.select_dtypes(include="object").columns
        if clean_data[col].nunique() == 2
    ]
    _encoded_binary = clean_data[binary_cols].apply(
        lambda x: x.map({"Yes": 1, "No": 0})
    )

    categorical_cols = [
        col
        for col in clean_data.select_dtypes(include="object").columns
        if clean_data[col].nunique() > 2
    ]
    _encoded_categorical = pd.DataFrame(
        _one_hot_encoder.fit_transform(clean_data[categorical_cols]),
        columns=_one_hot_encoder.get_feature_names_out(categorical_cols),
        index=clean_data.index,
    ).astype(np.int64)


    numerical_cols = clean_data.select_dtypes(
        include=["float64", "int64"]
    ).columns.tolist()


    # Combine all encoded data
    ohe_data = pd.concat(
        [
            clean_data[numerical_cols],
            _encoded_categorical,
            _encoded_binary,
        ],
        axis=1,
    )
    # ohe_data[_numerical_cols] = StandardScaler().fit_transform(
    #     ohe_data[_numerical_cols]
    # )
    ohe_data
    return (ohe_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### OHE data with customer segments
    """)
    return


@app.cell(hide_code=True)
def _(OneHotEncoder, clean_data, np, ohe_data, pd):
    _segments_clusters = {
        "senior solitary": (
            (clean_data["seniorcitizen"] == "Yes")
            & (clean_data["partner"] == "No")
            & (clean_data["dependents"] == "No")
        ),
        "senior couple": (
            (clean_data["seniorcitizen"] == "Yes")
            & (clean_data["partner"] == "Yes")
            & (clean_data["dependents"] == "No")
        ),
        "senior single guardian": (
            (clean_data["seniorcitizen"] == "Yes")
            & (clean_data["partner"] == "No")
            & (clean_data["dependents"] == "Yes")
        ),
        "senior nuclear family": (
            (clean_data["seniorcitizen"] == "Yes")
            & (clean_data["partner"] == "Yes")
            & (clean_data["dependents"] == "Yes")
        ),
        "non-senior solitary": (
            (clean_data["seniorcitizen"] == "No")
            & (clean_data["partner"] == "No")
            & (clean_data["dependents"] == "No")
        ),
        "non-senior couple": (
            (clean_data["seniorcitizen"] == "No")
            & (clean_data["partner"] == "Yes")
            & (clean_data["dependents"] == "No")
        ),
        "non-senior single parent": (
            (clean_data["seniorcitizen"] == "No")
            & (clean_data["partner"] == "No")
            & (clean_data["dependents"] == "Yes")
        ),
        "non-senior nuclear family": (
            (clean_data["seniorcitizen"] == "No")
            & (clean_data["partner"] == "Yes")
            & (clean_data["dependents"] == "Yes")
        ),
    }

    # Apply the logic
    ohe_data_segment = ohe_data.copy()
    for segment, condition in _segments_clusters.items():
        ohe_data_segment.loc[condition, "customer_segment"] = segment

    ohe_data_segment_str = (
        ohe_data_segment.copy()
    )  # keep a string version for later use

    # OHE segment
    _one_hot_encoder = OneHotEncoder(sparse_output=False)
    _encoded_segment = pd.DataFrame(
        _one_hot_encoder.fit_transform(ohe_data_segment[["customer_segment"]]),
        columns=_one_hot_encoder.get_feature_names_out(["customer_segment"]),
        index=ohe_data_segment.index,
    ).astype(np.int64)
    ohe_data_segment = pd.concat(
        [
            ohe_data_segment.drop(columns=["customer_segment"]),
            _encoded_segment,
        ],
        axis=1,
    )
    ohe_data_segment
    return ohe_data_segment, ohe_data_segment_str


@app.cell(hide_code=True)
def _(mo, ohe_data):
    pca_select_col = mo.ui.multiselect(
        options=ohe_data.columns.tolist(),
        label="Select PCA Columns:",
        value=ohe_data.columns.tolist(),
    )
    return (pca_select_col,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PCA data
    """)
    return


@app.cell(hide_code=True)
def _(df_pca_combined):
    df_pca_combined
    return


@app.cell(hide_code=True)
def _(PCA, StandardScaler, alt, mo, np, ohe_data, pca_select_col, pd):
    ## get selected columns for PCA
    _selected_cols = [
        col
        for col in ohe_data.columns
        if any(col.startswith(val) for val in pca_select_col.value)
    ]
    if _selected_cols == []:
        _selected_cols = ohe_data.columns.tolist()[0:2]
        mo.md("**No columns selected, defaulting to first two columns.**")

    X_scaled = StandardScaler().fit_transform(ohe_data[_selected_cols])

    pca = PCA()
    pca.fit(X_scaled)

    # 2. Create DataFrame for Altair
    # Altair operates best on Pandas DataFrames
    df_pca = pd.DataFrame(
        {
            "Component": range(1, len(pca.explained_variance_ratio_) + 1),
            "Explained Variance": pca.explained_variance_ratio_,
            "Cumulative Variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )


    # 3. Create Plots

    # Scree Plot (Bar Chart)
    # scree = (
    #     alt.Chart(df_pca)
    #     .mark_bar()
    #     .encode(
    #         x=alt.X(
    #             "Component:O", title="Principal Component"
    #         ),  # :O treats it as ordinal (discrete)
    #         y=alt.Y("Explained Variance", title="Variance Ratio"),
    #         tooltip=["Component", alt.Tooltip("Explained Variance", format=".3f")],
    #     )
    #     .properties(title="Scree Plot")
    # )

    # Cumulative Plot (Line Chart)
    cumulative = (
        alt.Chart(df_pca)
        .mark_line(point=True)
        .encode(
            x=alt.X("Component:O", title="Principal Component"),
            y=alt.Y("Cumulative Variance", title="Cumulative Ratio"),
            tooltip=[
                "Component",
                alt.Tooltip("Cumulative Variance", format=".3f"),
            ],
        )
        .properties(title="Cumulative Explained Variance")
    )

    # Threshold Line (95%)
    rule = (
        alt.Chart(pd.DataFrame({"y": [0.95]}))
        .mark_rule(color="red", strokeDash=[5, 5])
        .encode(y="y")
    )

    # put dataframe back together with the pca
    _df_full = ohe_data.copy()
    _df_full.drop(columns=_selected_cols, inplace=True)
    df_pca_combined = pd.concat(
        [_df_full, pd.DataFrame(pca.transform(X_scaled))], axis=1
    )


    mo.vstack(
        [
            mo.md(
                "**No columns selected for PCA. Please select at least one column to display PCA plots.**"
            )
            if pca_select_col.value == []
            else "",
            pca_select_col,
            # scree.interactive(),
            (cumulative + rule).interactive(),
        ]
    )
    return (df_pca_combined,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PCA Conclusion

    PCA on services does not result in significant dimensionality reduction as the first few components do not capture a substantial portion of the variance. This suggests that the original features are relatively uncorrelated and each contributes unique information. Therefore, using PCA may not be beneficial for this dataset, and retaining the original features could be more effective for modeling tasks.

    PCA on all columns (except `churn`) shows 17 or 18 components would contain > 95% of the variance, indicating some potential for dimensionality reduction while preserving most information. Taking into account the number of binary and ternary features, this dimentionality reduction may just be due to the one hot encoding, meaning that PCA will not likely perform much better than using the original features.
    """)
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    # Plots
    """)
    return


@app.cell(hide_code=True)
def _():
    column_categories = {
        "customer": ["seniorcitizen", "partner", "dependents"],
        "account": ["contract", "paperlessbilling", "paymentmethod"],
        "services": [
            "onlinesecurity",
            "onlinebackup",
            "deviceprotection",
            "techsupport",
            "streamingtv",
            "streamingmovies",
            "phoneservice",
            "multiplelines",
            "internetservice",
        ],
        "churn": ["churn"],
    }
    return (column_categories,)


@app.cell(hide_code=True)
def _(column_categories, mo):
    pie_select_col = mo.ui.radio(
        options=[col for col in column_categories.keys()],
        label="Select data category:",
        value=list(column_categories.keys())[0],
    )
    return (pie_select_col,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Proportional distribution of features on data categories
    """)
    return


@app.cell(hide_code=True)
def _(column_categories, math, mo, pie_select_col, plt, raw_data):
    cols_in_selected_category = column_categories[pie_select_col.value]
    num_plots = len(cols_in_selected_category)
    cols_per_row = 3 if num_plots >= 3 else num_plots
    num_rows = math.ceil(num_plots / cols_per_row)

    _fig, _axes = plt.subplots(
        num_rows, cols_per_row, figsize=(5 * cols_per_row, 5 * num_rows)
    )

    # Flatten axes array to 1D for easy looping (works even if rows=1)
    # Note: If num_plots is 1, wrap axes in a list or handle separately
    _axes = _axes.flatten() if num_plots > 1 else [_axes]


    for _i, _col in enumerate(cols_in_selected_category):
        raw_data[_col].value_counts().plot.pie(
            ax=_axes[_i],
            autopct="%1.1f%%",
            startangle=90,
            counterclock=False,
            title=f"Distribution of {_col}",
        )
        _axes[_i].set_ylabel("")
        _axes[_i].set_xlabel("")

    plt.tight_layout()
    mo.vstack([pie_select_col, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correlation Heatmap
    """)
    return


@app.cell(hide_code=True)
def _(np, pd, plt, raw_data, sns):
    plt.figure(figsize=(25, 10))
    _corr = raw_data.apply(lambda x: pd.factorize(x)[0]).corr()

    mask = np.triu(np.ones_like(_corr, dtype=bool))

    sns.heatmap(
        _corr,
        mask=mask,
        xticklabels=_corr.columns.tolist(),
        yticklabels=_corr.columns.tolist(),
        annot=True,
        linewidths=0.2,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature proportion and distribution on churn
    """)
    return


@app.cell(hide_code=True)
def _(clean_data, mo, ohe_data_segment_str, pd):
    feature_dist_data = pd.concat(
        [clean_data, ohe_data_segment_str["customer_segment"]],
        axis=1,
    )
    feature_dist_select_col = mo.ui.radio(
        options=[
            col
            for col in feature_dist_data.select_dtypes(exclude="float64").columns
            if col != "churn"
        ],
        label="Select Column to Analyze Churn Distribution:",
        value=feature_dist_data.columns[0],
    )
    return feature_dist_data, feature_dist_select_col


@app.cell(hide_code=True)
def _(feature_dist_data, feature_dist_select_col, mo, pd, plt, sns):
    customer_count = sns.countplot(
        x=feature_dist_select_col.value,
        hue="churn",
        data=feature_dist_data,
    )
    customer_count.set_title(
        f"Volume: Count of Customers by {feature_dist_select_col.value}"
    )
    customer_count.set_ylabel("Number of Customers")
    customer_count.set_xlabel("")
    plt.xticks(rotation=70)


    cross_tab = pd.crosstab(
        feature_dist_data[feature_dist_select_col.value],
        feature_dist_data["churn"],
    )
    cross_tab_prop = cross_tab.div(cross_tab.sum(1), axis=0)

    customer_proportion = cross_tab_prop.plot(kind="bar", stacked=True)
    customer_proportion.set_title(
        f"Ratio: Churn Proportion by {feature_dist_select_col.value}"
    )
    customer_proportion.set_ylabel("Proportion")
    customer_proportion.set_xlabel("")
    plt.xticks(rotation=70)
    customer_proportion.legend(title="Churn", loc="upper right")

    plt.tight_layout()

    mo.hstack(
        [feature_dist_select_col, mo.vstack([customer_proportion, customer_count])]
    )
    return


@app.cell(hide_code=True)
def _(clean_data, plt, sns):
    _pseudo_total = clean_data["tenure"] * clean_data["monthlycharges"]
    _total_diff = clean_data["totalcharges"].astype(float) - _pseudo_total
    sns.scatterplot(data=clean_data, x=_pseudo_total, y="totalcharges")
    plt.xlabel("tenure × monthlycharges")
    plt.ylabel("totalcharges")
    plt.title("Linearity Check: tenure × monthlycharges vs totalcharges")
    plt.show()
    return


@app.cell
def _():
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    # Logistic Regression
    """)
    return


@app.cell(hide_code=True)
def _(dataset_selector):
    dataset_selector
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baseline
    """)
    return


@app.cell(hide_code=True)
def _(
    LogisticRegression,
    accuracy_score,
    dataset_selector,
    datasets,
    f1_score,
    train_test_split,
):
    _selected_dataset = datasets[dataset_selector.value]
    _y = _selected_dataset["churn"]
    _X = _selected_dataset.drop(columns=["churn"])

    _X_train, _X_test, _y_train, _y_test = train_test_split(
        _X, _y, test_size=0.2, stratify=_y, random_state=42
    )

    _base_log_reg = LogisticRegression(max_iter=100_000).fit(_X_train, _y_train)
    _base_preds = _base_log_reg.predict(_X_test)
    _base_accuracy = accuracy_score(_y_test, _base_preds)
    _base_f1 = f1_score(_y_test, _base_preds)
    print(f"Baseline Accuracy: {_base_accuracy}")
    print(f"Baseline F1: {_base_f1}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SMOTE
    """)
    return


@app.cell(hide_code=True)
def _():
    import imblearn
    from collections import Counter

    from imblearn.over_sampling import SMOTE, ADASYN, KMeansSMOTE
    from imblearn.combine import SMOTEENN
    return ADASYN, Counter, SMOTE, SMOTEENN


@app.cell(hide_code=True)
def _():
    # _selected_dataset = datasets[dataset_selector.value]
    # _y = _selected_dataset["churn"]
    # _X = _selected_dataset.drop(columns=["churn"])

    # base_results = {}
    # smote_results = {}
    # smoteenn_results = {}
    # adasyn_results = {}


    # _iterations = 3
    # for _i in range(_iterations):
    #     _X_train, _X_test, _y_train, _y_test = train_test_split(
    #         _X, _y, test_size=0.2, stratify=_y
    #     )
    #     # Baseline model
    #     _base_log_reg = LogisticRegression(max_iter=100_000).fit(
    #         _X_train, _y_train
    #     )
    #     base_preds = _base_log_reg.predict(_X_test)
    #     base_accuracy = accuracy_score(_y_test, base_preds)
    #     base_f1 = f1_score(_y_test, base_preds)
    #     base_results.append(base_accuracy)

    #     # Test SMOTE models
    #     _smote_models = [SMOTEENN(), ADASYN(), SMOTE()]
    #     for _smote_model in _smote_models:
    #         _X_resampled_train, _y_resampled_train = _smote_model.fit_resample(
    #             _X_train, _y_train
    #         )

    #         smote_log_reg = LogisticRegression(max_iter=100_000).fit(
    #             _X_resampled_train, _y_resampled_train
    #         )

    #         smote_preds = smote_log_reg.predict(_X_test)
    #         smote_accuracy = accuracy_score(_y_test, smote_preds)
    #         smote_f1 = f1_score(_y_test, smote_preds)

    #         if isinstance(_smote_model, SMOTEENN):
    #             smoteenn_results[_i] = {"accuracy": smote_accuracy, "f1": smote_f1}
    #         elif isinstance(_smote_model, SMOTE):
    #             smote_results[_i] = {"accuracy": smote_accuracy, "f1": smote_f1}
    #         elif isinstance(_smote_model, ADASYN):
    #             adasyn_results[_i] = {"accuracy": smote_accuracy, "f1": smote_f1}


    # # Print class distributions
    # _X_train, _X_test, _y_train, _y_test = train_test_split(
    #     _X, _y, test_size=0.2, stratify=_y
    # )
    # # Baseline model
    # print("Baseline Model classes:", Counter(_y_train))
    # # Test SMOTE models
    # _smote_models = [SMOTEENN(), ADASYN(), SMOTE()]
    # for _smote_model in _smote_models:
    #     _X_resampled_train, _y_resampled_train = _smote_model.fit_resample(
    #         _X_train, _y_train
    #     )
    #     print(
    #         f"{_smote_model.__class__.__name__} classes:",
    #         Counter(_y_resampled_train),
    #     )


    # print(f"\n\nResults of {_iterations} iterations:")
    # print("Baseline Mean Accuracy: ", np.mean(base_results))
    # print("SMOTEENN Mean Accuracy: ", np.mean(smoteenn_results))
    # print("ADASYN Mean Accuracy: ", np.mean(adasyn_results))
    # print("\nBaseline Max Accuracy: ", max(base_results))
    # print("SMOTEENN Max Accuracy: ", max(smoteenn_results))
    # print("ADASYN Max Accuracy: ", max(adasyn_results))
    return


@app.cell(hide_code=True)
def _(mo):
    smote_iterations_slider = mo.ui.slider(
        1,
        20,
        value=2,
        show_value=True,
        label="Number of iterations to run SMOTE: ",
    )
    smote_iterations_slider
    return (smote_iterations_slider,)


@app.cell(hide_code=True)
def _(
    ADASYN,
    Counter,
    LogisticRegression,
    SMOTE,
    SMOTEENN,
    accuracy_score,
    dataset_selector,
    datasets,
    f1_score,
    recall_score,
    smote_iterations_slider,
    train_test_split,
):
    # Setup Data
    _selected_dataset = datasets[dataset_selector.value]
    _y = _selected_dataset["churn"]
    _X = _selected_dataset.drop(columns=["churn"])

    # Dictionary to store lists of scores
    # Now tracking: Accuracy, F1, and Recall
    smote_model_metrics = {
        "Baseline": {"accuracy": [], "f1": [], "recall": []},
        "SMOTE": {"accuracy": [], "f1": [], "recall": []},
        "SMOTEENN": {"accuracy": [], "f1": [], "recall": []},
        "ADASYN": {"accuracy": [], "f1": [], "recall": []},
    }

    _iterations = smote_iterations_slider.value

    print(f"--- Starting {_iterations} Iterations ---")

    for _i in range(_iterations):
        # 1. Stratified Split
        _X_train, _X_test, _y_train, _y_test = train_test_split(
            _X, _y, test_size=0.2, stratify=_y
        )

        # 2. Define the pipeline of samplers to test
        samplers = [
            ("Baseline", None),
            ("SMOTE", SMOTE(random_state=42)),
            ("SMOTEENN", SMOTEENN(random_state=42)),
            ("ADASYN", ADASYN(random_state=42)),
        ]

        for name, sampler in samplers:
            # Resample logic
            if sampler:
                try:
                    X_train_res, y_train_res = sampler.fit_resample(
                        _X_train, _y_train
                    )
                except ValueError as e:
                    print(f"Skipping {name} iteration {_i} due to error: {e}")
                    continue
            else:
                X_train_res, y_train_res = _X_train, _y_train

            # Print Class Distribution (first iteration only)
            if _i == 0:
                print(
                    f"{name:<10} Training Class Distribution: {Counter(y_train_res)}"
                )

            # Train
            clf = LogisticRegression(max_iter=100_000, random_state=42)
            clf.fit(X_train_res, y_train_res)

            # Predict
            preds = clf.predict(_X_test)

            # Store Metrics
            smote_model_metrics[name]["accuracy"].append(
                accuracy_score(_y_test, preds)
            )
            smote_model_metrics[name]["f1"].append(f1_score(_y_test, preds))
            smote_model_metrics[name]["recall"].append(
                recall_score(_y_test, preds)
            )
    return (smote_model_metrics,)


@app.cell(hide_code=True)
def _(np, smote_model_metrics):
    # --- Results Output ---
    print(f"{'Model':<15} | {'Metric':<10} | {'Mean':<8} | {'Max':<8}")
    print("-" * 45)

    for _name, metrics in smote_model_metrics.items():
        if not metrics["accuracy"]:
            continue

        # Calculate stats
        mean_acc = np.mean(metrics["accuracy"])
        max_acc = np.max(metrics["accuracy"])

        mean_f1 = np.mean(metrics["f1"])
        max_f1 = np.max(metrics["f1"])

        mean_rec = np.mean(metrics["recall"])
        max_rec = np.max(metrics["recall"])

        print(f"{_name:<15} | Accuracy   | {mean_acc:.4f}   | {max_acc:.4f}")
        print(f"{_name:<15} | F1 Score   | {mean_f1:.4f}   | {max_f1:.4f}")
        print(f"{_name:<15} | Recall     | {mean_rec:.4f}   | {max_rec:.4f}")
        print("-" * 45)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cox Proportional Hazards Model
    """)
    return


@app.cell(hide_code=True)
def _():
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.plotting import (
        plot_interval_censored_lifetimes,
        plot_lifetimes,
        add_at_risk_counts,
    )
    return CoxPHFitter, KaplanMeierFitter


@app.cell(hide_code=True)
def _(CoxPHFitter, dataset_selector, datasets, train_test_split):
    _selected_dataset = datasets[dataset_selector.value]
    _train_data, _test_data = train_test_split(
        _selected_dataset,
        test_size=0.2,
        random_state=42,
        stratify=_selected_dataset["churn"],
    )

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df=_train_data, duration_col="tenure", event_col="churn")

    print(
        f"Train C-index: {cph.score(_train_data, scoring_method='concordance_index'):.3f}"
    )
    print(
        f"Test  C-index: {cph.score(_test_data, scoring_method='concordance_index'):.3f}"
    )
    return (cph,)


@app.cell(hide_code=True)
def _(cph):
    cph.print_summary()
    return


@app.cell(hide_code=True)
def _(cph):
    cox_significant_summary = cph.summary[cph.summary["p"] < 0.05]
    cox_significant_summary[["coef", "exp(coef)", "p"]]
    return (cox_significant_summary,)


@app.cell(hide_code=True)
def _(alt, cox_significant_summary, pd):
    _chart = (
        alt.Chart(cox_significant_summary.reset_index())
        .mark_bar()
        .encode(
            x=alt.X(
                field="exp(coef)",
                type="quantitative",
                title="Hazard Ration (Exp(Coef))",
            ),
            y=alt.Y(field="covariate", type="nominal", sort="-x", title=None),
            tooltip=[
                alt.Tooltip(field="exp(coef)", format=",.2f"),
                alt.Tooltip(field="covariate"),
            ],
            color=alt.condition(
                alt.datum["exp(coef)"] > 1,
                alt.value("#d62728"),  # Red
                alt.value("#1f77b4"),  # Blue
            ),
        )
    )
    _rule = (
        alt.Chart(pd.DataFrame({"x": [1]}))
        .mark_rule(color="black", strokeDash=[4, 4])
        .encode(x="x")
    )
    _chart = (_chart + _rule).properties(
        height=290,
        width="container",
        config={"axis": {"grid": False}},
        title="Cox Proportional Hazards Model Significant Exp(Coef)",
    )
    _chart
    return


@app.cell(hide_code=True)
def _(alt, cox_significant_summary):
    _chart = (
        alt.Chart(cox_significant_summary.reset_index())
        .mark_bar()
        .encode(
            x=alt.X(field="coef", type="quantitative"),
            y=alt.Y(field="covariate", type="nominal", sort="-x", title=None),
            tooltip=[
                alt.Tooltip(field="coef", format=",.2f"),
                alt.Tooltip(field="covariate"),
            ],
        )
        .properties(
            height=290,
            width="container",
            config={"axis": {"grid": False}},
            title="Cox Proportional Hazards Model Significant Coefficients",
        )
        .interactive()
    )
    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Logistic Regression Comparison
    """)
    return


@app.cell(hide_code=True)
def _(
    ConfusionMatrixDisplay,
    LogisticRegression,
    accuracy_score,
    confusion_matrix,
    cox_significant_summary,
    dataset_selector,
    datasets,
    plt,
    roc_auc_score,
    train_test_split,
):
    _selected_dataset = datasets[dataset_selector.value]
    _significant_cols = cox_significant_summary.T.columns.tolist()
    _y = _selected_dataset["churn"]
    _X = _selected_dataset[_significant_cols]
    print(f"Number of columns: {len(_X.columns)}")

    _X_train, _X_test, _y_train, _y_test = train_test_split(
        _X, _y, test_size=0.2, random_state=42, stratify=_y
    )

    _model = LogisticRegression()
    _model.fit(_X_train, _y_train)

    _y_preds = _model.predict(_X_test)
    _acc = accuracy_score(_y_preds, _y_test)
    _auc = roc_auc_score(y_true=_y_test, y_score=_y_preds)

    print(f"Accuracy: {_acc}")
    print(f"AUC: {_auc}")
    _conf_matrix = confusion_matrix(
        y_pred=_y_preds, y_true=_y_test, normalize="all"
    )
    ConfusionMatrixDisplay(_conf_matrix).plot()
    plt.show()
    return


@app.cell(hide_code=True)
def _(
    ConfusionMatrixDisplay,
    LogisticRegression,
    accuracy_score,
    confusion_matrix,
    dataset_selector,
    datasets,
    plt,
    roc_auc_score,
    train_test_split,
):
    _selected_dataset = datasets[dataset_selector.value]
    _y = _selected_dataset["churn"]
    _X = _selected_dataset.drop(columns=["churn"])
    print(f"Number of columns: {len(_X.columns)}")

    _X_train, _X_test, _y_train, _y_test = train_test_split(
        _X, _y, test_size=0.2, random_state=42, stratify=_y
    )

    _model = LogisticRegression()
    _model.fit(_X_train, _y_train)

    _y_preds = _model.predict(_X_test)
    _acc = accuracy_score(_y_preds, _y_test)
    _auc = roc_auc_score(y_true=_y_test, y_score=_y_preds)

    print(f"Accuracy: {_acc}")
    print(f"AUC: {_auc}")
    _conf_matrix = confusion_matrix(
        y_pred=_y_preds, y_true=_y_test, normalize="all"
    )
    ConfusionMatrixDisplay(_conf_matrix).plot()
    plt.show()
    return


@app.cell
def _():
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""
    # Survival Functions
    """)
    return


@app.cell(hide_code=True)
def _(dataset_selector):
    dataset_selector
    return


@app.cell
def _(KaplanMeierFitter, dataset_selector, datasets, plt):
    _selected_dataset = datasets[dataset_selector.value]
    _kmf = KaplanMeierFitter()
    _kmf.fit(
        durations=_selected_dataset["tenure"],
        event_observed=_selected_dataset["churn"],
    )
    plt.figure(figsize=(15, 6))
    _kmf.plot_survival_function(at_risk_counts=True)
    plt.title("Kaplan-Meier Survival Function (All columns)")
    plt.grid(True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(KaplanMeierFitter, dataset_selector, datasets, plt):
    def plot_survival_function_on_columns(columns: list[str]):
        df = datasets[dataset_selector.value]

        for _col in columns:
            _unique_values = df[_col].unique()

            # Create a new figure for THIS column
            plt.figure(figsize=(15, 8))
            _ax = plt.subplot(111)
            _fitters = []

            for _value in _unique_values:
                _kmf = KaplanMeierFitter()
                # Filter dynamically using the current column and value
                _subset = df[df[_col] == _value]

                _kmf.fit(
                    durations=_subset["tenure"],
                    event_observed=_subset["churn"],
                    label=str(_value),
                )

                _kmf.plot_survival_function(ax=_ax, ci_show=False)
                _fitters.append(_kmf)

            # add_at_risk_counts(*_fitters, ax=_ax)

            _ax.set_title(f"Kaplan-Meier Survival Function by {_col}")
            _ax.grid(True)
            plt.tight_layout()
            plt.show()
    return (plot_survival_function_on_columns,)


@app.cell
def _(plot_survival_function_on_columns):
    plot_survival_function_on_columns(
        columns=[
            "seniorcitizen",
            "dependents",
            "partner",
            "paperlessbilling",
            "phoneservice",
        ]
    )
    return


@app.cell
def _(plot_survival_function_on_columns):
    plot_survival_function_on_columns(['customer_segment'])
    return


@app.cell
def _(dataset_selector, datasets):
    datasets[dataset_selector.value].columns
    return


@app.cell
def _(dataset_selector, datasets, plot_survival_function_on_columns):
    plot_survival_function_on_columns(datasets[dataset_selector.value].columns.tolist()[2:])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
