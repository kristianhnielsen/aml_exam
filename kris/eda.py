import marimo

__generated_with = "0.18.4"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    # Introduction to the data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell
def _():
    import math
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import marimo as mo
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
    )

    from sklearn.cluster import KMeans
    return (
        KMeans,
        LabelEncoder,
        LogisticRegression,
        OneHotEncoder,
        PCA,
        StandardScaler,
        accuracy_score,
        alt,
        classification_report,
        math,
        mo,
        np,
        pd,
        plt,
        silhouette_score,
        sns,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Manipulation
    """)
    return


@app.cell
def _(pd):
    data = pd.read_csv("./data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    data.columns = data.columns.str.lower()
    data.drop(columns=["customerid", "gender"], inplace=True)

    # there were values with whitespace, which didn't register as NaN-like
    print("Before:\t", data.shape)
    data.drop(labels=data[data["totalcharges"] == " "].index, inplace=True)
    data["totalcharges"] = data["totalcharges"].astype(float)
    print("After:\t", data.shape)
    data.head()
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary
    """)
    return


@app.cell
def _(data):
    data.info()
    return


@app.cell
def _(data):
    data.describe(include="all")
    return


@app.cell
def _(data):
    data.isna().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checking Values
    """)
    return


@app.cell
def _(data):
    for _col in data.columns:
        print(f"{_col}: \t{data[_col].nunique()} unique values")
        print(data[_col].unique(), "\n")
    return


@app.cell
def _(data):
    data["tenure"].value_counts(sort=True)
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    # General Plots
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pie Charts
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
def _(column_categories, data, math, mo, pie_select_col, plt):
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
        data[_col].value_counts().plot.pie(
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
    Only a 26.6% of the observations are categorized as churn. Is this too imbalanced?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correlation Heatmap
    """)
    return


@app.cell(hide_code=True)
def _(data, np, pd, plt, sns):
    corr_data = data
    plt.figure(figsize=(25, 10))
    corr = corr_data.apply(lambda x: pd.factorize(x)[0]).corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))

    ax = sns.heatmap(
        corr,
        mask=mask,
        xticklabels=corr.columns.tolist(),
        yticklabels=corr.columns.tolist(),
        annot=True,
        linewidths=0.2,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    plt.show()
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    # Churn vs No Churn
    """)
    return


@app.cell(hide_code=True)
def _(data, mo):
    select_col = mo.ui.radio(
        options=[
            col
            for col in data.select_dtypes(exclude="float64").columns
            if col != "churn"
        ],
        label="Select Column to Analyze Churn Distribution:",
        value=data.columns[0],
    )
    return (select_col,)


@app.cell(hide_code=True)
def _(data, mo, pd, plt, select_col, sns):
    customer_count = sns.countplot(
        x=select_col.value,
        hue="churn",
        data=data,
    )
    customer_count.set_title(f"Volume: Count of Customers by {select_col.value}")
    customer_count.set_ylabel("Number of Customers")


    cross_tab = pd.crosstab(data[select_col.value], data["churn"])
    cross_tab_prop = cross_tab.div(cross_tab.sum(1), axis=0)

    customer_proportion = cross_tab_prop.plot(kind="bar", stacked=True)
    customer_proportion.set_title(f"Ratio: Churn Proportion by {select_col.value}")
    customer_proportion.set_ylabel("Proportion")
    plt.xticks(rotation=70)
    customer_proportion.legend(title="Churn", loc="upper right")

    plt.tight_layout()

    mo.hstack([select_col, mo.vstack([customer_proportion, customer_count])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Slightly more senior citizens and single customers (no partner and/or no dependents) churn.

    - Customers on a month-to-month `contract` are much more likely to churn.

    - Customers who opt for `paperlessbilling` and electronic check `paymentmethod` are more likely to churn.

    - People who **don't** buy the add-on services for `onlinesecurity`, `onlinebackup`, `deviceprotection` and `techsupport` churn.

    - Porportions are the same for `streamingtv` & `streamingmovies`: ~50/50 on Yes/No

    - `phoneservice` & `multiplelines` have no significant difference in proportion
    """)
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""
    # Total Charges vs Monthly Charges
    I assumed that Total Charges was an interaction between Monthly Charges X Tenure.

    Although there is a linear relation, the difference between the assumed interaction and the Total Charges value are too far apart to assume that the assumption holds.
    """)
    return


@app.cell
def _(data):
    pseudo_total = data["tenure"] * data["monthlycharges"]
    total_diff = data["totalcharges"].astype(float) - pseudo_total
    return pseudo_total, total_diff


@app.cell(hide_code=True)
def _(data, plt, pseudo_total, sns):
    sns.scatterplot(data=data, x=pseudo_total, y="totalcharges")
    plt.xlabel("tenure × monthlycharges")
    plt.ylabel("totalcharges")
    plt.title("Linearity Check: tenure × monthlycharges vs totalcharges")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    bin_slider = mo.ui.slider(1, 50, value=30, show_value=True)
    return (bin_slider,)


@app.cell(hide_code=True)
def _(bin_slider, mo, sns, total_diff):
    _hist_plot = sns.histplot(total_diff, bins=bin_slider.value)

    mo.vstack([mo.md(f"Number of bins: {bin_slider}"), _hist_plot])
    return


@app.cell
def _():
    return


@app.cell(column=4, hide_code=True)
def _(mo):
    mo.md(r"""
    # PCA
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prep data
    """)
    return


@app.cell(hide_code=True)
def _(data):
    binary_cols = [
        col
        for col in data.select_dtypes(exclude="float64").columns
        if data[col].nunique() == 2 and col != "seniorcitizen"
    ]
    ternary_cols = [
        col
        for col in data.select_dtypes(exclude="float64").columns
        if data[col].nunique() == 3
    ]
    categorical_cols = [
        col
        for col in data.select_dtypes(exclude="float64").columns
        if data[col].nunique() > 3
    ]

    numerical_cols = data.select_dtypes(include="float64").columns.tolist()

    # binary_cols, ternary_cols, categorical_cols, numerical_cols
    return binary_cols, categorical_cols, numerical_cols, ternary_cols


@app.cell(hide_code=True)
def _(
    OneHotEncoder,
    binary_cols,
    categorical_cols,
    data,
    numerical_cols,
    pd,
    ternary_cols,
):
    ohe = OneHotEncoder(sparse_output=False)


    encoded_binary = data[binary_cols].apply(lambda x: x.map({"Yes": 1, "No": 0}))

    encoded_ternary = pd.DataFrame(
        ohe.fit_transform(data[ternary_cols]),
        columns=ohe.get_feature_names_out(ternary_cols),
        index=data.index,
    )


    encoded_categorical = pd.DataFrame(
        ohe.fit_transform(data[categorical_cols]),
        columns=ohe.get_feature_names_out(categorical_cols),
        index=data.index,
    )

    encoded_data = pd.concat(
        [
            data[numerical_cols],
            data[["seniorcitizen"]],
            encoded_categorical,
            encoded_binary,
            encoded_ternary,
        ],
        axis=1,
    )
    encoded_data
    return (encoded_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PCA Plots
    """)
    return


@app.cell(hide_code=True)
def _(categorical_cols, mo, ternary_cols):
    pca_select_col = mo.ui.multiselect(
        options=[*ternary_cols, *categorical_cols],
        label="Select PCA Columns:",
        value=ternary_cols,
    )
    return (pca_select_col,)


@app.cell(hide_code=True)
def _(PCA, StandardScaler, alt, encoded_data, mo, np, pca_select_col, pd):
    ## get selected columns for PCA
    _selected_cols = [
        col
        for col in encoded_data.columns
        if any(col.startswith(val) for val in pca_select_col.value)
    ]
    if _selected_cols == []:
        _selected_cols = encoded_data.columns.tolist()[0:2]
        mo.md("**No columns selected, defaulting to first two columns.**")

    X_scaled = StandardScaler().fit_transform(encoded_data[_selected_cols])

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
    scree = (
        alt.Chart(df_pca)
        .mark_bar()
        .encode(
            x=alt.X(
                "Component:O", title="Principal Component"
            ),  # :O treats it as ordinal (discrete)
            y=alt.Y("Explained Variance", title="Variance Ratio"),
            tooltip=["Component", alt.Tooltip("Explained Variance", format=".3f")],
        )
        .properties(title="Scree Plot")
    )

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
    _df_full = encoded_data.copy()
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
            scree.interactive(),
            (cumulative + rule).interactive(),
        ]
    )
    return


@app.cell(column=5, hide_code=True)
def _(mo):
    mo.md(r"""
    # Clustering
    """)
    return


@app.cell(hide_code=True)
def _(data):
    segments_clusters = {
        "senior single": (data["seniorcitizen"] == 1)
        & (data["partner"] == "No")
        & (data["dependents"] == "No"),
        "senior couple": (data["seniorcitizen"] == 1)
        & (data["partner"] == "Yes")
        & (data["dependents"] == "No"),
        "senior single parent": (data["seniorcitizen"] == 1)
        & (data["partner"] == "No")
        & (data["dependents"] == "Yes"),
        "senior with family": (data["seniorcitizen"] == 1)
        & ((data["partner"] == "Yes") | (data["dependents"] == "Yes")),
        "non-senior single": (data["seniorcitizen"] == 0)
        & (data["partner"] == "No")
        & (data["dependents"] == "No"),
        "non-senior couple": (data["seniorcitizen"] == 0)
        & (data["partner"] == "Yes")
        & (data["dependents"] == "No"),
        "non-senior single parent": (data["seniorcitizen"] == 0)
        & (data["partner"] == "No")
        & (data["dependents"] == "Yes"),
        "non-senior with family": (data["seniorcitizen"] == 0)
        & ((data["partner"] == "Yes") | (data["dependents"] == "Yes")),
    }
    clustering_data = data.copy()
    for segment, condition in segments_clusters.items():
        clustering_data.loc[condition, "customer_segment"] = segment
    # clustering_data["totalcharges_log"] = np.log1p(clustering_data["totalcharges"])
    # clustering_data["monthlycharges_log"] = np.log1p(
    #     clustering_data["monthlycharges"]
    # )
    # clustering_data["tenure_log"] = np.log1p(clustering_data["tenure"])
    clustering_data
    return (clustering_data,)


@app.cell(hide_code=True)
def _(mo):
    tenure_bin_slider = mo.ui.slider(1, 12, value=4, show_value=True)
    mo.md(f"Number of bins: {tenure_bin_slider}")
    return (tenure_bin_slider,)


@app.cell(hide_code=True)
def _(alt, clustering_data, tenure_bin_slider):
    _chart = (
        alt.Chart(clustering_data)
        .mark_point()
        .encode(
            x=alt.X(field="totalcharges", type="quantitative", bin=True),
            y=alt.Y(
                field="tenure",
                type="quantitative",
                bin={"step": tenure_bin_slider.value},
            ),
            color=alt.Color(field="churn", type="nominal"),
            tooltip=[
                alt.Tooltip(field="totalcharges", format=",.2f", bin=True),
                alt.Tooltip(
                    field="tenure",
                    format=",.0f",
                    bin={"step": tenure_bin_slider.value},
                ),
                alt.Tooltip(field="churn"),
            ],
        )
        .properties(height=290, width="container", config={"axis": {"grid": True}})
        .interactive()
    )

    _chart
    return


@app.cell(hide_code=True)
def _(alt, clustering_data):
    _chart = (
        alt.Chart(clustering_data)
        .mark_arc()
        .encode(
            color=alt.Color(field="customer_segment", type="nominal"),
            theta=alt.Theta(field="churn", type="nominal", aggregate="count"),
            tooltip=[
                alt.Tooltip(field="churn", aggregate="count"),
                alt.Tooltip(field="churn", aggregate="count"),
                alt.Tooltip(field="customer_segment"),
            ],
        )
        .properties(
            title="Proportion of segmentation",
        )
    )
    _chart
    return


@app.cell(hide_code=True)
def _(alt, clustering_data):
    _chart = (
        alt.Chart(clustering_data)
        .mark_arc()
        .encode(
            color=alt.Color(field="customer_segment", type="nominal"),
            theta=alt.Theta(field="churn", type="nominal"),
            tooltip=[
                alt.Tooltip(field="churn"),
                alt.Tooltip(field="churn"),
                alt.Tooltip(field="customer_segment"),
            ],
        )
        .properties(
            title="Proportion of churn based on segmentation",
        )
    )
    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kmeans
    """)
    return


@app.cell(hide_code=True)
def _(LabelEncoder, StandardScaler, clustering_data):
    _kmeans_scaler = StandardScaler()
    _kmeans_le = LabelEncoder()
    kmeans_data = clustering_data.copy()

    _num_cols = kmeans_data.select_dtypes(include=["float64", 'int64']).columns.to_list()
    kmeans_data[_num_cols] = _kmeans_scaler.fit_transform(kmeans_data[_num_cols])
    
    for _col in kmeans_data.select_dtypes(include=["object"]).columns.to_list():
        kmeans_data[_col] = _kmeans_le.fit_transform(kmeans_data[_col])
    kmeans_data
    return (kmeans_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Elbow Method & Silhouette Scores for KMeans
    """)
    return


@app.cell(hide_code=True)
def _(kmeans_data, mo):
    n_clusters_slider = mo.ui.slider(2, 10, value=4, show_value=True)
    kmeans_column_select = mo.ui.multiselect(
        options=kmeans_data.columns.tolist(),
        label="Select KMeans Columns:",
        value=kmeans_data.columns.tolist(),
    )
    mo.vstack(
        [
            mo.md(f"Number of clusters: {n_clusters_slider}"),
            mo.md(f"{kmeans_column_select}"),
        ]
    )
    return kmeans_column_select, n_clusters_slider


@app.cell(hide_code=True)
def _(KMeans, kmeans_column_select, kmeans_data, pd, silhouette_score):
    _results = {}
    _data = kmeans_data.copy()
    _X = _data[
        kmeans_column_select.value
        if kmeans_column_select.value != []
        else _data.columns.tolist()
    ]
    for _k in range(2, 13):
        _kmeans = KMeans(n_clusters=_k, random_state=42)
        _kmeans.fit(_X)
        _results[_k] = {}
        _results[_k]["sil"] = silhouette_score(_X, _kmeans.labels_)
        _results[_k]["inert"] = _kmeans.inertia_
    kmeans_results_df = (
        pd.DataFrame(_results)
        .T.reset_index()
        .rename(
            columns={
                "index": "n_clusters",
                "sil": "silhouette_score",
                "inert": "inertia",
            }
        )
    )
    return (kmeans_results_df,)


@app.cell(hide_code=True)
def _(alt, kmeans_results_df):
    _chart = (
        alt.Chart(kmeans_results_df)
        .mark_line()
        .encode(
            x=alt.X(field='n_clusters', type='quantitative'),
            y=alt.Y(field='inertia', type='quantitative'),
            tooltip=[
                alt.Tooltip(field='n_clusters'),
                alt.Tooltip(field='inertia', format=',.2f')
            ]
        )
        .properties(
            title='Inertia (TD2)',
            height=290,
            # width='container',
            config={
                'axis': {
                    'grid': True
                }
            }
        )
    )
    _chart
    return


@app.cell(hide_code=True)
def _(alt, kmeans_results_df):
    _chart = (
        alt.Chart(kmeans_results_df)
        .mark_line()
        .encode(
            x=alt.X(field='n_clusters', type='quantitative'),
            y=alt.Y(field='silhouette_score', type='quantitative'),
            tooltip=[
                alt.Tooltip(field='n_clusters', format=',.0f'),
                alt.Tooltip(field='silhouette_score', format=',.2f')
            ]
        )
        .properties(
            title='Silhouette Scores',
            height=290,
            width='container',
            config={
                'axis': {
                    'grid': True
                }
            }
        )
    )
    _chart
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Explore Clusters
    """)
    return


@app.cell(hide_code=True)
def _(KMeans, kmeans_column_select, kmeans_data, n_clusters_slider):
    kmeans = KMeans(n_clusters=n_clusters_slider.value, random_state=42)
    kmeans.fit(
        kmeans_data[kmeans_column_select.value]
    )
    kmeans_data["kmeans_cluster"] = kmeans.labels_
    return


@app.cell(hide_code=True)
def _(kmeans_data, mo):
    kmeans_scatter_chart_x = mo.ui.dropdown(
        options=kmeans_data.columns.tolist(),
        label="Select KMeans X-Axis Columns:",
        value='totalcharges')
    kmeans_scatter_chart_y = mo.ui.dropdown(
        options=kmeans_data.columns.tolist(),
        label="Select KMeans Y-Axis Columns:",
        value='tenure')
    kmeans_scatter_chart_color_scheme = mo.ui.dropdown(
        options=["accent", "dark2", "category10", "set1","set3", 'paired', 'tableau10'],
        label="Select KMeans Color Scheme:",
        value="accent")


    mo.hstack([kmeans_scatter_chart_x, kmeans_scatter_chart_y, kmeans_scatter_chart_color_scheme], gap=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    _chart = (
        alt.Chart(kmeans_data)
        .mark_point()
        .encode(
            x=alt.X(field=kmeans_scatter_chart_x.value, type="quantitative",scale=alt.Scale(padding=1)),
            y=alt.Y(field=kmeans_scatter_chart_y.value, type="quantitative",scale=alt.Scale(padding=1)),
            color=alt.Color(
                field="kmeans_cluster", type="quantitative", scale={"scheme": kmeans_scatter_chart_color_scheme.value}
            ),
            tooltip=[
                alt.Tooltip(field=kmeans_scatter_chart_x.value),
                alt.Tooltip(field=kmeans_scatter_chart_y.value),
                alt.Tooltip(field="kmeans_cluster", format=",.0f"),
                alt.Tooltip(field="churn", format=",.0f"),
            ],
        )
        .properties(height=290, width="container", config={"axis": {"grid": True}}, title="KMeans Clustering Scatter Plot")
    )
    kmeans_scatter_chart = mo.ui.altair_chart(_chart)
    kmeans_scatter_chart
    """)
    return


@app.cell(hide_code=True)
def _(kmeans_scatter_chart):
    kmeans_scatter_chart.value
    return


@app.cell
def _():
    return


@app.cell(column=6, hide_code=True)
def _(mo):
    mo.md(r"""
    # Logistic Regression Baseline
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One Hot Encoded Data
    """)
    return


@app.cell(hide_code=True)
def _(OneHotEncoder, StandardScaler, data, numerical_cols, pd):
    one_hot_encoder = OneHotEncoder(sparse_output=False)

    # Identify column types
    _binary_cols = [
        col
        for col in data.select_dtypes(exclude="float64").columns
        if data[col].nunique() == 2 and col != "seniorcitizen"
    ]
    _ternary_cols = [
        col
        for col in data.select_dtypes(exclude="float64").columns
        if data[col].nunique() == 3
    ]
    _categorical_cols = [
        col
        for col in data.select_dtypes(exclude="float64").columns
        if data[col].nunique() > 3
    ]

    _numerical_cols = data.select_dtypes(include="float64").columns.tolist()


    # Encode data of different types
    _encoded_binary = data[_binary_cols].apply(
        lambda x: x.map({"Yes": 1, "No": 0})
    )

    _encoded_ternary = pd.DataFrame(
        one_hot_encoder.fit_transform(data[_ternary_cols]),
        columns=one_hot_encoder.get_feature_names_out(_ternary_cols),
        index=data.index,
    )


    _encoded_categorical = pd.DataFrame(
        one_hot_encoder.fit_transform(data[_categorical_cols]),
        columns=one_hot_encoder.get_feature_names_out(_categorical_cols),
        index=data.index,
    )


    # Combine all encoded data
    ohe_encoded_data = pd.concat(
        [
            data[_numerical_cols],
            data[["seniorcitizen"]],
            _encoded_categorical,
            _encoded_binary,
            _encoded_ternary,
        ],
        axis=1,
    )

    ohe_encoded_data[numerical_cols] = StandardScaler().fit_transform(
        ohe_encoded_data[numerical_cols]
    )
    return (ohe_encoded_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Logistic Regression
    """)
    return


@app.cell(hide_code=True)
def _(
    LogisticRegression,
    accuracy_score,
    classification_report,
    ohe_encoded_data,
    train_test_split,
):
    _y = ohe_encoded_data["churn"]
    _X = ohe_encoded_data.drop(columns=["churn"])
    _X_train, _X_test, _y_train, _y_test = train_test_split(
        _X, _y, test_size=0.2, random_state=42, stratify=_y
    )

    ohe_log_reg = LogisticRegression(max_iter=10_000).fit(_X_train, _y_train)
    ohe_preds = ohe_log_reg.predict(_X_test)
    ohe_accuracy = accuracy_score(_y_test, ohe_preds)
    print(f"Accuracy: {ohe_accuracy}")
    print(classification_report(_y_test, ohe_preds))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Label Encoded Data
    """)
    return


@app.cell(hide_code=True)
def _(LabelEncoder, StandardScaler, data, numerical_cols):
    label_encoder = LabelEncoder()
    label_encoded_data = data.copy()
    for col in label_encoded_data.select_dtypes(
        include=["object"]
    ).columns.to_list():
        label_encoded_data[col] = label_encoder.fit_transform(
            label_encoded_data[col]
        )
    label_encoded_data[numerical_cols] = StandardScaler().fit_transform(
        label_encoded_data[numerical_cols]
    )
    return (label_encoded_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Logistic Regression
    """)
    return


@app.cell(hide_code=True)
def _(
    LogisticRegression,
    accuracy_score,
    classification_report,
    label_encoded_data,
    train_test_split,
):
    _y = label_encoded_data["churn"]
    _X = label_encoded_data.drop(columns=["churn"])
    _X_train, _X_test, _y_train, _y_test = train_test_split(
        _X, _y, test_size=0.2, random_state=42, stratify=_y
    )

    le_log_reg = LogisticRegression(max_iter=10_000, class_weight="balanced").fit(
        _X_train, _y_train
    )
    le_preds = le_log_reg.predict(_X_test)
    le_accuracy = accuracy_score(_y_test, le_preds)
    print(f"Accuracy: {le_accuracy}")
    print(classification_report(_y_test, le_preds))
    return


@app.cell
def _():
    return


@app.cell(column=7, hide_code=True)
def _(mo):
    mo.md(r"""
    # SMOTE

    Hypothesis: Since the data is imbalanced (churn = 26.6%), applying SMOTE and other oversampling techniques should improve accuracy.

    Results: SMOTE does little to no improvement to accuracy. 26.6% is not a great enough imbalance to get greatly improved results.
    """)
    return


@app.cell(hide_code=True)
def _():
    import imblearn
    from collections import Counter

    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTEENN
    return Counter, SMOTE


@app.cell
def _(
    Counter,
    LogisticRegression,
    SMOTE,
    accuracy_score,
    label_encoded_data,
    np,
    train_test_split,
):
    _y = label_encoded_data["churn"]
    _X = label_encoded_data.drop(columns=["churn"])

    results = []

    for _ in range(10):
        _X_train, _X_test, _y_train, _y_test = train_test_split(
            _X, _y, test_size=0.2, stratify=_y
        )
        _smote = SMOTE()
        _X_resampled_train, _y_resampled_train = _smote.fit_resample(
            _X_train, _y_train
        )

        smote_le_log_reg = LogisticRegression(
            max_iter=10_000, class_weight="balanced"
        ).fit(_X_resampled_train, _y_resampled_train)

        smote_le_preds = smote_le_log_reg.predict(_X_test)
        smote_le_accuracy = accuracy_score(_y_test, smote_le_preds)
        results.append(smote_le_accuracy)
    print(Counter(_y).values())
    print(Counter(_y_resampled_train).values())

    print(
        f"Highest accuracy: {max(results)} \nAverage accuracy: {np.mean(results)}"
    )
    return


@app.cell(column=8)
def _():
    return


if __name__ == "__main__":
    app.run()
