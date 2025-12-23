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
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import marimo as mo
    import altair as alt
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.decomposition import PCA
    return OneHotEncoder, PCA, StandardScaler, alt, mo, np, pd, plt, sns


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
    # Correlation Heatmap
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
    mo.vstack(
        [
            mo.md("**No columns selected for PCA. Please select at least one column to display PCA plots.**") if pca_select_col.value == [] else "",
            pca_select_col,
            scree.interactive(), 
            (cumulative + rule).interactive(),
        ]
    )
    return


@app.cell(column=5)
def _():
    return


if __name__ == "__main__":
    app.run()
