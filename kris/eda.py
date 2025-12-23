import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import marimo as mo
    return mo, np, pd, plt, sns


@app.cell
def _(pd):
    data = pd.read_csv("./data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    data.head()
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Manipulation
    """)
    return


@app.cell
def _(data):
    data.columns = data.columns.str.lower()
    data.drop(columns=["customerid"], inplace=True)

    # there were values with whitespace, which didn't register as NaN-like
    print("Before:\t", data.shape)
    data.drop(labels=data[data["totalcharges"] == " "].index, inplace=True)
    print("After:\t", data.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Total Charges vs Monthly Charges
    I assumed that Total Charges was an interaction between Monthly Charges X Tenure.
    However the difference between the assumed interaction and the Total Charges value are too far apart to assume that the assumption holds.
    """)
    return


@app.cell
def _(mo):
    bin_slider = mo.ui.slider(1, 50, value=30, show_value=True)
    bin_slider
    return (bin_slider,)


@app.cell
def _(bin_slider, mo):
    mo.md(f"""
    Bins: {bin_slider.value}
    """)
    return


@app.cell
def _(bin_slider, data, plt, sns):
    data["pseudo_total"] = data["tenure"] * data["monthlycharges"]
    data["total_diff"] = data["totalcharges"].astype(float) - data[
        "pseudo_total"
    ].astype(float)
    data[["tenure", "monthlycharges", "totalcharges", "pseudo_total", "total_diff"]]
    sns.histplot(data["total_diff"], bins=bin_slider.value)
    plt.show()
    data.drop(columns=["total_diff", "pseudo_total"], inplace=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Overview
    """)
    return


@app.cell
def _(data):
    data.info()
    return


@app.cell
def _(data):
    data.describe()
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
def _(mo):
    mo.md(r"""
    ## Hist plots of all columns
    """)
    return


@app.cell
def _(data, plt, sns):
    for _col in data.columns:
        sns.histplot(x=_col, data=data, legend=True, bins=30)
        plt.xticks(rotation=45)
        plt.show()
        print()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hist plots comparing Churn vs Not Churn
    """)
    return


@app.cell
def _(data, plt, sns):
    for _col in data.columns:
        churned = data[data["churn"] == "Yes"]
        not_churned = data[data["churn"] == "No"]
        sns.histplot(x=_col, data=data, hue="churn", bins=30)
        plt.xticks(rotation=45)
        plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Correlation Heatmap
    """)
    return


@app.cell
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tenure

    Months being a customer

    # Services

    All the services have 3 values: Yes, No, No internet service

    Can we change "No internet service" to "No" with minimal effect?

    # Charges

    There must be multicolinearity between Total Charges and Tenure \* Monthly Charges
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
