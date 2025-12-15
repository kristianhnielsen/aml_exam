import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from data.data import load_data
    import seaborn as sns
    import pprint as pp
    return load_data, mo, plt, sns


@app.cell
def _(mo):
    mo.md(r"""
    # Todo:
    - Analyze dataset structure: distributions, outliers, skewness
    - Examine correlations among numerical features
    - Explore categorical feature cardinality
    - Study churn vs. non-churn population imbalance
    - Identify missing values and propose imputation strategies
    """)
    return


@app.cell
def _(load_data):
    df = load_data()
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.isnull().sum()
    return


@app.cell
def _(df, plt):
    # Analyze dataset structure: distributions, outliers, skewness
    print("Dataset Shape:", df.shape)
    print("\nDataset Info:")
    df.info()

    print("\n" + "="*50)
    print("Statistical Summary:")
    print("="*50)
    print(df.describe())

    print("\n" + "="*50)
    print("Skewness of Numerical Features:")
    print("="*50)
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    skewness = df[numerical_features].skew().sort_values(ascending=False)
    print(skewness)

    # Visualize distributions with box plots to identify outliers
    fig, axes = plt.subplots(len(numerical_features)//3 + 1, 3, figsize=(15, len(numerical_features)*2))
    axes = axes.flatten()

    for idx, col in enumerate(numerical_features):
        axes[idx].boxplot(df[col].dropna())
        axes[idx].set_title(f'{col}\nSkewness: {df[col].skew():.2f}')
        axes[idx].set_ylabel('Values')

    # Remove empty subplots
    for idx in range(len(numerical_features), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()
    return (col,)


@app.cell
def _(df, plt, sns):
    # Calculate correlation matrix for numerical features
    correlation_matrix = df.select_dtypes(include=['float64', 'int64']).corr()

    # Visualize correlations using a heatmap

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(col, df):
    # Explore categorical feature cardinality
    print("="*50)
    print("Categorical Feature Cardinality:")
    print("="*50)

    categorical_features = df.select_dtypes(include=['object', 'category']).columns

    for column in categorical_features:
        unique_count = df[col].nunique()
        print(f"\n{column}:")
        print(f"  Unique values: {unique_count}")
        print(f"  Value counts:\n{df[col].value_counts()}")
        print(f"  Percentage distribution:\n{df[col].value_counts(normalize=True) * 100}")
    return


@app.cell
def _(df, plt):
    # Study churn vs. non-churn population imbalance
    print("="*50)
    print("Churn vs. Non-Churn Population Analysis:")
    print("="*50)

    # Assuming the target variable is named 'Churn' or similar
    # Adjust the column name if different
    churn_col = 'Churn'  # Update this if your target column has a different name

    if churn_col in df.columns:
        churn_counts = df[churn_col].value_counts()
        churn_percentages = df[churn_col].value_counts(normalize=True) * 100

        print(f"\nChurn Distribution:")
        print(churn_counts)
        print(f"\nChurn Percentage Distribution:")
        print(churn_percentages)

        # Calculate imbalance ratio
        imbalance_ratio = churn_counts.max() / churn_counts.min()
        print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")

        # Visualize the imbalance
        fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))

        # Count plot
        churn_counts.plot(kind='bar', ax=axes1[0], color=['#2ecc71', '#e74c3c'])
        axes1[0].set_title('Churn Distribution (Counts)')
        axes1[0].set_xlabel('Churn Status')
        axes1[0].set_ylabel('Count')
        axes1[0].set_xticklabels(axes1[0].get_xticklabels(), rotation=0)

        # Pie chart
        axes1[1].pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', 
                    colors=['#2ecc71', '#e74c3c'], startangle=90)
        axes1[1].set_title('Churn Distribution (Percentage)')

        plt.tight_layout()
        plt.show()
    else:
        print(f"Column '{churn_col}' not found. Available columns: {df.columns.tolist()}")
    return


if __name__ == "__main__":
    app.run()
