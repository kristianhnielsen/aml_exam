import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from data.data import load_data
    import seaborn as sns
    import pandas as pd
    return load_data, mo, pd, plt, sns


@app.cell(hide_code=True)
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
def _(df, pd):
    _col = 'TotalCharges'

    if _col in df.columns:
        print(f"Column '{_col}' found. Current dtype: {df[_col].dtype}")
        total_count = len(df)
        before_na = df[_col].isna().sum()
        # attempt conversion
        converted = pd.to_numeric(df[_col], errors='coerce')
        after_na = converted.isna().sum()
        coerced = after_na - before_na
        non_null_before = total_count - before_na
        non_null_after = total_count - after_na

        print(f"Total rows: {total_count}")
        print(f"Non-null before conversion: {non_null_before}")
        print(f"Non-null after conversion:  {non_null_after}")
        print(f"Values coerced to NaN during conversion: {coerced} ({coerced/total_count*100:.2f}%)")

        if coerced > 0:
            print("\nExamples of values that could not be converted (first 20):")
            failed_examples = df.loc[converted.isna() & df[_col].notna(), _col].head(20)
            print(failed_examples.to_string(index=True))

        # Assign converted series back to the dataframe (in-place change)
        df[_col] = converted

        print(f"\nAfter casting, dtype: {df[_col].dtype}")
        print("\nSummary statistics for the converted column:")
        print(df[_col].describe())
    else:
        print(f"Column '{_col}' not found. Available columns: {df.columns.tolist()}")
    return


@app.cell
def _(df):
    df.isnull().sum()
    return


@app.cell
def _(df):
    # Display rows with missing data in TotalCharges
    df[df['TotalCharges'].isna()]
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

    for idx, dist_col in enumerate(numerical_features):
        axes[idx].boxplot(df[dist_col].dropna())
        axes[idx].set_title(f'{dist_col}\nSkewness: {df[dist_col].skew():.2f}')
        axes[idx].set_ylabel('Values')

    # Remove empty subplots
    for idx in range(len(numerical_features), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()
    return (numerical_features,)


@app.cell
def _(df, plt, sns):
    # Calculate correlation matrix for numerical features
    correlation_matrix = df.select_dtypes(include=['float64', 'int64']).drop('SeniorCitizen', axis=1).corr()

    # Visualize correlations using a heatmap

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df):
    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    print("\n" + "="*50)
    print("Categorical Features Analysis:")
    print("="*50)
    for cat_col in categorical_features:
        print(f"\nAnalyzing Categorical Feature: {cat_col}")
        value_counts = df[cat_col].value_counts(dropna=False)
        print(value_counts)
    return (categorical_features,)


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
    return (churn_col,)


@app.cell
def _(churn_col, df, numerical_features, plt, sns):
    # Visualize Numerical Features vs Churn
    for num_churn_col in numerical_features:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=churn_col, y=num_churn_col, data=df)
        plt.title(f'{num_churn_col} distribution by {churn_col}')
        plt.show()
    return


@app.cell
def _(categorical_features, churn_col, df, plt, sns):
    # Visualize Categorical Features vs Churn
    for col in categorical_features.drop('customerID'):
        if col != churn_col:
            plt.figure(figsize=(10, 5))
            sns.countplot(x=col, hue=churn_col, data=df)
            plt.title(f'{col} distribution by {churn_col}')
            plt.xticks(rotation=45)
            plt.legend(title=churn_col, loc='upper right')
            plt.tight_layout()
            plt.show()
    return


@app.cell
def _(df):
    print(f"Duplicate rows: {df.duplicated().sum()}")
    return


@app.cell
def _(df, plt):
    # Sanity Check: TotalCharges vs (Tenure * MonthlyCharges)
    # Note: Might not be exact due to price changes over time, but should be linear.
    plt.figure(figsize=(8, 6))
    plt.scatter(df['tenure'] * df['MonthlyCharges'], df['TotalCharges'], alpha=0.5)
    plt.xlabel('Calculated (Tenure * MonthlyCharges)')
    plt.ylabel('Actual TotalCharges')
    plt.title('Sanity Check: TotalCharges Logic')
    plt.plot([0, df['TotalCharges'].max()], [0, df['TotalCharges'].max()], 'r--') # Identity line
    plt.show()
    return


if __name__ == "__main__":
    app.run()
