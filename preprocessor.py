from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from data.data import load_data
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def preprocessor():
    df = load_data()

    # remove TotalCharges empty strings
    df = df[df["TotalCharges"] != " "]
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], downcast="float")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # create 2 pipelines for categorical and numrical data
    # perform scaling and one hot encoding
    numerical_cols = X.select_dtypes(include=["int32", "float32"]).columns
    categorical_cols = X.select_dtypes(include=["object", "bool", "category"]).columns

    numerical_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[
            # sparse_output=False is usually needed to return a dense DataFrame
            # compatible with standard pandas operations
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        verbose_feature_names_out=False,  # Optional: keeps column names cleaner
    )

    # This tells sklearn to return pandas DataFrames instead of numpy arrays
    preprocessor.set_output(transform="pandas")

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test
