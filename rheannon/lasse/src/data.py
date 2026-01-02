import pandas as pd


def load_data(
    file_path: str = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
) -> pd.DataFrame:
    # Define the schema
    schema = {
        "customerID": "string",
        "gender": "category",
        "SeniorCitizen": "bool",
        "Partner": "category",
        "Dependents": "category",
        "tenure": "int32",
        "PhoneService": "category",
        "MultipleLines": "category",
        "InternetService": "category",
        "OnlineSecurity": "category",
        "OnlineBackup": "category",
        "DeviceProtection": "category",
        "TechSupport": "category",
        "StreamingTV": "category",
        "StreamingMovies": "category",
        "Contract": "category",
        "PaperlessBilling": "category",
        "PaymentMethod": "category",
        "MonthlyCharges": "float32",
        # TotalCharges contains empty strings which need handling before conversion
        # so we read it as object/string first
        "TotalCharges": "object",
        "Churn": "category",
    }

    data = pd.read_csv(
        file_path, dtype=schema, converters={"Churn": lambda x: 1 if x == "Yes" else 0}
    )
    return data
