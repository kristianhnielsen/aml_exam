import pandas as pd


def load_data(
    file_path: str = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    return data
