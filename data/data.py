import pandas as pd


def load_data(file_path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    data = pd.read_csv(file_path)
    return data
