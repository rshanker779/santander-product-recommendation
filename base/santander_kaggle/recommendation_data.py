"""
Module to provide functions that access data
"""

import pandas as pd
import os


class Data:
    data_directory = os.path.join("..", "data")
    results_directory = os.path.join(os.path.join(os.path.dirname(__file__)), "results")
    full_train_data = None


def categorical_cols() -> list:
    return [
        "employee_index",
        "country",
        "sex",
        "is_new_customer",
        "customer_seniority",
        "is_primary",
        "customer_type_beginning_of_the_month",
        "customer_relation_beginning_of_the_month",
        "residence_index",
        "foreigner_index",
        "spouse_index",
        "entry_channel",
        "deceased_index",
        "province_code",
        "address",
        "activity_index",
        "segmentation",
    ]


def continuous_cols() -> list:
    return ["age", "gross_household_income"]


def date_cols() -> list:
    return ["snapshot_date", "initial_signup_date", "last_date_as_primary"]


def get_train_data() -> pd.DataFrame:
    if Data.full_train_data is None:
        train_data = pd.read_csv(os.path.join(Data.data_directory, "train_ver2.csv"))
        Data.full_train_data = train_data
    else:
        train_data = Data.full_train_data
    return train_data


def get_customer_dataframe() -> pd.DataFrame:
    customer_csv = os.path.join(Data.data_directory, "train_subsample.csv")
    if not os.path.exists(customer_csv):
        customer_data = pd.read_csv(
            os.path.join(Data.data_directory, "train_ver2.csv"),
            nrows=1000,
            low_memory=False,
        )
        customers = customer_data["ncodpers"].unique()
        train_data = get_train_data()
        small_cust_data = train_data[train_data["ncodpers"].isin(customers)]
        small_cust_data.to_csv(customer_csv)
        return small_cust_data
    return pd.read_csv(customer_csv, index_col=[0])


def get_date_limited_train_data(date_list=None, customers=None) -> pd.DataFrame:
    if date_list is None:
        date_list = ["2015-04-28", "2015-05-28", "2015-06-28"]
    train_data = get_train_data()
    train_data = train_data[train_data["fecha_dato"].isin(date_list)]
    if customers is None:
        return train_data
    else:
        customers = list(train_data["ncodpers"].unique())[:customers]
        train_data = train_data[train_data["ncodpers"].isin(customers)]
        return train_data


def get_test_data() -> pd.DataFrame:
    return pd.read_csv(os.path.join(Data.data_directory, "test_ver2.csv"))


def get_prediction_data() -> pd.DataFrame:
    return pd.read_csv(os.path.join(Data.results_directory, "predictions.csv"))
