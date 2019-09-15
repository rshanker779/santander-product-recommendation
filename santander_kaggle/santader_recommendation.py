import io
import os
from typing import Tuple, Callable

import lightgbm as lgb
import pandas as pd
from rshanker779_common.logger import get_logger
from rshanker779_common.utilities import time_it

logger = get_logger(__name__)


#TODO
#Make sure we make at least one prediction for each customer
#Fix order of predictions
##One hot encode/map appropriate variables

class Config:
    """
    Class to hold any global config
    """

    clean_raw_data = True
    load_clean_raw_data = not clean_raw_data
    process_raw_data = True
    # For memory reasons, only keep subset of data
    limit_dates = True
    date_limit = "2015-09-01"
    train_test_date_split = "2016-04-01"
    data_directory = os.path.join("..", "data")


def translate_columns(df: pd.DataFrame) -> pd.DataFrame:
    translation_dict = get_translation_dict()
    df = df.rename(columns=translation_dict)
    return df


def log_dataframe_information(df: pd.DataFrame) -> None:
    logger.info("Shape %s", df.shape)
    # logger.info("Dtypes \n %s", df.dtypes)
    buffer = io.StringIO()
    df.info(buf=buffer)
    logger.info(buffer.getvalue())


def get_translation_dict() -> dict:
    translation_dict = {
        "fecha_dato": "snapshot_date",
        "ncodpers": "customer_id",
        "ind_empleado": "employee_index",
        # A active,B exemployed,F filial,N notemployee,P pasive
        "pais_residencia": "country",
        "sexo": "sex",
        "age": "age",
        "fecha_alta": "initial_signup_date",
        "ind_nuevo": "is_new_customer",
        # 1 if the customer registered in the last 6 months.
        "antiguedad": "customer_seniority",  # (in months)
        "indrel": "is_primary",
        # 1(First/Primary)',99(Primary customer during the month but not at the end of the month)
        "ult_fec_cli_1t": "last_date_as_primary",
        # (if he isn't at the end of the month)
        "indrel_1mes": "customer_type_beginning_of_the_month",
        # ',1(First/Primary customer)',2(co-owner)',P(Potential)',3(former primary)',4(former co-owner)
        "tiprel_1mes": "customer_relation_beginning_of_the_month",
        # ',A(active)',I(inactive)',P(formercustomer)',R(Potential)
        "indresi": "residence_index",
        # (S(Yes) or N(No) if the residence country is the same than the bank country)
        "indext": "foreigner_index",
        # (S(Yes)orN(No)if the customer's birth country is different than the bank country)
        "conyuemp": "spouse_index",  # 1 if the customer is spouse of an employee
        "canal_entrada": "entry_channel",  # channel used by the customer to join
        "indfall": "deceased_index",  # Deceased index.N/S
        "tipodom": "address",  # primary address
        "cod_prov": "province_code",  # (customer's address)
        "nomprov": "province_name",
        "ind_actividad_cliente": "activity_index",
        # (1,active customer; 0,inactive customer)
        "renta": "gross_household_income",
        "segmento": "segmentation",
        # '01-VIP', 02 - Individuals 03 - college graduated
        "ind_ahor_fin_ult1": "has_saving_account",
        "ind_aval_fin_ult1": "has_guarantees",
        "ind_cco_fin_ult1": "has_current_account",
        "ind_cder_fin_ult1": "has_derivada_account",
        "ind_cno_fin_ult1": "has_payroll_account",
        "ind_ctju_fin_ult1": "has_junior_account",
        "ind_ctma_fin_ult1": "has_mas_particular_account",
        "ind_ctop_fin_ult1": "has_particular_account",
        "ind_ctpp_fin_ult1": "has_particular_plus_account",
        "ind_deco_fin_ult1": "has_short_term_deposits",
        "ind_deme_fin_ult1": "has_medium_term_deposits",
        "ind_dela_fin_ult1": "has_long_term_deposits",
        "ind_ecue_fin_ult1": "has_e_account",
        "ind_fond_fin_ult1": "has_funds",
        "ind_hip_fin_ult1": "has_mortgage",
        "ind_plan_fin_ult1": "has_plan_pensions",
        "ind_pres_fin_ult1": "has_loans",
        "ind_reca_fin_ult1": "has_taxes",
        "ind_tjcr_fin_ult1": "has_credit_card",
        "ind_valo_fin_ult1": "has_securities",
        "ind_viv_fin_ult1": "has_home_account",
        "ind_nomina_ult1": "has_payroll",
        "ind_nom_pens_ult1": "has_nom_pensions",
        "ind_recibo_ult1": "has_direct_debit",
    }
    return translation_dict


def get_customer_dataframe() -> pd.DataFrame:
    customer_data = pd.read_csv(
        os.path.join(Config.data_directory, "train_ver2.csv"),
        nrows=1000,
        low_memory=False,
    )
    customer_data = translate_columns(customer_data)
    customers = customer_data["customer_id"].unique()
    train_data = get_train_data()
    small_cust_data = train_data[train_data["customer_id"].isin(customers)]
    return small_cust_data


def get_train_data() -> pd.DataFrame:
    return pd.read_csv(os.path.join(Config.data_directory, "train_ver2.csv"))


def get_test_data() -> pd.DataFrame:
    return pd.read_csv(os.path.join(Config.data_directory, "test_ver2.csv"))


class DataMapper:
    """
    Class to hold functions that describe or transform data that will be
    resused regularly
    """

    test_column_check = lambda i: "prev" not in i and "has" in i
    pred_column_check = lambda i: "will" in i


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Clean date formats
    for date_col in ("snapshot_date", "initial_signup_date"):
        df[date_col] = pd.to_datetime(df[date_col])

    if Config.limit_dates:
        df = df[(df["snapshot_date"] > Config.date_limit) | ~df["is_train"]]
        logger.info("After date filtering shape is %s", df.shape)
    # Clean missing data types #TODO
    df = df.replace("NA", pd.np.nan)
    df = df.replace(" NA", pd.np.nan)

    # Change S,N boolean columns, and 1, 99 columns
    df["sex"] = df["sex"].replace(to_replace={"H": 0, "V": 1})
    df["is_primary"] = df["is_primary"].replace(to_replace={99: 0}).astype(float)
    bool_dict = {"S": 1, "N": 0}
    for i in ("residence_index", "foreigner_index", "spouse_index", "deceased_index"):
        # Check for when subsampling, as may have all null col
        if "S" in df[i].unique() or "N" in df[i].unique():
            df[i] = df[i].replace(to_replace=bool_dict)
            df[i] = df[i].astype(float)
            logger.info((i, df[i].unique()))
    df["age"] = df["age"].astype(float)
    # Median imputation
    # TODO
    # need to decide what to do
    suspect_rows = df[df["foreigner_index"] + df["residence_index"] == 0]
    logger.info("Have %s suspect rows", len(suspect_rows))
    return df


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(df["snapshot_date"].unique())
    df = df.sort_values(by="snapshot_date")
    cust_df = df.groupby("customer_id")
    for col in df.columns:
        # TODO change to num_months
        for i in range(1, 2):
            df["prev_{}_{}".format(i, col)] = cust_df[col].shift(i)
    # Ignore rows with no product bought
    rows_with_acquisition = get_product_deltas(df)
    rows_with_acquisition = rows_with_acquisition.any(axis=1)
    rows_to_keep = ~df["is_train"] | rows_with_acquisition
    df = df[rows_to_keep]
    # for col in df.columns:
    #     # TODO change to num_months
    #     for i in range(2,4):
    #         df["prev_{}_{}".format(i, col)] = cust_df[col].shift(i)

    return df


def get_product_deltas(df: pd.DataFrame) -> pd.DataFrame:
    prev_product_cols = [i for i in df.columns if "has" in i and "1" in i]
    col_name_map = {i: i.replace("prev_1_", "") for i in prev_product_cols}
    rows_with_acquisition = pd.DataFrame()
    logger.info(col_name_map)
    for i, v in col_name_map.items():
        bool_col = df[v] > df[i]
        rows_with_acquisition = pd.concat([rows_with_acquisition, bool_col], axis=1)
        rows_with_acquisition.columns = list(rows_with_acquisition.columns)[:-1] + [v]
    # Increase 0 to 1 represents an acquisition, 1 to 0 a loss. So need to check
    # greater than rather than  not equals
    # rows_with_acquisition = df[product_cols] > prev_acquisition_df
    # rows_with_acquisition = df[product_cols] > df[prev_product_cols]

    return rows_with_acquisition


def build_models(df: pd.DataFrame) -> dict:
    # train_set
    model_map = {}
    # Ignore first row, as don't know if there is an acquistion
    train_df = df[
        (df["snapshot_date"] < Config.train_test_date_split)
        & (df["snapshot_date"] > "2015-01-28")
    ]
    test_df = df[(df["snapshot_date"] >= Config.train_test_date_split)]
    logger.info("Train has %s snapshots", len(train_df["snapshot_date"].unique()))
    logger.info("Test has %s snapshots", len(test_df["snapshot_date"].unique()))
    train_x, train_y = get_x_y_variables(train_df)
    test_x, test_y = get_x_y_variables(test_df)
    log_dataframe_information(train_x)
    log_dataframe_information(test_x)
    log_dataframe_information(train_y)
    log_dataframe_information(test_y)

    # TODO fix modelling needs to predict delta, not actual value
    params = {"objective": "binary", "metric": "binary_logloss"}
    for i in test_y.columns:
        logger.info("Training model for feature %s", i)
        train_dataset = lgb.Dataset(train_x, label=train_y.loc[:, i])
        test_dataset = lgb.Dataset(test_x, label=test_y.loc[:, i])
        model = lgb.train(
            train_set=train_dataset,
            params=params,
            valid_sets=test_dataset,
            early_stopping_rounds=10,
        )
        model_map[i] = model
    return model_map


def get_x_y_variables(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # TODO one hot encode some of these features
    train_column_check = lambda i: not DataMapper.test_column_check(i) and not any(
        [
            j in i
            for j in (
                "snapshot_date",
                "employee_index",
                "country",
                "sex",
                "age",
                "initial_signup_date",
                "customer_seniority",
                "last_date_as_primary",
                "customer_relation_beginning_of_the_month",
                "residence_index",
                "entry_channel",
                "province_name",
                "segmentation",
                "gross_household_income",
                "is_train",
                "customer_type_beginning_of_the_month",
            )
        ]
    )
    train_x = train_df[[i for i in train_df.columns if train_column_check(i)]]
    # train_y = train_df[[i for i in train_df.columns if test_column_check(i)]]
    train_y = get_product_deltas(train_df)
    return train_x, train_y


def get_prediction(df: pd.DataFrame, model_map: dict) -> pd.DataFrame:
    # TODO need to order items to respect MAP
    prediction_rows = df
    prediction_x, prediction_y = get_x_y_variables(prediction_rows)
    logger.info("Making predictions for %s rows", len(prediction_x))
    for i, v in model_map.items():
        df.loc[
            df["snapshot_date"] > "2016-05-01", "will_have_{}".format(i)
        ] = v.predict(prediction_x)
        # prediction_rows["will_have_{}".format(i)] = v.predict(prediction_x)
    return prediction_rows


def get_formatted_prediction(prediction_rows: pd.DataFrame) -> None:
    ind_prod_prediction = prediction_rows[
        [i for i in prediction_rows.columns if DataMapper.pred_column_check(i)]
    ]
    bought_product = ind_prod_prediction > 0.5
    translation_dict = get_translation_dict()
    inverse_translation_dict = {v: i for i, v in translation_dict.items()}
    prediction_cols = bought_product.apply(
        lambda x: ",".join(
            [
                inverse_translation_dict[i.replace("will_have_", "")]
                for i, v in x.iteritems()
                if v
            ]
        ),
        axis=1,
    )
    pred_df = pd.concat(
        [pd.DataFrame(prediction_cols), prediction_rows["customer_id"]], axis=1
    )
    # pred_df = pd.DataFrame(prediction_cols.reset_index())
    pred_df.columns = ["added_products", "ncodpers"]
    pred_df = pred_df[list(pred_df.columns)[::-1]]

    # pred_df['ncodpers'] = prediction_rows['customer_id']
    logger.info("Saving predictions to csv")
    pred_df.to_csv("my_pred.csv", index=False)


def apply_pipeline(df: pd.DataFrame, *args: Callable) -> pd.DataFrame:
    out = df
    for callable in args:
        logger.info("Applying callable %s to data", callable)
        out = callable(out)
    return out


@time_it
def main():
    train_data = get_train_data()
    test_data = get_test_data()
    train_data["is_train"] = True
    test_data["is_train"] = False
    pipeline = (translate_columns, clean_data, process_df)
    train_data = apply_pipeline(train_data, *pipeline)
    log_dataframe_information(train_data)
    test_data = apply_pipeline(test_data, *pipeline)
    log_dataframe_information(test_data)
    joint_data = train_data.append(test_data)

    logger.info("Have %s customers ", len(joint_data["customer_id"].unique()))
    logger.info(
        "Have %s test customers ",
        len(joint_data.loc[joint_data["is_train"] == 0, "customer_id"].unique()),
    )
    models = build_models(train_data)
    prediction_rows = get_prediction(test_data, models)
    get_formatted_prediction(prediction_rows)


if __name__ == "__main__":
    main()
