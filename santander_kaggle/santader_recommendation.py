import io
import os
from typing import Tuple, Callable
import numpy as np
import lightgbm as lgb
import pandas as pd
from rshanker779_common.logger import get_logger, get_default_formatter
from rshanker779_common.utilities import time_it
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
import datetime
import re
import logging
from collections import defaultdict
from functools import partial

logger = get_logger(__name__)
file_handler = logging.FileHandler("{}.log".format(__file__))
file_handler.setFormatter(get_default_formatter())
logger.addHandler(file_handler)

# TODO
# More look back and delta features
# Make pred nicer


class Config:
    """
    Class to hold any global config
    """

    # For memory reasons, only keep subset of data
    limit_dates = False
    date_limit = "2016-03-01"
    train_test_date_split = "2015-10-01"
    data_directory = os.path.join("..", "data")
    look_back_level = 2
    impute_missing_mode_values = False
    impute_missing_median_values = True
    one_hot_encode = False
    label_encode = True
    product_recommendation_threshold = 0.5


class Transfomers:
    """Class to hold transformer data"""

    label_encoder_map = defaultdict(lambda: LabelEncoder())
    ordinal_transformer = None
    ordinal_cols = None
    label_encoder = None
    label_encoder_cols = None
    mode_imputer = None
    mode_cols = None
    median_imputer = None
    median_cols = None
    one_hot_encoder = None
    one_hot_cols = None
    final_cols = None


class Data:
    full_train_data = None


def get_encoder(existing_encoder, default_encoder):
    if existing_encoder is None:
        logger.info(
            "No existing encoder, making one of name %s",
            default_encoder.__class__.__name__,
        )
        encoder = default_encoder
        existing_encoder = encoder
    return existing_encoder


def is_encoder_trained(encoder):
    return hasattr(encoder, "statistics_") or hasattr(encoder, "categories_")


def translate_columns(df: pd.DataFrame) -> pd.DataFrame:
    translation_dict = get_translation_dict()
    df = df.rename(columns=translation_dict)
    return df


def log_dataframe_information(df: pd.DataFrame) -> None:
    logger.info("Shape %s", df.shape)
    # logger.info("Dtypes \n %s", df.dtypes)
    buffer = io.StringIO()
    df.info(buf=buffer, verbose=True)
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
    customer_csv = os.path.join(Config.data_directory, "train_subsample.csv")
    if not os.path.exists(customer_csv):
        customer_data = pd.read_csv(
            os.path.join(Config.data_directory, "train_ver2.csv"),
            nrows=1000,
            low_memory=False,
        )
        # customer_data = translate_columns(customer_data)
        customers = customer_data["ncodpers"].unique()
        train_data = get_train_data()
        # train_data = translate_columns(train_data)
        small_cust_data = train_data[train_data["ncodpers"].isin(customers)]
        small_cust_data.to_csv(customer_csv)
        return small_cust_data
    return pd.read_csv(customer_csv, index_col=[0])


def get_train_data() -> pd.DataFrame:
    if Data.full_train_data is None:
        train_data = pd.read_csv(os.path.join(Config.data_directory, "train_ver2.csv"))
        Data.full_train_data = train_data
    else:
        train_data = Data.full_train_data
    return train_data


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
    # train_data = train_data[]


def get_test_data() -> pd.DataFrame:
    return pd.read_csv(os.path.join(Config.data_directory, "test_ver2.csv"))


def is_test_column(i: str) -> bool:
    return "prev" not in i and "has" in i


def is_pred_column(i: str) -> bool:
    return "will" in i


def get_time_col(default, format, x):
    return x.strftime(format) if not pd.isnull(x) else default


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # for date_col in ("snapshot_date", "initial_signup_date"):
    #     df[date_col] = pd.to_datetime(df[date_col])

    # Clean date formats
    for date_col in ["snapshot_date", "initial_signup_date", "last_date_as_primary"]:
        df[date_col] = pd.to_datetime(df[date_col])
        df["month_{}".format(date_col)] = (
            df[date_col].apply(partial(get_time_col, -1, "%m")).astype(int)
        )
        df["year_{}".format(date_col)] = (
            df[date_col].apply(partial(get_time_col, -1, "%y")).astype(int)
        )
        df["code_{}".format(date_col)] = (
            df[date_col].apply(partial(get_time_col, -1, "%y%m")).astype(int)
        )
    if Config.limit_dates:
        df = df[(df["snapshot_date"] > Config.date_limit) | ~df["is_train"]]
        logger.info("After date filtering shape is %s", df.shape)
    df = df.replace("\s*NA", pd.np.nan, regex=True)
    na_cols = [i for i in df.columns if i != "is_train" and not is_pred_column(i)]
    df[na_cols] = df[na_cols].fillna(-1)

    df["customer_seniority"] = df["customer_seniority"].astype(float)
    df["customer_type_beginning_of_the_month"] = (
        df["customer_type_beginning_of_the_month"].replace("P", 5).astype(float)
    )

    # Imputation for categorical columns
    drop_cols = ["province_name"]  # Already take code data, so redundant
    mode_cols = [
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
    median_cols = ["age", "gross_household_income"]
    all_one_hot_cols = [
        # "snapshot_date", ##TODO standard one hot encode doesn't work
        "employee_index",
        "country",
        "sex",
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
        "segmentation",
    ]
    mode_cols = [i for i in mode_cols if len(df[i].unique()) > 1]
    one_hot_cols = [
        i
        for i in all_one_hot_cols
        if len(df[i].unique()) > 1 or not any(pd.isnull(df[i].unique()))
    ]

    mode_cols = (
        Transfomers.mode_cols if Transfomers.mode_cols is not None else mode_cols
    )
    one_hot_cols = (
        Transfomers.one_hot_cols
        if Transfomers.one_hot_cols is not None
        else one_hot_cols
    )
    Transfomers.one_hot_cols = one_hot_cols
    Transfomers.mode_cols = mode_cols
    logger.info("Mode cols %s", mode_cols)
    logger.info("One hot  cols %s", one_hot_cols)
    # label_encode_cols = [i for i in df.columns if df[i].dtype=='object' ]
    # Transfomers.label_encoder = get_encoder(Transfomers.label_encoder, LabelEncoder())
    # label_enc =Transfomers.label_encoder
    if Config.label_encode:
        Transfomers.ordinal_transformer = get_encoder(
            Transfomers.ordinal_transformer, OrdinalEncoder(dtype=int)
        )
        ordinal = Transfomers.ordinal_transformer
        label_cols = mode_cols + one_hot_cols
        if not is_encoder_trained(ordinal):
            method = ordinal.fit_transform
            df[label_cols] = method(df[label_cols].astype(str))
        else:
            # Annoyingly have to do this, otherwise sklearn can't handle missing values
            ordinal_dict = {}
            for i, v in zip(label_cols, ordinal.categories_):
                mapping_dict = {k: j[0] for j, k in np.ndenumerate(v)}
                ordinal_dict[i] = mapping_dict

            def f(x: pd.Series):
                base_dict = ordinal_dict[x.name]
                return x.apply(lambda y: base_dict.get(str(y), -1))

            df[label_cols] = df[label_cols].apply(f)

    if Config.impute_missing_mode_values:
        Transfomers.mode_imputer = get_encoder(
            Transfomers.mode_imputer, SimpleImputer(strategy="most_frequent")
        )
        imputer = Transfomers.mode_imputer
        if is_encoder_trained(imputer):
            method = imputer.transform
        else:
            method = imputer.fit_transform
        df[mode_cols] = method(df[mode_cols])
        Transfomers.mode_cols = mode_cols
        # Median imputation
    if Config.impute_missing_median_values:
        Transfomers.median_imputer = get_encoder(
            Transfomers.median_imputer, SimpleImputer(strategy="median")
        )
        median_imputer = Transfomers.median_imputer
        if is_encoder_trained(median_imputer):
            method = median_imputer.transform
        else:
            method = median_imputer.fit_transform
        df[median_cols] = method(df[median_cols])
    if Config.one_hot_encode:
        Transfomers.one_hot_encoder = get_encoder(
            Transfomers.one_hot_encoder,
            OneHotEncoder(sparse=False, handle_unknown="ignore"),
        )
        one_hot = Transfomers.one_hot_encoder
        if is_encoder_trained(one_hot):
            method = one_hot.transform
        else:
            method = one_hot.fit_transform
        res = method(df[one_hot_cols].astype(str))
        Transfomers.one_hot_cols = one_hot_cols
        col_list = []
        for i, j in zip(one_hot_cols, one_hot.categories_):
            col_list += ["is_{}_{}".format(str(k).replace(" ", "_"), i) for k in j]
        one_hot_df = pd.DataFrame(res, columns=col_list, dtype=int)
        one_hot_df.index = df.index
        df = pd.concat([df, one_hot_df], axis=1)

    log_dataframe_information(df)
    suspect_rows = df[df["foreigner_index"] + df["residence_index"] == 0]
    logger.info("Have %s suspect rows", len(suspect_rows))
    if Transfomers.final_cols is not None:
        df = df[Transfomers.final_cols]
    else:
        final_cols = [
            i
            for i in df.columns
            if i not in set(one_hot_cols) and i not in set(drop_cols)
        ]
        df = df[final_cols]
        Transfomers.final_cols = final_cols
    df["is_train"] = df["is_train"].astype(bool)
    return df


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(df["snapshot_date"].unique())
    df = df.sort_values(by="snapshot_date")
    cust_df = df.groupby("customer_id")
    # Will assume there cols are unchanging
    look_back_ignore_columns = [
        "customer_id",
        "spouse_index",
        "address",
        "province_code",
        "entry_channel",
        "country",
        "sex",
        "residence_index",
        "foreigner_index",
        "deceased_index",
    ]
    look_back_cols = []
    for i in df.columns:
        for j in look_back_ignore_columns:
            if j in i:
                break
        else:
            look_back_cols.append(i)

    for col in df[look_back_cols]:
        for i in range(1, Config.look_back_level + 1):
            df["prev_{}_{}".format(i, col)] = cust_df[col].shift(i)
    # Ignore rows with no product bought
    rows_with_acquisition = get_product_deltas(df)
    rows_with_acquisition = rows_with_acquisition.any(axis=1)
    rows_to_keep = ~df["is_train"] | rows_with_acquisition
    df = df[rows_to_keep]

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
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_iterations": 1000,
        "max_depth": 8,
        "learning_rate": 0.06,
    }
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
    train_column_check = lambda i: not is_test_column(i) and not any(
        [j in i for j in ("snapshot_date", "is_train", "customer_id")]
    )
    train_x = train_df[[i for i in train_df.columns if train_column_check(i)]]
    # train_y = train_df[[i for i in train_df.columns if test_column_check(i)]]
    train_y = get_product_deltas(train_df)
    return train_x, train_y


def get_prediction(df: pd.DataFrame, model_map: dict) -> pd.DataFrame:
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
    pred_cols = [i for i in prediction_rows.columns if is_pred_column(i)]
    translation_dict = get_translation_dict()
    inverse_translation_dict = {v: i for i, v in translation_dict.items()}
    # translated_pred_cols = np.array([inverse_translation_dict[i.replace('will_have_','')] for i in pred_cols])
    ind_prod_prediction = prediction_rows[pred_cols]
    # bought_product = ind_prod_prediction > 0.5
    def get_products(x):
        products = []
        x = x.sort_values(ascending=False)
        for i, v in x.iteritems():
            if (
                v <= Config.product_recommendation_threshold and len(products) < 7
            ) or v > Config.product_recommendation_threshold:
                products.append(inverse_translation_dict[i.replace("will_have_", "")])
        return " ".join(products)

    # preds =np.argsort(ind_prod_prediction, axis=1)
    # preds = np.fliplr(preds)[:, :7]
    # preds = [' '.join(translated_pred_cols[i]) for i in preds]

    prediction_cols = ind_prod_prediction.apply(get_products, axis=1)

    pred_df = pd.concat(
        [pd.DataFrame(prediction_cols), prediction_rows["customer_id"]], axis=1
    )
    pred_df.columns = ["added_products", "ncodpers"]
    pred_df = pred_df[list(pred_df.columns)[::-1]]

    # pred_df['ncodpers'] = prediction_rows['customer_id']
    logger.info("Saving predictions to csv")

    pred_df.to_csv(
        "my_pred_{}.csv".format(datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")),
        index=False,
    )
    logger.info("Saved predictions to csv")


def apply_pipeline(df: pd.DataFrame, *args: Callable) -> pd.DataFrame:
    out = df
    for callable in args:
        logger.info("Applying callable %s to data", callable)
        out = callable(out)
    return out


@time_it
def main():
    try:
        logger.info("Starting pipeline")
        train_date_list = [
            "2015-01-28",
            "2015-02-28",
            "2015-03-28",
            "2015-04-28",
            "2015-05-28",
            "2015-06-28",
            "2015-07-28",
            "2015-08-28",
            "2015-09-28",
            "2015-10-28",
            "2015-11-28",
            "2015-12-28",
        ]
        test_date_list = ["2016-04-28", "2016-05-28"]
        train_data = get_date_limited_train_data(train_date_list)
        extra_test_data = get_date_limited_train_data(test_date_list)
        Data.full_train_data = None
        # all_data = get_date_limited_train_data(date_list=train_date_list+test_date_list,customers=10)
        # train_data = all_data[all_data['fecha_dato'].isin(train_date_list)]
        # train_data = get_customer_dataframe()
        train_data["is_train"] = True
        # additional_rows_for_test = train_data[train_data[snapshot_col]==max_snapshot]
        pipeline = (translate_columns, clean_data, process_df)
        train_data = apply_pipeline(train_data, *pipeline)
        log_dataframe_information(train_data)
        test_data = get_test_data()
        # test_data = test_data[test_data['ncodpers'].isin(train_data['customer_id'].unique())]
        test_data["is_train"] = False
        test_data = test_data.append(extra_test_data)
        test_data = apply_pipeline(test_data, *pipeline)
        test_data = test_data[~test_data["is_train"]]
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
    except:
        logger.exception("Error occurred")


if __name__ == "__main__":
    main()
