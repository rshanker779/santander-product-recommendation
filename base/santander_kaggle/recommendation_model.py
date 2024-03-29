import os
from typing import Tuple, Callable, Any
import lightgbm as lgb
import pandas as pd
from rshanker779_common.logger import get_logger, get_default_formatter
from rshanker779_common.utilities import time_it
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import datetime
import logging
from functools import partial
import joblib
import santander_kaggle.recommendation_data as d
import santander_kaggle.recommendation_utilities as utils
import numpy as np


logger = get_logger(__name__)
file_handler = logging.FileHandler("{}.log".format(__file__))
file_handler.setFormatter(get_default_formatter())
logger.addHandler(file_handler)


class Config:
    """
    Class to hold any global config. Note no guarantee (in fact very unlikely)
    all possible combos of config result in a working program
    """

    data_directory = os.path.join("..", "data")
    train_test_date_split = "2015-10-01"
    look_back_level = 2
    extra_look_back = None
    impute_missing_cat_values = False
    impute_missing_con_values = True
    one_hot_encode = False
    label_encode = True
    product_recommendation_threshold = 0.5
    save_sample_pred_df = True
    save_models_and_pipeline = True
    compute_map_error = True
    customer_limt = None


class RecommendationModelTransformer:
    """
    Class to hold transformer information for our data
    """

    def __init__(self):
        self.one_hot = utils.Encoder(
            OneHotEncoder(sparse=False, handle_unknown="ignore"), d.categorical_cols()
        )
        self.ordinal = utils.Encoder(OrdinalEncoder(dtype=int), d.categorical_cols())
        self.categorical_imputer = utils.Encoder(
            SimpleImputer(strategy="most_frequent"), d.categorical_cols()
        )
        self.continuous_imputer = utils.Encoder(
            SimpleImputer(strategy="median"), d.continuous_cols()
        )
        self.pipeline = (translate_columns, clean_data, process_df)


class PersistedModel:
    """
    Class to hold our trained models and pipeline
    """

    def __init__(self, model_map: dict, transformer: RecommendationModelTransformer):
        self.model_map = model_map
        self.transformer = transformer


def translate_columns(df: pd.DataFrame) -> pd.DataFrame:
    translation_dict = utils.get_translation_dict()
    df = df.rename(columns=translation_dict)
    return df


def is_test_column(i: str) -> bool:
    return "prev" not in i and "has" in i and "will" not in i


def is_pred_column(i: str) -> bool:
    return "will" in i


def get_time_col(default, format, x):
    return x.strftime(format) if not pd.isnull(x) else default


def get_columns_that_exist(desired_columns: list, df_columns: list):
    return [i for i in desired_columns if i in set(df_columns)]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    # Clean date formats
    for date_col in d.date_cols():
        df[date_col] = pd.to_datetime(df[date_col])
        for (col_name, date_format) in (("month_{}", "%m"), ("year_{}", "%y")):
            df[col_name.format(date_col)] = (
                df[date_col].apply(partial(get_time_col, -1, date_format)).astype(int)
            )

    df = df.replace("\s*NA", pd.np.nan, regex=True)
    na_cols = [i for i in df.columns if i != "is_train" and not is_pred_column(i)]
    df[na_cols] = df[na_cols].fillna(-1)

    drop_cols = ["province_name", "initial_signup_date", "last_date_as_primary"]

    final_cols = get_features()
    df = df[final_cols]
    if Config.label_encode:
        ordinal = transformer.ordinal.encoder
        label_cols = get_columns_that_exist(transformer.ordinal.cols, df.columns)
        if not utils.is_encoder_trained(ordinal):
            df[label_cols] = ordinal.fit_transform(df[label_cols].astype(str))
        else:
            logger.info("Encoder %s is trained", ordinal)
            # Annoyingly have to do this, otherwise sklearn can't handle missing values
            # ordinal_dict = utils.get_ordinal_dict_from_encoder(ordinal, label_cols)
            ordinal_dict = utils.get_ordinal_dict_from_encoder(ordinal, label_cols)
            f = partial(map_dataframe_with_dict_and_default, -1, ordinal_dict)
            df[label_cols] = df[label_cols].apply(f)
        df[label_cols] = df[label_cols].astype(int)
    if Config.impute_missing_cat_values:

        imputer = transformer.categorical_imputer.encoder
        mode_cols = get_columns_that_exist(
            transformer.categorical_imputer.cols, df.columns
        )
        if utils.is_encoder_trained(imputer):
            logger.info("Encoder %s is trained", imputer)
            method = imputer.transform
        else:
            method = imputer.fit_transform
        df[mode_cols] = method(df[mode_cols].astype(str))

    if Config.impute_missing_con_values:
        median_imputer = transformer.continuous_imputer.encoder
        median_cols = get_columns_that_exist(
            transformer.continuous_imputer.cols, df.columns
        )
        if utils.is_encoder_trained(median_imputer):
            logger.info("Encoder %s is trained", median_imputer)
            method = median_imputer.transform
        else:
            method = median_imputer.fit_transform
        df[median_cols] = method(df[median_cols])

    if Config.one_hot_encode:
        one_hot = transformer.one_hot.encoder
        one_hot_cols = get_columns_that_exist(transformer.one_hot.cols, df.columns)
        if utils.is_encoder_trained(one_hot):
            method = one_hot.transform
        else:
            method = one_hot.fit_transform
        res = method(df[one_hot_cols].astype(str))
        drop_cols += one_hot_cols
        col_list = []
        for i, j in zip(one_hot_cols, one_hot.categories_):
            col_list += ["is_{}_{}".format(str(k).replace(" ", "_"), i) for k in j]
        one_hot_df = pd.DataFrame(res, columns=col_list, dtype=int)
        one_hot_df.index = df.index
        df = pd.concat([df, one_hot_df], axis=1)

    utils.log_dataframe_information(df)
    # suspect_rows = df[df["foreigner_index"] + df["residence_index"] == 0]
    # logger.info("Have %s suspect rows", len(suspect_rows))
    df = df[[i for i in df.columns if i not in set(drop_cols)]]
    if "is_train" in set(df.columns):
        df["is_train"] = df["is_train"].astype(bool)
    else:
        df["is_train"] = True
    # df = df[final_cols]
    logger.info("Dataframe cols %s", df.columns)
    return df


def get_features():
    final_cols = [
        "snapshot_date",
        "customer_id",
        "age",
        "is_new_customer",
        "address",
        "activity_index",
        "gross_household_income",
        # "spouse_index",
        # "employee_index",
        # "customer_seniority",
        # "is_primary",
        # "customer_type_beginning_of_the_month",
        # "customer_relation_beginning_of_the_month",
        # "province_code",
        # "entry_channel",
        # "country",
        "sex",
        "residence_index",
        "foreigner_index",
        "deceased_index",
        "segmentation",
        "has_saving_account",
        "has_guarantees",
        "has_current_account",
        "has_derivada_account",
        "has_payroll_account",
        "has_junior_account",
        "has_mas_particular_account",
        "has_particular_account",
        "has_particular_plus_account",
        "has_short_term_deposits",
        "has_medium_term_deposits",
        "has_long_term_deposits",
        "has_e_account",
        "has_funds",
        "has_mortgage",
        "has_plan_pensions",
        "has_loans",
        "has_taxes",
        "has_credit_card",
        "has_securities",
        "has_home_account",
        "has_payroll",
        "has_nom_pensions",
        "has_direct_debit",
        "is_train",
        "month_snapshot_date",
        "year_snapshot_date",
        "month_initial_signup_date",
        "year_initial_signup_date",
        "month_last_date_as_primary",
        "year_last_date_as_primary",
    ]
    return final_cols


def map_dataframe_with_dict_and_default(default: Any, mapping_dict: dict, x: pd.Series):
    col_dict = mapping_dict[x.name]
    return x.apply(lambda y: col_dict.get(str(y), default))


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(df["snapshot_date"].unique())
    df = df.sort_values(by="snapshot_date")
    cust_df = df.groupby("customer_id")
    # Will assume there cols are unchanging
    look_back_ignore_columns = [
        "customer_id",
        "is_train",
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
    if Config.extra_look_back is not None:
        deeper_look_back_level = Config.extra_look_back
        deeper_look_back_cols = [i for i in df.columns if is_test_column(i)]
        for col in df[deeper_look_back_cols]:
            for i in range(Config.look_back_level + 1, deeper_look_back_level + 1):
                df["prev_{}_{}".format(i, col)] = cust_df[col].shift(i)
    # Ignore rows with no product bought
    logger.info(
        "Before removing non acquisitions, had %s customers, shape %s",
        len(df["customer_id"].unique()),
        df.shape,
    )
    rows_with_acquisition = get_product_deltas(df)
    rows_with_acquisition = rows_with_acquisition.any(axis=1)
    rows_to_keep = ~df["is_train"] | rows_with_acquisition
    df = df[rows_to_keep]
    logger.info(
        "After removing non acquisitions, had %s customers, shape %s",
        len(df["customer_id"].unique()),
        df.shape,
    )

    return df


def get_product_deltas(df: pd.DataFrame) -> pd.DataFrame:
    prev_product_cols = [i for i in df.columns if "has" in i and "1" in i]
    col_name_map = {i: i.replace("prev_1_", "") for i in prev_product_cols}
    rows_with_acquisition = pd.DataFrame()
    logger.info(col_name_map)
    for i, v in col_name_map.items():
        # Increase 0 to 1 represents an acquisition, 1 to 0 a loss. So need to check
        # greater than rather than  not equals
        bool_col = df[v] > df[i]
        rows_with_acquisition = pd.concat([rows_with_acquisition, bool_col], axis=1)
        rows_with_acquisition.columns = list(rows_with_acquisition.columns)[:-1] + [v]

    return rows_with_acquisition


def build_models(df: pd.DataFrame) -> dict:
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
    utils.log_dataframe_information(train_x)
    utils.log_dataframe_information(test_x)
    utils.log_dataframe_information(train_y)
    utils.log_dataframe_information(test_y)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_iterations": 100000,
        "max_depth": 15,
        "learning_rate": 0.001,
        "num_leaves": 2 ** 12,
        "min_data_in_leaf": 1000,
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
        feat_imp = pd.DataFrame(
            sorted(zip(model.feature_importance(), train_df.columns)),
            columns=["Value", "Feature"],
        )
        logger.info(feat_imp)
    return model_map


def get_x_y_variables(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_column_check = lambda i: not is_test_column(i) and not any(
        [j in i for j in ("snapshot_date", "is_train", "customer_id")]
    )
    train_x = train_df[[i for i in train_df.columns if train_column_check(i)]]
    train_y = get_product_deltas(train_df)
    return train_x, train_y


def get_prediction(
    df: pd.DataFrame, model_map: dict, cut_off_date="2016-05-01"
) -> pd.DataFrame:
    prediction_rows = df
    prediction_x, prediction_y = get_x_y_variables(prediction_rows)
    logger.info("Making predictions for %s rows", len(prediction_x))
    for i, v in model_map.items():
        df.loc[
            df["snapshot_date"] > cut_off_date, "will_have_{}".format(i)
        ] = v.predict(prediction_x)
    return prediction_rows


def get_formatted_prediction(prediction_rows: pd.DataFrame, save_csv=True) -> None:
    pred_cols = [i for i in prediction_rows.columns if is_pred_column(i)]
    translation_dict = utils.get_translation_dict()
    inverse_translation_dict = {v: i for i, v in translation_dict.items()}
    translated_pred_cols = np.array(
        [inverse_translation_dict[i.replace("will_have_", "")] for i in pred_cols]
    )
    ind_prod_prediction = prediction_rows[pred_cols]

    preds = np.argsort(ind_prod_prediction, axis=1)
    preds = np.fliplr(preds)[:, :7]
    preds = [" ".join(translated_pred_cols[i]) for i in preds]
    pred_df = pd.DataFrame(preds)
    pred_df.index = ind_prod_prediction.index

    pred_df = pd.concat([pred_df, prediction_rows["customer_id"]], axis=1)
    pred_df.columns = ["added_products", "ncodpers"]
    pred_df = pred_df[list(pred_df.columns)[::-1]]
    if save_csv:
        logger.info("Saving predictions to csv")

        pred_df.to_csv(
            "my_pred_{}.csv".format(
                datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
            ),
            index=False,
        )
        if Config.save_sample_pred_df:
            pred_df.iloc[:10].to_csv(
                os.path.join(d.Data.results_directory, "predictions.csv"), index=False
            )
        logger.info("Saved predictions to csv")
    return pred_df


def get_map_error(df):
    predictions = df[[i for i in df.columns if is_pred_column(i)]]
    actual = df[[i for i in df.columns if is_test_column(i)]]
    col_map = {i.replace("will_have_", ""): i for i in predictions.columns}
    actual = actual.rename(columns=col_map)
    look_up_array = np.array(predictions.columns)
    preds = np.argsort(predictions, axis=1)
    preds = np.fliplr(preds)[:, :7]
    preds = pd.DataFrame(preds, index=predictions.index)

    def f(x):
        # a = actual.iloc[x.name]
        z = x.apply(lambda y: 1 if actual.loc[x.name, look_up_array[y]] == 1 else 0)
        return z

    prediction_correct = preds.apply(f, axis=1)
    prediction_score = np.cumsum(prediction_correct, axis=1)
    divs = np.array([i + 1 for i in range(prediction_correct.shape[1])])
    prediction_score = prediction_score / divs
    map_score = np.multiply(prediction_score, prediction_correct).sum(axis=1)
    acutal_acquired = np.minimum(actual.sum(axis=1), 7)
    total_score = np.mean(np.divide(map_score, acutal_acquired))
    logger.info("Total MAP7 score is %s", total_score)
    return total_score


def apply_pipeline(df: pd.DataFrame, *args: Callable) -> pd.DataFrame:
    """
    Quick pipeline function that chains a series of functions that take a dataframe and
    return a dataframe. Could instead use sci-kit learn Pipeline
    """
    out = df
    for callable in args:
        logger.info("Applying callable %s to data", callable)
        out = callable(out)
    return out


transformer = RecommendationModelTransformer()


@time_it
def main():
    try:
        logger.info("Starting pipeline")
        train_date_list = [
            # "2015-03-28",
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
        test_date_list = ["2016-03-28", "2016-04-28", "2016-05-28"]
        train_data = d.get_date_limited_train_data(
            train_date_list, Config.customer_limt
        )
        logger.info("Customers: %s", len(train_data["ncodpers"].unique()))
        extra_test_data = d.get_date_limited_train_data(test_date_list)
        d.Data.full_train_data = None
        train_data["is_train"] = True
        pipeline = (translate_columns, clean_data, process_df)
        train_data = apply_pipeline(train_data, *pipeline)
        utils.log_dataframe_information(train_data)
        test_data = d.get_test_data()
        test_data["is_train"] = False
        test_data = test_data.append(extra_test_data)
        if Config.customer_limt is not None:
            test_data = test_data[
                test_data["ncodpers"].isin(train_data["customer_id"].unique())
            ]
        test_data = apply_pipeline(test_data, *pipeline)
        test_data = test_data[~test_data["is_train"]]
        utils.log_dataframe_information(test_data)

        models = build_models(train_data)
        if Config.compute_map_error:
            validation_data = train_data[
                train_data["snapshot_date"] > Config.train_test_date_split
            ]
            pred_rows = get_prediction(
                validation_data, models, Config.train_test_date_split
            )
            get_map_error(validation_data)
        prediction_rows = get_prediction(test_data, models)
        get_formatted_prediction(prediction_rows, True)
        if Config.save_models_and_pipeline:
            joblib.dump(
                PersistedModel(models, transformer),
                "transfomer_{}.pkl".format(
                    datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
                ),
            )

    except:
        logger.exception("Error occurred")


if __name__ == "__main__":
    main()
