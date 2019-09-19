import io

import pandas as pd
from rshanker779_common.logger import get_logger
logger = get_logger(__name__)
from typing import Callable, Iterable
from sklearn.base import BaseEstimator
import numpy as np
class Encoder:
    """Class to hold details about specific encoder"""
    def __init__(self, encoder:BaseEstimator, cols:Iterable[str]):
        self.encoder = encoder
        self.cols = cols

    # def fit(self, *args, **kwargs):
    #     return self.encoder.fit(*args, **kwargs)
    #
    # def transform(self, *args, **kwargs):
    #     return self.encoder.transform(*args, **kwargs)
    #
    # def fit_transform(self, *args, **kwargs):
    #     return self.encoder.fit_transform(*args, **kwargs)


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


def log_dataframe_information(df: pd.DataFrame) -> None:
    logger.info("Shape %s", df.shape)
    buffer = io.StringIO()
    df.info(buf=buffer, verbose=True)
    logger.info(buffer.getvalue())

def is_encoder_trained(encoder):
    return hasattr(encoder, "statistics_") or hasattr(encoder, "categories_")

def get_ordinal_dict_from_encoder(encoder:BaseEstimator, cols:Iterable):
    ordinal_dict = {}
    for i, v in zip(cols, encoder.categories_):
        mapping_dict = {k: j[0] for j, k in np.ndenumerate(v)}
        ordinal_dict[i] = mapping_dict
    return ordinal_dict