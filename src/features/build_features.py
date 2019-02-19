import pandas as pd
import featuretools as ft
import logging

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

def build_card_one_hot():
    """ Reads in the raw data from train.csv and creates
        one-hot encodings for the feature and date fields.

        :return: Data frame with one-hot encoding
    """

    logger = logging.getLogger(__name__)
    logger.info("Reading in data.")
    df = pd.read_csv('data/raw/train.csv')
    df['first_active_month'] = pd.to_datetime(df['first_active_month'] + "-01")


    logger.info("Creating entity set")
    es = ft.EntitySet()
    es = es.entity_from_dataframe(entity_id='transactions',
                            dataframe=df,
                            index='card_id',
                            time_index="first_active_month",
                            variable_types={'feature_1': ft.variable_types.Categorical,
                                            'feature_2': ft.variable_types.Categorical,
                                            'feature_3': ft.variable_types.Categorical})

    feature_matrix, feature_defs = ft.dfs(entityset=es,
                                      target_entity="transactions")

    feature_matrix_enc, features_enc = ft.encode_features(feature_matrix, feature_defs)

    return feature_matrix_enc