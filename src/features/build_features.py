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

    variable_types = {'feature_1': ft.variable_types.Categorical,
                        'feature_2': ft.variable_types.Categorical,
                        'feature_3': ft.variable_types.Categorical}

    logger = logging.getLogger(__name__)
    logger.info("Reading in data.")
    df = pd.read_csv('data/raw/train.csv')
    df['first_active_month'] = pd.to_datetime(df['first_active_month'] + "-01")

    logger.info("Creating entity set")
    es_train = ft.EntitySet()
    es_train = es_train.entity_from_dataframe(entity_id='transactions',
                                                dataframe=df,
                                                index='card_id',
                                                time_index="first_active_month",
                                                variable_types=variable_types)

    feature_matrix, feature_defs = ft.dfs(entityset=es_train,
                                      target_entity="transactions")

    logger.info("Creating one-hot training data")
    train_feature_matrix_enc, features_enc = ft.encode_features(feature_matrix, feature_defs)

    ft.save_features(features_enc, "feature_definitions")
    saved_features = ft.load_features('feature_definitions')

    logger.info("Creating one-hot test data")
    df = pd.read_csv('data/raw/test.csv')
    df['first_active_month'] = pd.to_datetime(df['first_active_month'] + "-01")
    df['target'] = 0
    es_test = ft.EntitySet()
    es_test = es_test.entity_from_dataframe(entity_id='transactions',
                                                dataframe=df,
                                                index='card_id',
                                                time_index="first_active_month",
                                                variable_types=variable_types)

    test_feature_matrix_enc = ft.calculate_feature_matrix(saved_features, es_test)
    test_feature_matrix_enc.drop(columns='target', inplace=True)

    return train_feature_matrix_enc, test_feature_matrix_enc