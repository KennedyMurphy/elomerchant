import pandas as pd
import numpy as np
import featuretools as ft
import logging
import gc

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

TRANSACTION_LOAD_DTYPES = {
    'authorized_flag': 'object',
    'card_id': 'object',
    'city_id': 'int64',
    'category_1': 'category',
    'installments': 'int64',
    'category_3': 'category',
    'merchant_category_id': 'int64',
    'merchant_id': 'object',
    'month_lag': 'int64',
    'purchase_amount': 'float64',
    'category_2': 'category',
    'state_id': 'int64',
    'subsector_id': 'int64'
}



CARD_TYPES = {
    'feature_1': ft.variable_types.Categorical,
    'feature_2': ft.variable_types.Categorical,
    'feature_3': ft.variable_types.Categorical
}

TRANSACTION_TYPES = {
    'authorized_flag': ft.variable_types.Numeric,
    'category_1': ft.variable_types.Categorical,
    'category_2': ft.variable_types.Categorical,
    'category_3': ft.variable_types.Categorical,
    'installments': ft.variable_types.Numeric,
    'merchant_category_id': ft.variable_types.Categorical,
    'month_lag': ft.variable_types.Numeric,
    'purchase_amount': ft.variable_types.Numeric,
    'state_id': ft.variable_types.Categorical,
    'subsector_id': ft.variable_types.Categorical
}

def normalize_series(series, min_max):
    """ Normalizes a series where all values between
        0 and 1. Values can exceed 1 if they exceed the provided
        max in the min_max tuple.

        :param series:      Pandas series to normalize
        :param min_max:     Tuple of minimum and maximum value for
                                normalization.
        
        :return:            Normalized series
    """

    if min_max is None:
        min_max = series.agg(['min', 'max']).values
    
    assert min_max[0] < min_max[1]
    series = (series - min_max[0]) / (min_max[1] - min_max[0])

    return series

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
    es_train = ft.EntitySet()
    es_train = es_train.entity_from_dataframe(entity_id='transactions',
                                                dataframe=df,
                                                index='card_id',
                                                time_index="first_active_month",
                                                variable_types=CARD_TYPES)

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
                                                variable_types=CARD_TYPES)

    test_feature_matrix_enc = ft.calculate_feature_matrix(saved_features, es_test)
    test_feature_matrix_enc.drop(columns='target', inplace=True)

    return train_feature_matrix_enc, test_feature_matrix_enc


def build_transaction_data():
    """ Builds a data set from raw card and transaction data
        using the featuretools package.

        The resulting data set will be strictly concerned
        with transactions shown in the historical transactions CSV,
        and linking them to the proper card.

        :return:    training, testing feature matrices
    """

    logger = logging.getLogger(__name__)
    logger.info("Reading in card data")
    customer_df = pd.read_csv("data/raw/train.csv")
    customer_df['first_active_month'] = pd.to_datetime(customer_df['first_active_month'] + "-01")

    customer_df.drop(columns='target', inplace=True)

    logger.info("Reading in transactions")
    transactions_df = pd.read_csv("data/raw/historical_transactions.csv", dtype=TRANSACTION_LOAD_DTYPES)
    transactions_df['authorized_flag'] = np.where(transactions_df['authorized_flag'] == 'Y', 1, 0)
    transactions_df.reset_index(inplace=True)

    logger.info("Creating training entity set")
    es_train = ft.EntitySet()
    es_train = es_train.entity_from_dataframe(
        entity_id='customer', 
        dataframe=customer_df,
        index='card_id',
        time_index='first_active_month',
        variable_types=CARD_TYPES
    )

    es_train = es_train.entity_from_dataframe(
        entity_id='transactions',
        dataframe=transactions_df,
        index='index',
        variable_types=TRANSACTION_TYPES
    )

    del customer_df
    gc.collect()

    logger.info("Defining relationships")
    relationship = ft.Relationship(es_train['customer']['card_id'],
                                    es_train['transactions']['card_id'])

    es_train = es_train.add_relationship(relationship)

    feature_matrix, feature_defs = ft.dfs(entityset=es_train, target_entity='customer')

    train_feature_matrix_enc, features_enc = ft.encode_features(feature_matrix, feature_defs)

    ft.save_features(features_enc, "feature_definitions")
    saved_features = ft.load_features('feature_definitions')

    logger.info("Loading test data")
    customer_df = pd.read_csv("data/raw/test.csv")
    customer_df['first_active_month'] = pd.to_datetime(customer_df['first_active_month'] + "-01")

    logger.info("Creating testing entity set")
    es_test = ft.EntitySet()
    es_test = es_test.entity_from_dataframe(
        entity_id='customer', 
        dataframe=customer_df,
        index='card_id',
        time_index='first_active_month',
        variable_types=CARD_TYPES
    )

    es_test = es_test.entity_from_dataframe(
        entity_id='transactions',
        dataframe=transactions_df,
        index='index',
        variable_types=TRANSACTION_TYPES
    )
    
    es_test = es_test.add_relationship(relationship)

    test_feature_matrix_enc = ft.calculate_feature_matrix(saved_features, es_test)

    for col in train_feature_matrix_enc.columns:
        logger.debug(f"Normalizing feature [{col}]")
        old_min, old_max = train_feature_matrix_enc[col].agg(['min', 'max'])

        if (old_min == old_max):
            logger.debug(f"Droping feature [{col}] due to lack of variation")
            train_feature_matrix_enc.drop(columns=col, inplace=True)
            test_feature_matrix_enc.drop(columns=col, inplace=True)

            continue

        train_feature_matrix_enc[col] = normalize_series(
            series=train_feature_matrix_enc[col], 
            min_max=(old_min, old_max))

        assert col in test_feature_matrix_enc.columns

        test_feature_matrix_enc[col] = normalize_series(
            series=test_feature_matrix_enc[col], 
            min_max=(old_min, old_max))

    logger.info("Dropping SKEW features.")
    # TODO: Determine why these have lower counts than other features
    drop_cols = [c for c in train_feature_matrix_enc.columns if "SKEW" in c]
    train_feature_matrix_enc.drop(columns=drop_cols, inplace=True)
    test_feature_matrix_enc.drop(columns=drop_cols, inplace=True)

    return train_feature_matrix_enc, test_feature_matrix_enc