# -*- coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np
import gc
import src.features.build_features as feats
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


def summarise_purchase_amount(df, prefix):
    """ Summarises the purchase amount for each card id
        in the provided data frame. Adds the prefix to each
        column.

        :param df:      Pandas data frame.
        :param prefix:  Prefix to append to each columns

        :return:        Summarized purchase data.
    """

    logger  = logging.getLogger(__name__)

    assert "card_id" in df.columns
    assert "purchase_amount" in df.columns

    logger.debug(f"Summarizing {prefix} purchases")
    df = df.groupby('card_id').purchase_amount.agg(
        {'sum', 'std', 'min', 'max', 'mean'})
    
    logger.debug(f"Renaming {prefix} columns")
    rename_dict = {c: prefix+c for c in df.columns}
    df.rename(columns=rename_dict, inplace=True)

    del rename_dict
    gc.collect()

    return df


def summarise_purchase_frequency(df, prefix):
    """ Summarises the time aspect of purchases
        for each card id in the provided data frame. 
        Adds the prefix to each column.

        :param df:      Pandas data frame
        :param prefix:  Prefix to append to each columns

        :return:        Summarized purchase time data.
    """

    logger = logging.getLogger(__name__)

    assert "card_id" in df.columns
    assert "purchase_date" in df.columns
    assert "purchase_amount" in df.columns

    logger.debug(f"Summarizing {prefix} daily purchases")
    dates = df.groupby(
        ['card_id', pd.Grouper(key='purchase_date', freq='d')]).purchase_amount.agg(['sum', 'count'])
    
    dates = dates.groupby("card_id")[['sum', 'count']].mean()

    dates.rename(columns={
        "sum": f"{prefix}_avg_daily_purchase", 
        "count": f"{prefix}_avg_daily_transactions"}, inplace=True)

    logger.debug(f"Summarizing {prefix} transaction frequencies.")
    df['time_since'] = df.groupby('card_id').purchase_date.diff()
    df['time_since'] = df.time_since.values /  np.timedelta64(1, 'h')  # Convert to hours

    df = df.groupby('card_id').time_since.agg(['mean', 'std'])

    df.rename(columns={
        "mean": f"{prefix}avg_transaction_freq", 
        "std": f"{prefix}std_transaction_freq"}, inplace=True)

    logger.debug(f"Combining {prefix} daily purchases and transaction frequencies.")

    dates = dates.merge(df, on='card_id')

    gc.collect()

    return dates


def parse_transactions(input_file, output_file):
    """ Parses historical or new transaction data found in the
        input_filepath and saves the result to the output filepath.

        Generated features:
        - Average amount
        - STD amount
        - total amount
        - frequency

        by category, state, city, authorized, total

        :param input_file:      Name of file to parse in raw data directory
        :param output_file:     File name to use when saving to interim directory.
    """

    transaction_types = {
        "card_id": "object",
        "month_lag": np.int64,
        "purchase_date": "object",
        "authorized_flag": "object",
        "category_1": "object",
        "category_2": "object",
        "category_3": "object",
        "installments": np.int64,
        "merchant_category_id": np.int64,
        "subsector_id": np.int64,
        "purchase_amount": np.float64,
        "city_id": np.int64,
        "state_id": np.int64
    }

    logger = logging.getLogger(__name__)

    if input_file != "historical_transactions.csv" and input_file != "new_merchant_transactions.csv":
        logger.error(f"Input file [{input_file}] not a transaction data source.")

    # Determine prefix
    if input_file == 'historical_transactions.csv':
        prefix = "hist_"
    else:
        prefix = 'new_'

    logger.info(f"Reading in {input_file}")
    df = pd.read_csv(f"data/raw/{input_file}", dtype=transaction_types)

    logger.info("Casting date values to datetime")
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])

    # Sort values based on date.
    df.sort_values('purchase_date', inplace=True)
    # Convert authorized flag to int
    df['authorized_flag'] = df['authorized_flag'] == 'Y'

    logger.info("Generating overall purchase summary")
    feats = summarise_purchase_amount(
        df=df[['card_id', 'purchase_amount']],
        prefix=f'{prefix}purchase_')

    logger.info("Generating approved purchase summary")
    temp = summarise_purchase_amount(
        df=df[df['authorized_flag']][['card_id', 'purchase_amount']],
        prefix=f'{prefix}authorized_')
    
    feats = feats.merge(temp, on='card_id', how='outer')
    del temp
    gc.collect()

    for i in range(3):
        vals = df[f"category_{i+1}"].unique()
        for v in vals:
            logger.debug(f"Feature set dimensions: {feats.shape}")
            logger.info(f"Generating category {i+1} == {v} purchase summary")
            temp = summarise_purchase_amount(
                    df=df[df[f"category_{i+1}"] == v][['card_id', 'purchase_amount']],
                    prefix=f'{prefix}category{i+1}{v}_authorized_')
            
            feats = feats.merge(temp, on='card_id', how='outer')
            del temp
            gc.collect()

    logger.info("Generating transaction time features")
    temp = summarise_purchase_frequency(
        df=df[['card_id', 'purchase_date', 'purchase_amount']],
        prefix=prefix)
    
    feats = feats.merge(temp, on='card_id', how='left')
    del temp
    gc.collect()

    logger.info("Filling missing values")
    feats.fillna(0, inplace=True)

    logger.info(f"Saving data to {output_file}")
    feats.to_csv(f"data/interim/{output_file}")


def train_validation_split(valprop = 0.2):
    """ Generates features for train, test set and
        splits the training set into training and validation
        sets.

        :param valprop:     Proportion of training observations
                                for validation
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating train/validation set split of {1-valprop}/{valprop}")

    logger.info("Reading in target data.")
    train_target = pd.read_csv("data/raw/train.csv", usecols=['card_id', 'target'])
    test_target = pd.read_csv("data/raw/test.csv", usecols=['card_id'])

    logger.info("Adding historical transaction features.")
    try:
        trans = pd.read_csv("data/interim/historical_transactions.csv")

        train_target = train_target.merge(trans, on='card_id', how='left')
        test_target = test_target.merge(trans, on='card_id', how='left')

        del trans, 
        gc.collect()
    except FileNotFoundError:
        logger.warning("Could not add historical transaction features.")
    
    logger.info("Adding new transaction features.")
    try:
        trans = pd.read_csv("data/interim/new_merchant_transactions.csv")

        train_target = train_target.merge(trans, on='card_id', how='left')
        test_target = test_target.merge(trans, on='card_id', how='left')

        del trans, 
        gc.collect()
    except FileNotFoundError:
        logger.warning("Could not add new merchant transaction features.")

    feat_cols = [c for c in train_target.columns if c not in ['card_id', 'target']]
    
    if len(feat_cols) == 0:
        logger.warning("No features added to training and testing set.")
    
    assert np.isin(feat_cols, test_target.columns).all()

    # Set index
    train_target.set_index('card_id', inplace=True)
    test_target.set_index('card_id', inplace=True)

    # Fill NAs
    train_target.fillna(0, inplace=True)
    test_target.fillna(0, inplace=True)

    train_ids = train_target.index.values
    validation_ids = np.random.choice(train_ids, int(len(train_target) * valprop))

    logger.info(f"Selecting {len(validation_ids)} of {len(train_ids)} training " +
        "observations for validation set."
    )

    logger.info("Saving validation set data...")
    train_target.loc[validation_ids][feat_cols].to_csv(
        "data/processed/validation_features.csv")
    train_target.loc[validation_ids][['target']].to_csv(
        "data/processed/validation_target.csv")

    logger.info("Saving training set data...")
    train_target.loc[~train_target.index.isin(validation_ids)][feat_cols].to_csv(
        "data/processed/train_features.csv")
    train_target.loc[~train_target.index.isin(validation_ids)][['target']].to_csv(
        "data/processed/train_target.csv")

    logger.info("Saving testing features")
    test_target.to_csv("data/processed/test_features.csv")


def main():
    """ Reads in raw data pulled via make data and generates a Postgresql
        database.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    parse_transactions("historical_transactions.csv", "historical_transactions.csv")
    parse_transactions("new_merchant_transactions.csv", "new_merchant_transactions.csv")

    train_validation_split(0.2)
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()