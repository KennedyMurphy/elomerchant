# -*- coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np
import src.features.build_features as feats
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


def train_validation_split(valprop = 0.2):
    """ Generates features for train, test set and
        splits the training set into training and validation
        sets.

        :param valprop:     Proportion of training observations
                                for validation
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating train/validation set split of {1-valprop}/{valprop}")

    train_data, test_data = feats.build_transaction_data()
    target_data = pd.read_csv('data/raw/train.csv', usecols=['card_id', 'target'])
    target_data.set_index('card_id', inplace=True)

    train_ids = train_data.index.values
    validation_ids = np.random.choice(train_ids, int(len(train_data) * valprop))

    logger.info(f"Selecting {len(validation_ids)} of {len(train_ids)} training " +
        "observations for validation set."
    )

    logger.info("Saving validation set data...")
    train_data.loc[validation_ids].to_csv(
        "data/processed/validation_features.csv")
    target_data.loc[validation_ids].to_csv(
        "data/processed/validation_target.csv")

    logger.info("Saving training set data...")
    train_data.loc[~train_data.index.isin(validation_ids)].to_csv(
        "data/processed/train_features.csv")
    target_data.loc[~train_data.index.isin(validation_ids)].to_csv(
        "data/processed/train_target.csv")

    logger.info("Saving testing features")
    test_data.to_csv("data/processed/test_features.csv")


def main():
    """ Reads in raw data pulled via make data and generates a Postgresql
        database.
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    train_validation_split(0.2)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
