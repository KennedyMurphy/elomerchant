""" Trains a Feed Forward Neural Network using features generated
    by build_transaction_data().
"""
import pandas as pd
import numpy as np
import logging
import gc
import mlflow
import src.features.outlier_correction as oc
from src.features import build_features
from src.models import feed_forward
from src.models import metric_calculation
from pathlib import Path


def train_ffnn(data_dir, drop_outliers, model_name, layers, optim_type, loss_type, batch_size, epochs):
    """ Trains a Feed Forward Neural Network (FFNN) and
        records the accuracy on the validation set.

        :param data_dir:        Directory to find training & validation data
        :param drop_outliers:   Boolean indicating if outliers should be
                                    removed from data set.
        :param model_name:      Name of the model to be generated
        :param layers:          Layer arguments to pass to FeedForward
        :param optim_type:      Optimizer type
        :param loss_type:       Loss type.
        :param batch_size:      Batch size for modelling
        :param epochs:          Number of epochs for model training.
    """

    logger = logging.getLogger(__name__)
    n_sd = 4

    ########################### Training data ###########################
    # Read in the data
    logger.info("Loading training data")
    train_feats = pd.read_csv(f"{data_dir}train_features.csv")
    train_target = pd.read_csv(f"{data_dir}train_target.csv")

    train_feats.set_index("card_id", inplace=True)
    train_target.set_index("card_id", inplace=True)

    logger.info("Flagging training set outliers")
    outliers = oc.flag_normal_outliers(train_target, n_sd)

    if drop_outliers and outliers.values.any():
        logger.info(f"Flagged {outliers.values.sum()} of {len(outliers)} training observation as outliers. Dropping.")

        outliers = outliers.values.reshape(-1)
        train_feats = train_feats[~outliers]
        train_target = train_target[~outliers]

    # Create input matrices
    X_train = train_feats.values
    y_train = train_target.values.reshape(-1)

    # Setup the number of inputs
    num_inputs=X_train.shape[1]

    ########################### Validation data ###########################
    logger.info("Loading validation data")
    val_feats = pd.read_csv(f"{data_dir}validation_features.csv")
    val_target = pd.read_csv(f"{data_dir}validation_target.csv")

    val_feats.set_index("card_id", inplace=True)
    val_target.set_index("card_id", inplace=True)

    X_val = val_feats.values
    y_val = val_target.values.reshape(-1)

    logger.info("Flagging validation set outliers")
    outliers = oc.flag_normal_outliers(val_target, n_sd)

    if drop_outliers and outliers.values.any():
        logger.info(f"Flagged {outliers.values.sum()} of {len(outliers)} validation observation as outliers. Dropping.")

        outliers = outliers.values.reshape(-1)
        val_feats = val_feats[~outliers]
        val_target = val_target[~outliers]

    del outliers, val_feats, val_target, train_feats, train_target
    gc.collect()

    with mlflow.start_run():
        logger.info(f"Defining {model_name}")
        ffnn = feed_forward.FeedForward(model_name, layers, num_inputs, optim_type, loss_type)

        logger.info(f"Starting training process with epochs:{epochs} & batch size:{batch_size}")
        ffnn.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        # Generate validation predictions
        y_pred = ffnn.model.predict(X_val, batch_size=batch_size).reshape(-1)

        logger.debug("Logging MLFLOW parameters")
        mlflow.log_param("model_name", ffnn.model_name)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("optimizer", ffnn.optim_type)
        mlflow.log_param("loss", ffnn.loss_type)

        for i in range(len(layers)):
            mlflow.log_param(f"layer{i+1}_nodes", layers[i][0])
            mlflow.log_param(f"layer{i+1}_kernel", layers[i][1])
            mlflow.log_param(f"layer{i+1}_activation", layers[i][2])

        logger.debug("Logging MLFLOW metrics")
        metrics = metric_calculation.calculate_error(y_val, y_pred)
        for key, val in metrics.items():
            mlflow.log_metric(key, val)
        
        del X_val, y_val, y_pred, metrics
        gc.collect()

    # ########################### Prediction Cycle ###########################
    # logger.info("Starting test set prediction cycle.")
    # test_feats = pd.read_csv(f"{data_dir}test_features.csv")
    # test_feats.set_index("card_id", inplace=True)

    # X_test = test_feats.values

    # logger.info(f"Creating predictions with batch size:{batch_size}")
    # y_pred = ffnn.model.predict(X_test, batch_size=batch_size, verbose=1)

    # logger.info(f"Saving predictions")
    # df_test = pd.DataFrame({'card_id': test_feats.index.values, 'target': y_pred.reshape(-1, )})

    # if df_test.target.isnull().any():
    #     null_count = df_test.target.isnull().sum()
    #     logger.warning(f"Found {null_count} NaN preditions. Patching with 0...")
    #     df_test.target.fillna(0, inplace=True)

    # df_test.to_csv("data/processed/FeedForward.csv", index=False)

def main():
    """ Reads in raw data pulled via make data and generates a Postgresql
        database.
    """

    # Set up data dir
    data_dir = "data/processed/"

    # Define arguments to be passed to model
    epochs=100
    batch_size=64

    # Keras arguments
    optim_type = 'rmsprop'
    loss_type = 'mse'
    layers = [(64, 'normal', 'relu'), (10, 'normal', 'relu')]

    ########################### Outlier free model ###########################
    # train_ffnn(
    #     data_dir=data_dir,
    #     drop_outliers=True,
    #     model_name="Outlier Free FFNN",
    #     layers=layers,
    #     optim_type=optim_type,
    #     loss_type=loss_type,
    #     batch_size=batch_size,
    #     epochs=epochs)

    
    # ########################### Outlier included model ###########################
    # train_ffnn(
    #     data_dir=data_dir,
    #     drop_outliers=False,
    #     model_name="FFNN",
    #     layers=layers,
    #     optim_type=optim_type,
    #     loss_type=loss_type,
    #     batch_size=batch_size,
    #     epochs=epochs)

    ########################### Single Layer Perceptron ###########################
    optim_type = 'adam'
    loss_type = 'mse'
    layers = [(64, 'normal', 'relu')]

    ########################### Outlier free model ###########################
    train_ffnn(
        data_dir=data_dir,
        drop_outliers=True,
        model_name="Outlier Free Perceptron",
        layers=layers,
        optim_type=optim_type,
        loss_type=loss_type,
        batch_size=batch_size,
        epochs=epochs)

    
    ########################### Outlier included model ###########################
    train_ffnn(
        data_dir=data_dir,
        drop_outliers=False,
        model_name="Perceptron",
        layers=layers,
        optim_type=optim_type,
        loss_type=loss_type,
        batch_size=batch_size,
        epochs=epochs)
        

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()