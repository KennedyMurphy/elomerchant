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
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set up data dir
data_dir = "data/processed/"

# Define arguments to be passed to model
epochs=100
batch_size=64
num_outputs=1
num_features=100
n_sd = 4

# Keras arguments
optim_type = 'rmsprop'
loss_type = 'mse'

metrics = {'MAE': mean_absolute_error, "MSE": mean_squared_error}

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

########################### Training cycle ###########################
# Read in the data
logger.info("Loading training data")
train_feats = pd.read_csv(f"{data_dir}train_features.csv")
train_target = pd.read_csv(f"{data_dir}train_target.csv")

train_feats.set_index("card_id", inplace=True)
train_target.set_index("card_id", inplace=True)

logger.info("Flagging training set outliers")
outliers = oc.flag_normal_outliers(train_target, n_sd)

if outliers.values.any():
    logger.info(f"Flagged {outliers.values.sum()} of {len(outliers)} training observation as outliers. Dropping.")

    outliers = outliers.values.reshape(-1)
    train_feats = train_feats[~outliers]
    train_target = train_target[~outliers]

del outliers
gc.collect()

logger.info("Creating input matrices")
X_train = train_feats.values
y_train = train_target.values.reshape(-1)

# Setup the number of inputs
num_inputs=X_train.shape[1]

with mlflow.start_run():
    logger.info("Defining Feed Forward")
    layers = [(64, 'normal', 'relu'), (20, 'normal', 'relu')]
    ffnn = feed_forward.FeedForward("Feed Forward", layers, num_inputs, optim_type, loss_type)

    logger.info(f"Starting training process with epochs:{epochs} & batch size:{batch_size}")
    ffnn.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    ########################### Validation Cycle ###########################
    logger.info("Loading validation data")
    val_feats = pd.read_csv(f"{data_dir}validation_features.csv")
    val_target = pd.read_csv(f"{data_dir}validation_target.csv")

    val_feats.set_index("card_id", inplace=True)
    val_target.set_index("card_id", inplace=True)

    X_val = val_feats.values
    y_val = val_target.values.reshape(-1)

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
    mlflow.log_metric("mse", mean_squared_error(y_val, y_pred))
    mlflow.log_metric("mae", mean_absolute_error(y_val, y_pred))
    
    del val_feats, val_target, X_val, y_val, y_pred
    gc.collect()

########################### Prediction Cycle ###########################
logger.info("Starting test set prediction cycle.")
test_feats = pd.read_csv(f"{data_dir}test_features.csv")
test_feats.set_index("card_id", inplace=True)

X_test = test_feats.values

logger.info(f"Creating predictions with batch size:{batch_size}")
y_pred = ffnn.model.predict(X_test, batch_size=batch_size, verbose=1)

logger.info(f"Saving predictions")
df_test = pd.DataFrame({'card_id': test_feats.index.values, 'target': y_pred.reshape(-1, )})

if df_test.target.isnull().any():
    null_count = df_test.target.isnull().sum()
    logger.warning(f"Found {null_count} NaN preditions. Patching with 0...")
    df_test.target.fillna(0, inplace=True)

df_test.to_csv("data/processed/FeedForward.csv", index=False)