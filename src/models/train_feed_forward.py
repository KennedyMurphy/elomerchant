""" Trains a Feed Forward Neural Network using features generated
    by build_transaction_data().
"""
import pandas as pd
import numpy as np
import logging
import gc
from src.models import record_model
from src.features import build_features
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set up data dir
data_dir = "data/processed/"

# Define arguments to be passed to model
epochs=50
batch_size=64
learning_rate=0.01
num_outputs=1
num_features=100
hidden_layers=2

metrics = {'MAE': mean_absolute_error, "MSE": mean_squared_error}

recorder = record_model.ModelRecord(
    record_file_path='models/model_log.csv',
    metrics=metrics,
    batch_size=batch_size, 
    epochs=epochs)

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

logger.info("Creating input matrices")
X_train = train_feats.values
y_train = train_target.values.reshape(-1)

# Setup the number of inputs
num_inputs=X_train.shape[1]

logger.info("Defining Feed Forward")
model = Sequential()
model.add(Dense(batch_size, input_dim=num_inputs, kernel_initializer='normal', activation='relu'))
model.add(Dense(batch_size, kernel_initializer='normal', activation='relu'))
model.add(Dense(num_outputs, kernel_initializer='normal'))
# Compile model
model.compile(optimizer='rmsprop', loss='mse')

logger.info(f"Starting training process with epochs:{epochs} & batch size:{batch_size}")
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

########################### Validation Cycle ###########################
logger.info("Loading validation data")
val_feats = pd.read_csv(f"{data_dir}validation_features.csv")
val_target = pd.read_csv(f"{data_dir}validation_target.csv")

val_feats.set_index("card_id", inplace=True)
val_target.set_index("card_id", inplace=True)

X_val = val_feats.values
y_val = val_target.values.reshape(-1)

recorder.log("Feed Forward", model, X_val, y_val)

# del val_feats, val_target
########################### Prediction Cycle ###########################
test_feats = pd.read_csv(f"{data_dir}test_features.csv")
test_feats.set_index("card_id", inplace=True)

X_test = test_feats.values

logger.info(f"Creating predictions with batch size:{batch_size}")
y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)

logger.info(f"Saving predictions")
df_test = pd.DataFrame({'card_id': test_feats.index.values, 'target': y_pred.reshape(-1, )})

if df_test.target.isnull().any():
    null_count = df_test.target.isnull().sum()
    logger.warning(f"Found {null_count} NaN preditions. Patching with 0...")
    df_test.target.fillna(0, inplace=True)


df_test.to_csv("data/processed/FeedForward.csv", index=False)