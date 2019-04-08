""" Trains a perceptron (linear regression) using one-hot encoding
    data generated using build_features.build_card_one_hot.
"""
import pandas as pd
import numpy as np
import logging
import gc
from src.features import build_features
from keras.models import Sequential
from keras.layers import Dense

# Set up data dir
data_dir = "data/processed/"

# Define arguments to be passed to model
epochs=50
batch_size=64
learning_rate=0.01
num_outputs=1

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

# # Cast int64 to float64
# for col in train_data.select_dtypes(['int', 'float']).columns:
#     train_data[col] = train_data[col].astype(np.float32)
    
#     assert col in test_data.columns
#     test_data[col] = test_data[col].astype(np.float32)

# target_data['target'] = target_data.target.astype(np.float32)

logger.info("Creating input matrices")
X_train = train_feats.values
y_train = train_target.values.reshape(-1)

# Setup the number of inputs
num_inputs=X_train.shape[1]
num_hidden=num_inputs * 2

del train_feats, train_target
gc.collect()

logger.info("Defining Perceptron")
model = Sequential()
model.add(Dense(batch_size, input_dim=num_inputs, kernel_initializer='normal', activation='relu'))
model.add(Dense(num_outputs, kernel_initializer='normal'))
# Compile model
model.compile(optimizer='rmsprop', loss='mse')

logger.info(f"Starting training process with epochs:{epochs} & batch size:{batch_size}")
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

########################### Validation Cycle ###########################
# logger.info("Loading validation data")
# val_feats = pd.read_csv(f"{data_dir}validation_features.csv")
# val_target = pd.read_csv(f"{data_dir}validation_target.csv")

# val_feats.set_index("card_id", inplace=True)
# val_target.set_index("card_id", inplace=True)

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


df_test.to_csv("data/processed/Perceptron.csv", index=False)