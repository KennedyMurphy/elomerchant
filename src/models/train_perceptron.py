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

# Define arguments to be passed to model
epochs=50
batch_size=64
learning_rate=0.01
num_outputs=1
num_examples=60000


log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

# Read in the data
logger.info("Loading data")
train_data, test_data = build_features.build_transaction_data()
target_data = pd.read_csv('data/raw/train.csv', usecols=['target'], dtype=np.float32)

# Cast int64 to float64
for col in train_data.select_dtypes(['int', 'float']).columns:
    train_data[col] = train_data[col].astype(np.float32)
    
    assert col in test_data.columns
    test_data[col] = test_data[col].astype(np.float32)

target_data['target'] = target_data.target.astype(np.float32)

logger.info("Creating input matrices")
X_train = train_data[[c for c in train_data.columns]].values
y_train = target_data.target.values

X_test = test_data[[c for c in test_data.columns]].values 
test_ids = test_data.index.values

# Setup the number of inputs
num_inputs=X_train.shape[1]
num_hidden=num_inputs * 2

del train_data, test_data, target_data
gc.collect()

logger.info("Defining Perceptron")
model = Sequential()
model.add(Dense(batch_size, input_dim=num_inputs, kernel_initializer='normal', activation='relu'))
model.add(Dense(num_outputs, kernel_initializer='normal'))
# Compile model
model.compile(optimizer='rmsprop', loss='mse')

logger.info(f"Starting training process with epochs:{epochs} & batch size:{batch_size}")
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

logger.info(f"Creating predictions with batch size:{batch_size}")
y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)

logger.info(f"Saving predictions")
df_test = pd.DataFrame({'card_id': test_ids, 'target': y_pred.reshape(-1, )})

if df_test.target.isnull().any():
    null_count = df_test.target.isnull().sum()
    logger.warning(f"Found {null_count} NaN preditions. Patching with 0...")
    df_test.target.fillna(0, inplace=True)


df_test.to_csv("data/processed/Perceptron.csv", index=False)