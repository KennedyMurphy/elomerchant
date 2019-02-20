""" Trains a multilayer perceptron using one-hot encoding
    data generated using build_features.build_card_one_hot.
"""
import pandas as pd
import numpy as np
import mxnet as mx
import logging
from src.features import build_features
from mxnet import nd, autograd, gluon

# Define arguments to be passed to model
epochs=10
batch_size=64
num_hidden=64
num_outputs=1
num_examples=60000


log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

logger.info("Setting contexts")
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
data_ctx = ctx
model_ctx = ctx

# Read in the data
train_data, test_data = build_features.build_card_one_hot()

# Cast int64 to float64
for col in train_data.select_dtypes('int64').columns:
    train_data[col] = train_data[col].astype(np.float32)
    
    assert col in test_data.columns
    test_data[col] = test_data[col].astype(np.float32)


logger.info("Defining data loader")
X_train = train_data[[c for c in train_data.columns if c != 'target']].values
y_train = train_data.target.values.reshape(-1, 1)

X_test = test_data[[c for c in test_data.columns if c != 'target']].values 
X_test = gluon.data.ArrayDataset(X_test)

# Setup the number of inputs
num_inputs=X_train.shape[1]

train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X_train, y_train), 
                                    batch_size=batch_size, shuffle=True)

logger.info("Defining MLP")
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    net.add(gluon.nn.Dense(num_outputs))

# Parameter initialization
net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=model_ctx)

# Define loss function
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# Optimizer
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})

logger.info("Starting training loop")
for e in range(epochs):
    cumulative_loss = 0
    # Evaluation metric -- Root Mean Squared Error
    train_accuracy = mx.metric.RMSE()

    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1, num_inputs))
        label = label.as_in_context(model_ctx)

        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        
        loss.backward()

        trainer.step(data.shape[0])
        cumulative_loss += nd.sum(loss).asscalar()
    
    # Calculate training error
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1, num_inputs))
        label = label.as_in_context(model_ctx)
        
        output = net(data)
        train_accuracy.update(label, output)
    
    logger.info("Epoch %s. Loss: %s, Train_acc %s" %
                (e, cumulative_loss/num_examples, train_accuracy.get()))

# logger.info("Creating test set predictions")
# df_test = pd.read_csv('data/raw/test.csv')
# df_test = df_test['card_id']
# test_output = net(X_test)
# df_test['target'] = test_output

# df_test.to_csv("data/processed/MLP Results.csv")