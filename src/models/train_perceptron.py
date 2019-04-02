""" Trains a perceptron (linear regression) using one-hot encoding
    data generated using build_features.build_card_one_hot.
"""
import pandas as pd
import numpy as np
import mxnet as mx
import logging
from src.features import build_features
from mxnet import nd, autograd, gluon

# Define arguments to be passed to model
epochs=50
batch_size=64
learning_rate=0.01
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
train_data, test_data = build_features.build_transaction_data()
target_data = pd.read_csv('data/raw/train.csv', usecols=['target'], dtype=np.float32)

# Cast int64 to float64
for col in train_data.select_dtypes(['int', 'float']).columns:
    train_data[col] = train_data[col].astype(np.float32)
    
    assert col in test_data.columns
    test_data[col] = test_data[col].astype(np.float32)

target_data['target'] = target_data.target.astype(np.float32)

logger.info("Defining data loader")
X_train = train_data[[c for c in train_data.columns]].values
y_train = target_data.target.values

X_test = test_data[[c for c in test_data.columns]].values 
test_ids = test_data.index.values

# Setup the number of inputs
num_inputs=X_train.shape[1]
num_hidden=num_inputs * 2

# Setup iterable data sets
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X_train, y_train), 
                                    batch_size=batch_size, shuffle=True)

test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X_test), 
                                    batch_size=batch_size, shuffle=False)

logger.info("Defining Perceptron")
# net = gluon.nn.Sequential()
# with net.name_scope():
#     net.add(gluon.nn.Dense(num_outputs))
net = gluon.nn.Dense(1)
    
# Parameter initialization
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)

# Define loss function
# loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
loss_function = gluon.loss.L2Loss()

# Optimizer
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})

logger.info("Starting training loop")
for e in range(epochs):
    cumulative_loss = 0
    # Evaluation metric -- Root Mean Squared Error
    train_accuracy = mx.metric.RMSE()

    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)

        with autograd.record(train_mode=True):
            output = net(data)
            loss = loss_function(output, label)
        
        loss.backward()
        trainer.step(batch_size)
        cumulative_loss += nd.sum(loss).asscalar()
    
    # Calculate training error
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1, num_inputs))
        label = label.as_in_context(model_ctx)
        
        output = net(data)
        train_accuracy.update(label, output)
    
    logger.info("Epoch %s. Loss: %s, Train_acc %s" %
                (e, cumulative_loss/num_examples, train_accuracy.get()))

logger.info("Creating test set predictions")
entry_counter = 0

df_test = pd.DataFrame()

for i, data in enumerate(test_data):
    data = data.as_in_context(model_ctx).reshape((-1, num_inputs))
    output = net(data)

    df_test = df_test.append(pd.DataFrame(
        {'card_id': test_ids[entry_counter:(entry_counter + len(output))], 
        'target': output.asnumpy().reshape(-1)}
    ))

    entry_counter += len(output)

df_test.to_csv("data/processed/Perceptron.csv", index=False)