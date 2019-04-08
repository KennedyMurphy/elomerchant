import pandas as pd
import numpy as np
import datetime

class ModelRecord():

    def __init__(self, record_file_path, metrics, batch_size, epochs):
        """ Creates a ModelRecord object that can take in
            a keras model and validation set, records
            accoracy and model settings.
        """

        self.record_file_path = record_file_path
        self.metrics = metrics
        self.batch_size = batch_size
        self.epochs = epochs

    def log(self, name,  model, X_val, y_val):
        """ Logs the performance of the model against 
            the validation set observations along with
            model parameters.

            :param name:    Name of the model
            :param model:   Keras object
            :param X_val:   numpy input matrix to the model
            :param y_val:   numpy array of target values
        """

        record = {
            "name": name,
            "epochs": self.epochs, 
            "time": datetime.datetime.now()}

        # Create predictions
        y_pred = model.predict(X_val, batch_size=self.batch_size)

        # Calculate each metric
        for key, val in self.metrics.items():
            record = {**record, key: val(y_val, y_pred)}

        try:
            log_df = pd.read_csv(self.record_file_path)
            log_df = log_df.append(pd.DataFrame(record, index=0), sort=True)
        except FileNotFoundError:
            log_df = pd.DataFrame(record, index=[0])

        
        log_df.to_csv(self.record_file_path, index=False)