import numpy as np
import logging
import gc
from keras.models import Sequential
from keras.layers import Dense


class FeedForward():

    def __init__(self, model_name: str, layers: list, num_inputs: int, optim_type, loss_type):
        """ Initialization of Feed Forward neural network.

            :param model_name:      Name of FFNN
            :param layers:          List of tuples: (batch_size, kernel, activation)
            :param num_inputs:      Number of features inputed to initial layer.
            :param optim_type:      Optimizer type
            :loss_type:             Loss type
        """
        
        self.logger = logging.getLogger(__name__)
    
        self.model_name = model_name
        self.layers = layers
        self.num_inputs = num_inputs
        self.optim_type = optim_type
        self.loss_type = loss_type

        self.set_model()
    
    def set_model(self):
        self.model = Sequential()
        first = True
        for batch_size, kernel, activation  in self.layers:
            if first:
                self.logger.debug(f"Adding first layer with {self.num_inputs} inputs," +
                            f" {batch_size} batch size {kernel} kernel initializer," +
                            f" and {activation} activation function")
                # set the first layer
                dense = Dense(batch_size, 
                            input_dim=self.num_inputs, 
                            kernel_initializer=kernel, 
                            activation=activation)
                first = False
            else:
                self.logger.debug(f"Adding layer with {batch_size} batch size " + 
                                f"{kernel} kernel initializer," +
                                f" and {activation} activation function")
                dense = Dense(batch_size, 
                            kernel_initializer=kernel, 
                            activation=activation)
            
            self.model.add(dense)
        
        self.logger.debug("Adding final layer with 1 output and normal kernel initializer")
        self.model.add(Dense(1, kernel_initializer='normal'))

        self.logger.debug(f"Compiling model with {self.optim_type} " + 
                            f"optimizer and {self.loss_type} loss")
        self.model.compile(optimizer=self.optim_type, loss=self.loss_type)