"""
Description : This file build LSTM RNN cell
Author      : hmshen
Date        : 02/01/2018
Reference   : https://github.com/tgjeon/TensorFlow-Tutorials-for-Time-Series
"""

import sys
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib import layers as tflayers
from tensorflow.contrib import rnn

def lstm_model(time_steps, rnn_layers, dense_layers=None, learning_rate=0.01,
               optimizer='Adagrad', learning_rate_decay_fn=None):

    '''
    Description: build a lstm_cells model
    Inputs:
    time_steps   : Time steps used to estimate the cloud fraction in the next data.
    layers       : num of layers in LSTM or a list of dict with layers configs.
    dense_layers : num of dense layers or a list of dict with dense layer configs.
    '''

    logging.info("Start building LSTM model ...")

    def lstm_cells(layers):

        logging.info("building lstm cells....")

        list_of_rnn = []
        if isinstance(layers[0], dict):
            # if layer configs are dictionary
            for layer in layers:
                if layer.get('keep_prob'):
                    list_of_rnn.append(rnn.DropoutWrapper(
                        rnn.BasicLSTMCell(layer['num_units'], state_is_tuple=True),
                        input_keep_prob=layer['input_keep_prob'],
                        output_keep_prob=layer['output_keep_prob']))
                else:
                    list_of_rnn.append(rnn.BasicLSTMCell(
                        layer['num_units'],
                        state_is_tuple=True))
        else:
            # if layer configs only contains the number layers
            for steps in layers:
                list_of_rnn.append(rnn.BasicLSTMCell(steps, state_is_tuple=True))

        logging.info("lstm cells has been built.")
        return list_of_rnn

    def dnn_layers(input_layers, layers):

        logging.info("building dense rnn layers...")

        if layers and isinstance(layers, dict):
            # stack required layers in to one
            return tflayers.stack(input_layers, tflayers.fully_connected,
                                  layers['layers'],
                                  activation=layers.get('activation'),
                                  dropout=layers.get('dropout'))
        elif layers:
            return tflayers.stack(input_layers, tflayers.fully_connected, layers)
        else:
            return input_layers

    def _lstm_model(features, labels):
        """
        Description: LSTM model generation.
        """

        logging.info("building LSTM model...")

        # create a stacked lstm accord. to input arguments
        stacked_lstm = rnn.MultiRNNCell(lstm_cells(rnn_layers), state_is_tuple=True)
        # split by time steps
        x_ =  tf.unstack(features, num=time_steps, axis=1)
        # apply rnn recurrently
        output, layers = rnn.static_rnn(stacked_lstm, x_, dtype=dtypes.float32)
        # feed the output of rnn into dnn
        output = dnn_layers(output[-1], dense_layers)
        # perform linear regression
        prediction, loss = tflearn.models.linear_regression(output, labels)
        train_optimizer = tf.contrib.layers.optimize_loss(
            loss, tf.train.get_global_step(), optimizer=optimizer,
            learning_rate = tf.train.exponential_decay(
                learning_rate, tf.contrib.framework.get_global_step(),
                decay_steps = 1000, decay_rate = 0.9, staircase=False, name=None))

        logging.info("LSTM model has been built...")

        return prediction, loss, train_optimizer

    # return configured lstm model
    return _lstm_model
