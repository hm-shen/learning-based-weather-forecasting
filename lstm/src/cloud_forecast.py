'''
Description: This file contains a class for cloud fraction forecast
Author: Haoming Shen
Date: 02/20/2018
'''

import sys
import logging
import numpy as np
import pandas as pd
import scipy.io as sio
from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from tensorflow.contrib.learn.python import SKCompat

import utils as utl
from rnn_lstm import lstm_model
from preprocessor import Cloud_Cover_Preprocessor, WRF_Irradiance_Preprocessor

class Base_Forecaster(object):

    '''
    Description: Base Weather Forecaster
    '''

    def __init__(self, raw_data, configs, outpath='../outpaths'):

        logging.info('Initializing Base weather forecaster ...')

        # load configurations
        self.raw_data = raw_data
        self.configs = configs
        # paths
        self.outpath = outpath

        logging.info('Initialization is complete.')

    def fit(self):

        pass

    def get_test_score(self):

        pass

class Cloud_Forecaster(Base_Forecaster):

    '''
    Description: Cloud cover forecast based on LSTM - RNN
    '''

    def __init__(self, raw_data, configs, mode, scale=1, outpath='../outputs'):

        '''
        scale: relationship bet. time steps and data interval (1/60)
        '''

        logging.info('Initializing Cloud Forecaster ...')

        super(Cloud_Forecaster, self).__init__(raw_data, configs, outpath)

        # load configurations
        self.scale = scale
        self.mode = mode

        logging.info('Cloud Forecaster is initialized.')

    def fit(self, datakey='total_cloud_fraction', ubd_min=8, lbd_max=15):

        logging.info('Start fitting data ...')

        # create tensorflow model
        self._init_model()

        # preprocess data
        ccp = Cloud_Cover_Preprocessor(self.raw_data, self.configs['time_steps'],
                                       datakey, self.scale, ubd_min, lbd_max,
                                       self.mode)
        self.feats, self.labels = ccp.preprocess()
        self._fit()

        logging.info('Fitting data is complete.')

    def _init_model(self):

        logging.info('Initializing LSTM model ...')

        self.regressor = SKCompat(
            learn.Estimator(
                model_fn=lstm_model(self.configs['time_steps'],
                                    self.configs['rnn_layers'],
                                    dense_layers=self.configs['dense_layers'],
                                    learning_rate=self.configs['learning_rate'])))

        logging.info('LSTM model is initialized.')

    def _fit(self):

        logging.info('Fitting training data ...')

        x_train = np.expand_dims(self.feats['train'], axis=2)
        y_train = np.expand_dims(self.labels['train'], axis=2)

        self.regressor.fit(x_train, y_train,
                           batch_size=self.configs['batch_size'],
                           steps=self.configs['training_steps'])

        logging.info('Training data is fitted.')

    def get_test_score(self):

        logging.info('Testing on test data sets ...')

        x_test = np.expand_dims(self.feats['test'], axis=2)

        preds = self.regressor.predict(x_test)
        mse = mean_squared_error(self.labels['test'], preds)

        rst_dict = {'preds': preds, 'labels': self.labels['test']}

        sio.savemat('../outputs/data_%s_fmt_%s.mat' %\
                    (self.configs['data_name'], self.mode),
                    rst_dict)

        logging.info('Testing is completed.')

        return np.sqrt(mse)

class WRF_Irradiance_Forecaster(Cloud_Forecaster):

    def fit(self, datakey='irradiance_diff'):

        logging.info('Start fitting data ...')

        # create tensorflow model
        self._init_model()

        # preprocess data
        wrfip = WRF_Irradiance_Preprocessor(self.raw_data,
                                            self.configs['time_steps'],
                                            datakey,
                                            self.scale, 0.2, self.mode)
        self.feats, self.labels = wrfip.preprocess()
        self._fit()

        logging.info('Fitting data is complete.')

        return self.get_test_score()
