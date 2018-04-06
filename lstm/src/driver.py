'''
Description: Driver for LSTM cloud forecast
'''

import os
import sys
import logging
import argparse
import cPickle
import numpy as np
import utils as utl
import pandas as pd
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from cloud_forecast import Cloud_Forecaster, WRF_Irradiance_Forecaster

# logging.basicConfig(filename='../logs/cloud_forecast.log',
#                     filemode='w',
#                     level=logging.INFO,
#                     format='%(levelname)s %(message)s')

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(levelname)s %(message)s')

def arg_parser():

    """
    Description: Parse input arguments
    """
    # initialize parser
    parser = argparse.ArgumentParser(
        description='Run test cases on Benders Decomposition')

    # add arguments
    parser.add_argument("-p", "--path", help="input data path",
                        type=str, default='../data/NREL_total_1314_train.mat')

    parser.add_argument("-f", "--fmt",
                        help="Format of training labels: avg or variance.",
                        type=str, default='average')

    parser.add_argument("-m", "--mode",
                        help="Solar Irradiance or Cloud Fraction",
                        type=str, default='cloud')

    parser.add_argument("-o", "--outpath", help="output path",
                        type=str, default='../outputs/')

    parser.add_argument("-n", "--name", help="name of the dataset",
                        type=str, default='winter')

    parser.add_argument("-s", "--scale",
                        help='relationship bet. timesteps and data interval, 60',
                        type=int, default=1)

    args = parser.parse_args()

    return args


def main(args):
    '''
    Description: Main function for cloud fraction forecast
    '''
    # configurations for model
    configs = {'time_steps'     : 4,
               'rnn_layers'     : [{'num_units': 128}],
               'dense_layers'   : [128, 128],
               'training_steps' : 100000,
               'print_steps'    : 500,
               'pct_val'        : 0.0,
               'pct_test'       : 0.2,
               'learning_rate'  : 0.01,
               'data_name'      : args.name,
               'batch_size'     : 128}

    data = sio.loadmat(args.path)

    print 'input configs are:', configs

    if args.mode == 'cloud':
        # complete cloud fractoin prediction using NREL data
        cf = Cloud_Forecaster(data, configs, args.fmt, args.scale, args.outpath)
        cf.fit('total_cloud_fraction', 8, 15)
        rmse = cf.get_test_score()

    elif args.mode == 'solar':
        # complete solar irradiance forecasting using WRF data
        wrf = WRF_Irradiance_Forecaster(data, configs, args.fmt, args.scale, args.outpath)
        wrf.fit('surface_irradiance_obs')
        rmse = wrf.get_test_score()

    print 'RMSE is %.2f' % rmse

if __name__ == '__main__':

    """
    Description: Driver for LSTM cloud forecast.
    """

    # set up for numpy precision
    np.set_printoptions(precision=4)
    # receive input arguments
    args = arg_parser()
    print 'input arguments are: ', args
    # exec main func
    main(args)
