"""
Description : This file contains utility functions for LSTM cloud forecast
Author      : hmshen
Date        : 02/01/2018
Reference   : https://github.com/tgjeon/TensorFlow-Tutorials-for-Time-Series
"""

import sys
import logging
import numpy as np
import scipy.io as sio
import pandas as pd
from datetime import timedelta, date

def preprocess_daytime_data(input_data, time_steps, val_size=0.2,
                            test_size=0.2, scale=1, labels=False):
    '''
    Description: preprocess daytime data into training set.
    '''

    logging.info('Start pre-processing input data ...')

    # replace nan with a negative value!
    input_data[np.isnan(input_data)] = -1

    print 'Range of possible cloud fraction is [ %.1f, %.1f ]' %\
            (np.min(input_data), np.max(input_data))

    # split input data set into training, validation and test set.
    data_train, data_val, data_test = split_data(input_data, val_size, test_size)

    return (rnn_daytime_data(data_train, time_steps, scale=scale, labels=labels),
            rnn_daytime_data(data_val, time_steps, scale=scale, labels=labels),
            rnn_daytime_data(data_test, time_steps, scale=scale, labels=labels))

def split_data(input_data, val_size, test_size):

    '''
    Description: split input data into training, validation and test sets
    '''

    logging.info('Split input data set ...')

    # debugging
    print 'val_size is %.1f' % val_size
    print 'test_size is %.1f' % test_size

    ndays = input_data.shape[0]

    # split input data accord. training size, val size and test size
    itest = int(np.round(ndays * (1 - test_size)))
    ival = int(np.round(itest * (1 - val_size)))

    print 'ndays is:', ndays
    print 'itest is:', itest
    print 'ival is:', ival

    data_train = input_data[:ival,:]
    data_val = input_data[ival:itest,:]
    data_test = input_data[itest:,:]

    return data_train, data_val, data_test

# def rnn_daytime_data(data, time_steps, labels=False):

#     '''
#     Description: turn time series into feature vector and labels.
#     '''

#     rnn_data = []
#     for day in range(data.shape[0]):
#         data_today = data[day, data[day,:] > 0]
#         for idx in range(len(data_today) - time_steps):
#             if labels:
#                 # return labels rather than feat vec
#                 rnn_data.append(data_today[idx + time_steps])
#             else:
#                 # return feat vectors
#                 rnn_data.append(data_today[idx: idx + time_steps])

#     return np.expand_dims(np.array(rnn_data, dtype=np.float32), axis=2)

def rnn_daytime_data(data, time_steps, scale=1, labels=False):

    '''
    Description: turn time series into feature vector and labels.
    Use the minute data to predict the average cloud cover in next hour
    '''

    rnn_data = []
    selected_data = []
    for day in range(data.shape[0]):
        data_today = data[day, data[day,:] > 0]
        # data_today = 2 * data_today - 1
        if data_today.size == 0:
            print 'void data!'
            continue
        for idx in range(0, len(data_today) - time_steps * scale, 1):
            if labels:
                # return labels rather than feat vec
                selected_data.append(data_today[idx + time_steps * scale: idx + (time_steps+1) * scale])
                # rnn_data.append(np.average(data_today[idx + time_steps * scale: \
                #                                       idx + (time_steps+1) * scale]))

                rnn_data.append(np.var(data_today[idx + time_steps * scale: \
                                                      idx + (time_steps+1) * scale]))

            else:
                # return feat vectors
                rnn_data.append(data_today[idx: idx + time_steps * scale])

    # if labels is False:
    #     print 'features are:\n', rnn_data[0]
    #     print 'Next feature is:\n', rnn_data[1][-1]
    #     print 'Next feature is:\n', rnn_data[2][-1]
    #     print 'Next feature is:\n', rnn_data[3][-1]
    #     print 'Next feature is:\n', rnn_data[4][-1]
    #     print 'Next feature is:\n', rnn_data[5][-1]
    #     print 'Next feature is:\n', rnn_data[6][-1]
    #     print 'Next feature is:\n', rnn_data[7][-1]
    # else:
    #     print 'labels are:\n', rnn_data[0]
    #     print 'selected_data are:\n', selected_data[0]
    #     print 'next feature is:\n', rnn_data[1]

    return np.expand_dims(np.array(rnn_data, dtype=np.float32), axis=2)

def daterange(start_date, end_date):
    '''
    Description: loop through a whole year
    Source: https://stackoverflow.com/questions/1060279/iterating-through-a-range-of-dates-in-python
    '''
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

if __name__ == '__main__':

    '''
    Description: Test code
    '''
    np.set_printoptions(precision=4)

    data_path = '../data/NREL_MICD_cloud_fraction_2013.mat'
    raw_data = sio.loadmat(data_path)

    data = np.reshape(raw_data['total_cloud_fraction'], (-1, 24))[0:50,:]

    print 'whole day data block is:', data[0:3,:]
    preprocess_daytime_data(data, 5)