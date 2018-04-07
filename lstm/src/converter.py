'''
Description: This file contains a converter for weather data
Author: Haoming Shen
Date: 03/04/2018
'''

import sys
import logging
import numpy as np

class Base_Converter(object):

    def __init__(self, func, time_steps, scale=1):

        # number of time_steps used to predict
        self.time_steps = time_steps
        # calculate average or variance
        self.func = func
        # Example:  scale = 60 and mode is average
        # means the label is the average of the next 60 points
        self.scale = scale

class Cloud_Cover_Converter(Base_Converter):

    def convert(self, rawdata):

        logging.info('Converting data ....')

        feats, labels = self._convert_to_rnn(rawdata)

        logging.info('Conversion is complete.')

        return feats, labels

    def _convert_to_rnn(self, data):

        return (self._get_rnn_data(data, labels=False),
                self._get_rnn_data(data, labels=True))

    def _get_rnn_data(self, indata, labels=False):

        '''
        Description: convert processed data to rnn format (supervised training data)
        '''

        logging.info('Converting input data to rnn data...')
        logging.debug('Shape of input data is: %s' % str(indata.shape))
        logging.debug('time steps is: %s' % str(self.time_steps))
        logging.debug('scale is: %s' % str(self.scale))

        rnn_data = []
        for day in range(indata.shape[0]):
            # get valid self.indata
            daily_data = indata[day,:]
            daily_data = daily_data[~np.isnan(daily_data)]

            if daily_data.size == 0:
                print indata[day,:]
                print 'Error: void data!'
                sys.exit(-1)
                continue

            daily_rnn_data = self._seq_to_rnn(daily_data, labels)
            rnn_data.extend(daily_rnn_data)

        # convert to np.float32 so that tensorflow can work
        return np.array(rnn_data, dtype=np.float32)

    def _seq_to_rnn(self, seqdata, labels=False):

        rnn_data = []
        for idx in range(0, len(seqdata) - self.time_steps, 1):
            if labels:
                # return labels rather than feat vec
                next_data = seqdata[idx + self.time_steps :\
                                       idx + self.time_steps + self.scale]
                label = self.func(next_data)
                rnn_data.append(label)

            else:
                # return feat vectors
                rnn_data.append(
                    seqdata[idx: idx + self.time_steps])

        return rnn_data


class WRF_Converter(Cloud_Cover_Converter):

    def _get_rnn_data(self, indata, labels=False):

        logging.info('Converting input data to rnn data...')
        logging.debug('Shape of input data is: %s' % str(indata.shape))
        logging.debug('time steps is: %s' % str(self.time_steps))
        logging.debug('scale is: %s' % str(self.scale))

        rnn_data = self._seq_to_rnn(indata, labels)
        return np.array(rnn_data, dtype=np.float32)
