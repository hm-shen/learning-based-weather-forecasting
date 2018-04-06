'''
Description: Preprocessor for weather data
Author: Haoming Shen
Date: 03/30/2018
'''

import sys
import logging
import numpy as np
from base_preprocessor import NREL_Preprocessor, WRF_Preprocessor
from sklearn.model_selection import ShuffleSplit
from converter import Cloud_Cover_Converter, WRF_Converter

class Base_Weather_Preprocesor(object):

    def __init__(self, raw_data, test_size=0.2, random_state=None):

        logging.info('Initializing Cloud cover preprocessor ...')

        self.rawdata = raw_data
        self.test_size = test_size
        self.random_state = random_state

        logging.info('Cloud cover preprocessor is initialized.')

    def preprocess(self):

        pass

    def _convert(self):

        pass

    def _split_train_test(self):

        pass

    def _clean(self):

        pass

class RNN_Weather_Preprocessor(Base_Weather_Preprocesor):

    def __init__(self, raw_data, time_steps, converter,
                 scale=1, test_size=0.2,
                 mode='average', random_state=None):

        super(RNN_Weather_Preprocessor, self).__init__(raw_data, test_size,
                                                       random_state)

        self.converter = converter

        '''
        time_steps should be the actural number of samples used to
        predict: 120 minuts when data is recored minutes by minutes
        2 hours when data is recored hour by hour.
        '''
        self.time_steps = time_steps

        '''
        Given time_steps, we predict the `average` of the next N
        sample points, `scale` specifies this N, `mode` specifies
        the function applied to those N sample points (average or
        variance, etc)
        '''
        self.scale = scale

        self.mode = mode
        if mode == 'average':
            self.func = np.average
        elif mode == 'deviation':
            self.func = np.variance


    def preprocess(self):

        self.data = self._clean()
        self._split_train_test()
        self._convert()

        return self.feats, self.labels

    def _convert(self):

        # convert preprocessed data to rnn data format
        converter = self.converter(self.func, self.time_steps, self.scale)
        x_train, y_train = converter.convert(self.train_set)
        x_test, y_test = converter.convert(self.test_set)

        # debug info
        logging.debug('Shape of x_train is: %s' % str(x_train.shape))
        logging.debug('Shape of x_test is: %s' % str(x_test.shape))
        logging.debug('Shape of y_train is: %s' % str(y_train.shape))
        logging.debug('Shape of y_test is: %s' % str(y_test.shape))

        # fit data and test model on the test set
        self.feats = dict(train=x_train, test=x_test)
        self.labels = dict(train=y_train, test=y_test)

    def _split_train_test(self):

        # randomly choose test and training sets
        # ss = ShuffleSplit(n_splits=1, test_size=self.test_size,
        #                   random_state=self.random_state)
        # # get ids of training sets and test sets
        # train_inds, test_inds = ss.split(self.data).next()

        # logging.debug('Dates in test sets are: %s' %\
        #               str([self.dates[ind] for ind in test_inds]))

        # get training set and test sets
        # self.train_set = self.data[train_inds, :]
        # self.test_set = self.data[test_inds, :]

        # self.train_set = self.data[train_inds]
        # self.test_set = self.data[test_inds]

        # choose the last several days as test sets
        n_train = int(self.data.shape[0] * (1 - self.test_size))
        # select on the first dimension
        self.train_set = self.data[:n_train]
        self.test_set = self.data[n_train:]


        logging.debug('Size of train_set is %s' % str(self.train_set.shape))
        logging.debug('Size of test_set is %s' % str(self.test_set.shape))

    def _clean(self):

        # a trivial one, just for illustration for its children
        return self.rawdata


class WRF_Irradiance_Preprocessor(RNN_Weather_Preprocessor):

    def __init__(self, raw_data, time_steps, datakey, scale=1, test_size=0.2,
                 mode='average', random_state=None):

        logging.info('Initializing WRF Irradiance Preprocessor ...')

        super(WRF_Irradiance_Preprocessor, self)\
            .__init__(raw_data, time_steps, WRF_Converter, scale,
                      test_size, mode, random_state)

        assert datakey in self.rawdata.keys(),\
            'ERROR: datakey does not exist in input dictionary!'

        # which data to process in the input dictionary
        self.datakey = datakey

        logging.info('Preprocessor is initialized.')

    def preprocess(self):

        logging.info('Start preprocess data for WRF Irradience ...')

        wrfpp = WRF_Preprocessor(outfmt='consecutive')
        wrfpp.load(self.rawdata)
        self.data, self.dates = wrfpp.preprocess(self.datakey)

        logging.debug('Shape of self.data is: %s' % str(self.data.shape))
        logging.debug('Shape of self.dates is: %s' % str(len(self.dates)))

        self._split_train_test()
        self._convert()

        logging.info('Preprocessing is complete.')

        return self.feats, self.labels


class Cloud_Cover_Preprocessor(RNN_Weather_Preprocessor):

    def __init__(self, raw_data, time_steps, datakey, scale, ubd_min, lbd_max,
                 mode='average', n_samples_per_day=1, test_size=0.2,
                 random_state=None):

        # use the init of its parent
        super(Cloud_Cover_Preprocessor, self).\
            __init__(raw_data, time_steps, Cloud_Cover_Converter,
                     scale, test_size, mode, random_state)

        '''
        only select days whose smallest index of valid data larger
        than ubd_min (upper bound for min) and whose largest index
        for valid data smaller than lbd_max (lower bound for max)
        '''
        self.ubd_min = ubd_min
        self.lbd_max = lbd_max

        # each day should have at least n_samples_per_day valid data
        self.n_samples_per_day = n_samples_per_day

        # set datakey
        self.datakey = datakey

    def preprocess(self):

        ''' preprocess, clean, train_test_split and convert to RNN form '''

        logging.info('Start preprocessing data for Cloud Cover Forecasting ...')

        # use NREL preprocessor to preprocess
        nrelpp = NREL_Preprocessor(outfmt='daybyday')

        # convert to day by day data
        daybyday_data, selected_dates, ind_mins, ind_maxes\
            = nrelpp.load_and_preprocess(self.rawdata, self.datakey)

        # only keep days with enough number of samples each day
        self.data, self.dates = self._clean(daybyday_data, selected_dates,
                                            ind_mins, ind_maxes)

        logging.debug('Shape of self.data is: %s' % str(self.data.shape))
        logging.debug('Shape of self.dates is: %s' % str(len(self.dates)))

        self._split_train_test()
        self._convert()

        logging.info('Preprocessing is complete.')

        return self.feats, self.labels

    def _clean(self, dirtydata, dirtydates, ind_mins, ind_maxes):

        logging.info('Start cleaning input data ...')

        logging.debug('Shape of dirtydata is %s' % str(dirtydata.shape))
        logging.debug('Shape of dirtydate is %s' % str(len(dirtydates)))
        logging.debug('ind_mins is: %s' % str(ind_mins))
        logging.debug('ind_maxes is: %s' % str(ind_maxes))
        logging.debug('mean of ind_min is: %d' % np.mean(ind_mins))
        logging.debug('mean of ind_max is: %d' % np.mean(ind_maxes))

        daylength = dirtydata.shape[1]

        # calculate number of valid data each day
        n_valid_data = ind_maxes - ind_mins + 1
        # mask = n_valid_data >= self.n_samples_per_day
        # mask = n_valid_data >= self.n_samples_per_day
        mask = (ind_mins <= self.ubd_min) & (ind_maxes >= self.lbd_max)

        # selected days with enough data each day
        tmp = dirtydata[mask, :]
        ind_mins = ind_mins[mask]
        ind_maxes = ind_maxes[mask]
        dates = [dirtydates[ind] for ind, val in enumerate(mask.tolist()) if val]

        logging.debug('maximum ind_mins is: %d' % np.max(ind_mins))
        logging.debug('minimum ind_maxes is: %d' % np.min(ind_maxes))

        # select common (part) features
        tmp = tmp[:-1, np.max(ind_mins) : (np.min(ind_maxes)+1)]

        logging.info('Input data is cleaned.')

        return tmp, dates
