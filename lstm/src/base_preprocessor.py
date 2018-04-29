'''
Description: Preprocessor for weather data
Author: Haoming Shen
Date: 03/30/2018
'''

import sys
import logging
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import interpolate
from datetime import timedelta, date

def daterange(start_date, end_date):
    '''
    Description : loop through a whole year
    Source      : https://stackoverflow.com/questions/1060279/
                : iterating-through-a-range-of-dates-in-python
    '''
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)


class Base_Preprocessor(object):

    '''
    Description: Base class for preprocessor
    '''

    def __init__(self, outfmt):

        logging.info('Initializing preprocessor ...')

        # set output format
        self.outfmt=outfmt
        # predefined required keys for input data
        self.RKEYS = ['hour', 'day', 'month', 'year']
        # a set of flags denoting the existence of a key
        self.keyflags = {'sec': False, 'min': False,
                         'hour': False, 'day': False,
                         'month': False, 'year': False}

        logging.info('Preprocessor is initialized.')

    def load(self, data):

        logging.info('Start loading data ...')

        self.rawdata = data
        # input checking
        self._input_check()

        logging.info('Input data has been loaded.')

    def preprocessor(self, data):

        pass

    def _input_check(self):

        ''' Check input data format '''

        # input data should have been loaded
        assert self.rawdata is not None, 'ERROR: please load data first!'

        # input data should be a python dictionary
        assert isinstance(self.rawdata, dict),\
            'ERROR: input data should be a dict!'

        # set key flags
        for keys in self.RKEYS:
            assert keys in self.rawdata.keys(),\
                'ERROR: input data does not contain %s' % keys
            self.keyflags[keys] = True
            self.rawdata[keys] = self.rawdata[keys].flatten().astype(int)

        # check if min and sec exists in input data key
        for keys in ['sec', 'min']:
            if keys in self.rawdata.keys():
                self.keyflags[keys] = True
                self.rawdata[keys] = self.rawdata[keys].flatten().astype(int)

        # remove invalid keys
        for badkey in ['__header__', '__globals__', '__version__']:
            # check if badkey exist in rawdata.keys().
            if badkey in self.rawdata.keys():
                # if exist, then delete it.
                del self.rawdata[badkey]


    def _clean(self):

        pass

    def _post_process(self):

        pass

class WRF_Preprocessor(Base_Preprocessor):

    '''
    Description: Preprocessor for WRF data
    '''

    def preprocess(self, datakey):

        if self.outfmt == 'consecutive':

            self.rawdata[datakey] = self.rawdata[datakey].flatten()
            self.data = self.rawdata[datakey]
            self.dates = [ date(self.rawdata['year'][ind],
                                self.rawdata['month'][ind],
                                self.rawdata['day'][ind])
                           for ind in range(self.rawdata[datakey].shape[0])]

            return (self.data, self.dates)


class NREL_Preprocessor(Base_Preprocessor):

    '''
    Description: Preprocessor for NREL data
    '''
    def __init__(self, outfmt):

         super(NREL_Preprocessor, self).__init__(outfmt)

         # number of samples in each day
         self.daylength = None

    def load(self, data):

        super(NREL_Preprocessor, self).load(data)

        # data is not consecutive, divide it day by day
        self._cal_daylength()

    def load_and_preprocess(self, data, datakey):

        self.load(data)
        return self.preprocess(datakey)

    def preprocess(self, datakey):

        logging.info('Start processing input data ...')

        if self.outfmt == 'daybyday':
            self.data = self._to_daybyday(datakey)
            self.data = self._post_process()
            logging.info('Input data is processed.')
            return (self.data, self.dates, self.ind_mins, self.ind_maxes)
        else:
            logging.error('Invalid output format!')
            print 'Output format is invalid!'
            sys.exit(-1)

    def _cal_daylength(self):

        logging.info('Start calculating daylength for input data ...')

        ''' calculate daylength for input data '''

        if self.keyflags['sec']:
            # have sec by sec data
            self.daylength = 24 * 60 * 3600
        elif self.keyflags['min']:
            # have minute by minute data
            self.daylength = 24 * 60
        else:
            # only have hourly data
            self.daylength = 24

    def _to_daybyday(self, datakey):

        '''
        Description: organize data in a matrix with size n_days x n_samples
        Input:
        datakey: identifies the data to be processed in input dictionary.
        Ouput:
        daybyday data with form n_days x n_samples per day.
        '''

        # get starting date
        # input data should placed accord. order !!!!!
        logging.debug('Shape of rawdata[day] is: %s' % str(self.rawdata['day'][0]))
        logging.debug('Shape of rawdata[month] is: %s' % str(self.rawdata['month'][0]))

        self.rawdata[datakey] = self.rawdata[datakey].flatten()
        st = date(self.rawdata['year'][0], self.rawdata['month'][0],
                  self.rawdata['day'][0])

        ed = date(self.rawdata['year'][-1], self.rawdata['month'][-1],
                  self.rawdata['day'][-1])

        n_days = ed - st

        # initialize a daylen entry for each day
        tmp = np.full((1, self.daylength), np.nan)
        dcnt = 0       # date counter
        self.ind_mins = []  # keep track of the first data each day
        self.ind_maxes = [] # keep track of the last data each day
        self.dates = []

        # loop over dates from st to ed
        for today in daterange(st, ed + timedelta(days=1)):

            # get daily data
            mask = (self.rawdata['year'] == today.year) &\
                   (self.rawdata['month'] == today.month) &\
                   (self.rawdata['day'] == today.day)

            # if today exists in data and contains valid data
            if (np.count_nonzero(mask) > 0) & (np.sum(self.rawdata[datakey][mask] >= 0) > 0):

                # get daily data
                logging.debug('Today is: %s' % today)
                self.dates.append(today)
                daily_data = self.rawdata[datakey][mask]

                # place data in the correct position
                if self.keyflags['sec']:
                    indicies = (self.rawdata['sec'][mask] +
                                self.rawdata['min'][mask] * 60.0 +
                                self.rawdata['hour'][mask] * 3600.0).astype(int)
                elif self.keyflags['min']:
                    indicies = (self.rawdata['min'][mask] +
                                self.rawdata['hour'][mask] * 60.0).astype(int)
                else:
                    indicies = self.rawdata['hour'][mask].astype(int)

                # assign daily data to tmp
                logging.debug('shape of daily data is: %s' % str(daily_data.shape) )
                logging.debug('shape of indicies is: %s' % str(indicies.shape) )
                tmp[-1, indicies] = daily_data.reshape((1,-1))

                # clean daily data
                tmp[-1,:], ind_min, ind_max  = self._clean(tmp[-1,:])

                if ind_min is None:
                    print 'Void data!'
                    sys.exit(-1)
                    continue

                # keep track of the starting and end of daily data
                # indicies = np.uint32(indicies)
                self.ind_mins.append(ind_min)
                self.ind_maxes.append(ind_max)

                # apend another entry for the next day
                tmp = np.append(tmp, np.full((1, self.daylength), np.nan),
                                axis=0)
            dcnt += 1

        self.ind_mins = np.array(self.ind_mins)
        self.ind_maxes = np.array(self.ind_maxes)
        # logging.debug('ind_mins are: %s' % str(self.ind_mins))
        # logging.debug('ind_maxes are: %s' % str(self.ind_maxes))

        # return processed data
        return tmp[:-1,:]

    def _clean(self, raw):

        ''' replace all invalid data (< 0) with np.nan '''
        raw[raw < 0] = np.nan

        raw = raw.flatten()
        mask = ~np.isnan(raw)
        # indices of valid data
        valid_inds = np.array(\
            [ ind for ind, val in np.ndenumerate(mask) if val]).flatten()

        # ind_min = int(valid_inds[0])
        # ind_max = ind(valid_inds[-1])
        if valid_inds is not None:
            ind_min, ind_max = np.min(valid_inds), np.max(valid_inds)
        else:
            ind_min = None
            ind_max = None
        return raw, ind_min, ind_max

    def _post_process(self):

        ''' post process data '''
        assert self.data is not None, 'ERROR: please process rawdata first!'

        return self.data
