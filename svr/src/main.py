'''
Source code for project solar forecast
'''

import logging
import numpy as np
import utils as utl
import matplotlib.pyplot as plt

from solar_forecast import *
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def main(configs) :

    # read mat data from file
    input_data = utl.read_mat(configs['DATA_PATH'])

    # data preprocessing
    input_data, proc_mask = utl.data_preprocessing(input_data,\
            configs['MONTH_SELECTION'])

    # generate feature vectors
    feats, labels = generate_features(input_data)

    # backup feats and labels
    feats_backup = feats
    labels_backup = labels

    # weather classification
    feats, labels, masks = weather_classification(feats, configs['MODE'], labels)

    if configs['MODE'] == 'grid search':

        grid_search_wrapper(feats, labels, configs)

    elif configs['MODE'] == 'holdout training':

        holdout_train_wrapper(feats, labels, configs, masks)

    elif configs['MODE'] == 'weather prediction':

        preds = weather_prediction(feats, labels, configs, masks)

        # compare predicted irradiance drop
        utl.plot_irradiance_drop(feats_backup[:,5] - preds, feats_backup[:,5] -
                labels_backup)
        utl.plot_irradiance_drop(preds, labels_backup)

        ''' regroup the data '''
        preds_cube, labels_cube = utl.regroup_data(preds, labels_backup, proc_mask)
        utl.compare_daily_mean(preds_cube, labels_cube, sensor_selection=24)
        plt.show()

if __name__ == '__main__' :

    # logging set up
    logging.basicConfig(filename='solar_forecast.log', filemode='w',
        level=logging.DEBUG,format='%(asctime)s %(levelname)s %(message)s')

    # path set up
    data_folder = '../data/'
    data_name = 'CESM_for_SVM.mat'
    # data_name = 'CESM_for_SVM_for_test.mat'
    fig_folder = '../figs/'

    # path to the optimal model under each weather
    # sunny_path = 'sunny-grid-search'
    # cloudy_path = 'cloudy-grid-search'
    # party_cloudy_path = 'partly_cloudy-grid-search'
    sunny_path = 'sunny-holdout'
    cloudy_path = 'cloudy-holdout'
    party_cloudy_path = 'partly_cloudy-holdout'

    # path to figure
    sunny_fig_path = fig_folder + 'sunny_pred_results'
    cloudy_fig_path = fig_folder + 'cloudy_pred_results'
    partly_cloudy_fig_path = fig_folder + 'partly_cloudy_pred_results'

    # consts SET PARAMETERS HERE !!!!!!!!!!
    DATA_PATH = data_folder + data_name
    FLAG = {'show_figs' : False}
    # MODE = 'grid search'
    # MODE = 'holdout training'
    MODE = 'weather prediction'
    MONTH_SELECTION = np.array([6,7,8])
    SUNNY_PARA = {'name': 'sunny', \
                  'test_size' : 0.2, \
                  'kernel' : 'rbf', \
                  'c' : 2e5, \
                  'epsilon' : 1, \
                  'gamma' : 2e-4, \
                  'k_for_kfold' : 5, \
                  'c_pool' : [2e4, 2e3, 2e2, 2e1, 2e0, 2e-1, 2e-2, 2e-3, 2e-4], \
                  'epsilon_pool' : [1.0], \
                  'gamma_pool' : [2e3, 2e2, 2e1, 2e0, 2e-1, 2e-2, 2e-3, 2e-4], \
                  'fig_path' : sunny_fig_path, \
                  'para_path' : sunny_path}

    PARTLY_CLOUDY_PARA = {'name' : 'partly_cloudy', \
                          'test_size' : 0.2, \
                          'kernel' : 'rbf', \
                          'c' : 2e5, \
                          'epsilon' : 1.0, \
                          'gamma' : 2e-5, \
                          'k_for_kfold' : 5, \
                          'c_pool' : [2e3, 2e4, 2e5], \
                          'epsilon_pool' : [1.0], \
                          'gamma_pool' : [2e-4, 2e-5, 2e-3], \
                          'fig_path' : partly_cloudy_fig_path, \
                          'para_path' : party_cloudy_path}


    CLOUDY_PARA = {'name' : 'cloudy', \
                   'test_size' : 0.2, \
                   'kernel' : 'rbf', \
                   'c' : 2e5, \
                   'epsilon' : 1.0, \
                   'gamma' : 2e-5, \
                   'k_for_kfold' : 5, \
                   'c_pool' : [2e-5, 2e-4], \
                   'epsilon_pool' : [1.0], \
                   'gamma_pool' : [2e1, 2e2], \
                   'fig_path' : cloudy_fig_path, \
                   'para_path' : cloudy_path}

    CONFIGS = {'data_name' : data_name, \
               'data_folder' : data_folder, \
               'fig_folder' : fig_folder, \
               'DATA_PATH' : DATA_PATH, \
               'MONTH_SELECTION': MONTH_SELECTION, \
               'MODE' : MODE, \
               'FLAG' : FLAG, \
               'SUNNY' : SUNNY_PARA, \
               'CLOUDY' : CLOUDY_PARA, \
               'PARTLY CLOUDY' : PARTLY_CLOUDY_PARA}

    # main()
    main(CONFIGS)
