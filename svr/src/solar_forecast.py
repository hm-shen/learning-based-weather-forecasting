import logging
import numpy as np
import utils as utl
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def generate_features(dataset):

    logging.info("generating features ...")

    # init features
    data_len = dataset['total_cloud_fraction'].shape[0]
    features = np.zeros([data_len,8])
    index = 0 # column selection index

    # generate features
    exclude_list = ['day','month','hour', 'surface_irradiance']
    for name, data in sorted(dataset.iteritems()):
        if name not in exclude_list:
            logging.debug("processing {}th data: {} ...".format((index+1),name))
            features[:,index] = np.squeeze(data)
            feat_max = np.max(features[:,index])
            feat_min = np.min(features[:,index])
            logging.debug("The {}th data has max {} and min {}"\
                    .format((index+1), feat_max, feat_min))
            index += 1

    logging.debug("features have size {}".format(features.shape))
    logging.info("features generated ...")

    # if the dataset is used for test, it may not have surface irradiance info
    if 'surface_irradiance' in dataset.keys():
        # generating labels
        logging.info("generating labels ...")

        # labels = dataset['toa_irradiance'] - dataset['surface_irradiance']
        # labels = utl.normalize_data(labels, 500)
        labels = dataset['surface_irradiance']

        logging.debug("the shape of label vector is {}".format(labels.shape))
        logging.info("labels generated ...")
    else :
        labels = None

    return features, labels

def weather_classification(feats, MODE, labels=None):

    logging.info("Start classifying weather types ...")

    '''
    Classify weather types in a hard way:
    '''
    # SUNNY_THRESHOLD = 0.05
    # CLOUDY_THRESHOLD = 0.95

    # sunny_mask = feats[:,6] <= SUNNY_THRESHOLD
    # cloudy_mask = feats[:,6] >= CLOUDY_THRESHOLD
    # partly_cloudy_mask = ~ (sunny_mask | cloudy_mask)

    '''
    Use K-means algorithm to classify different weather types automatically:
    '''

    if (MODE == 'grid search' or MODE == 'holdout training') :
        kmeans = KMeans(n_clusters=3, random_state=0).fit(feats[:,[0,1,2,3,4,6,7]])
        cluster_center = kmeans.cluster_centers_
        ''' sort according to cloud fraction '''
        cluster_center = cluster_center[cluster_center[:,5].argsort()]
        print 'clustering centers are:\n', cluster_center
        utl.save_model(kmeans, 'weather-classification_k-means')

        sunny_mask = kmeans.labels_ == 0
        partly_cloudy_mask = kmeans.labels_ == 1
        cloudy_mask = kmeans.labels_ == 2

    elif (MODE == 'weather prediction') :
        kmeans = utl.restore_model('weather-classification_k-means')
        cluster_center = kmeans.cluster_centers_
        predicted_label = kmeans.predict(feats[:,[0,1,2,3,4,6,7]])
        sort_ind = cluster_center[:,5].argsort()
        print 'sort ind is:', sort_ind
        # dictionary = dict(zip(keys, values))

        sunny_mask = predicted_label == sort_ind[0]
        partly_cloudy_mask = predicted_label == sort_ind[1]
        cloudy_mask = predicted_label == sort_ind[2]

    mask_dict = {'sunny': sunny_mask,
                 'cloudy': cloudy_mask,
                 'partly_cloudy': partly_cloudy_mask}

    feats_dict = {'sunny': feats[sunny_mask,:],
                  'cloudy': feats[cloudy_mask,:],
                  'partly_cloudy': feats[partly_cloudy_mask,:]}

    for key, val in feats_dict.items():
        logging.debug('{} feature has shape: {}'.format(key,val.shape))

    if labels is not None :
        labels_dict = {'sunny': labels[sunny_mask],
                       'cloudy': labels[cloudy_mask],
                       'partly_cloudy': labels[partly_cloudy_mask]}

        for key, val in labels_dict.items():
            logging.debug('{} label has shape: {}'.format(key,val.shape))
    else :
        labels_dict = None

    logging.info("Weather types classified.")

    return feats_dict, labels_dict, mask_dict

def grid_search(x_train, y_train, parameters):

    logging.info("Start grid searching ...")

    # parsing grid_search config
    num_of_folds = parameters['k_for_kfold']
    gs_config = [{'kernel': [parameters['kernel']],
                  'epsilon': parameters['epsilon_pool'],
                  'C': parameters['c_pool'],
                  'gamma': parameters['gamma_pool']}]

    svr = svm.SVR()
    clf = GridSearchCV(svr, gs_config, cv=num_of_folds, \
            scoring='neg_mean_squared_error', n_jobs=6)
    clf.fit(x_train, y_train)

    logging.info("The best parameters set found on development set: {}"\
            .format(clf.best_params_))
    logging.info("The correspnding score is {}".format(clf.best_score_))
    print("The best parameters set found on development set: {}"\
            .format(clf.best_params_))
    print("The correspnding score is {}".format(clf.best_score_))

    logging.info("Saving the best model on disk ...")
    utl.save_model(clf.best_estimator_, parameters['name'] + '-grid-search')

    return clf

def holdout_train(feats, labels, parameters):

    logging.info("Start holdout training ...")

    x_train,x_test,y_train,y_test = train_test_split(feats, labels, \
                                        test_size=parameters['test_size'])

    logging.debug("Dimension of training set: {}".format(x_train.shape))

    clf = svm.SVR(kernel=parameters['kernel'], C=parameters['c'], \
                    epsilon=parameters['epsilon'], gamma=parameters['gamma'])
    clf.fit(x_train, y_train)

    logging.info("Holdout training is complete.")

    y_pred = clf.predict(x_test)
    rmse = utl.compute_error(y_pred, y_test)

    logging.info("Saving the trained model on disk ...")
    utl.save_model(clf, parameters['name'] + '-holdout')

    return y_pred, y_test, rmse

def predict(x_test, model):
    logging.info("Start perdicting ...")
    y_pred = model.predict(x_test)
    logging.info("Prediction complete...")
    return y_pred

def grid_search_wrapper(feats, labels, parameters) :
    ''' grid search wrapper for solar forecast '''

    # grid search for each weather

    # sunny day
    logging.info("Start grid search on sunny days ...")
    grid_search(feats['sunny'], labels['sunny'], parameters['SUNNY'])
    logging.info("Grid search on sunny days is complete.")

    # partly cloudy day
    # logging.info("Start grid search on partly cloudy days ...")
    # grid_search(feats['partly_cloudy'], labels['partly_cloudy'], \
    #                 parameters['PARTLY CLOUDY'])
    # logging.info("Grid search on partly cloudy days is complete.")

    # cloudy day
    # logging.info("Start grid search on cloudy days ...")
    # grid_search(feats['cloudy'], labels['cloudy'], parameters['CLOUDY'])
    # logging.info("Grid search on cloudy days is complete.")

def holdout_train_wrapper(feats, labels, parameters, masks):
    ''' holdout training wrapper for solar forecast '''

    # perform holdout training on each weather

    # sunny day
    logging.info("Start holdout training on sunny days ...")
    # get prediction and true labels
    sunny_pred, sunny_test, sunny_errors = holdout_train(\
            feats['sunny'], labels['sunny'], parameters['SUNNY'])
    # write RMSE to log
    logging.info("RMSE errors of sunny days: {}".format(sunny_errors))
    print("RMSE errors of sunny days: {}".format(sunny_errors))
    # compare prediction results with true labels
    sunny_fig = utl.compare_pred_results(sunny_pred, sunny_test,\
            'sunny', style='b.')
    logging.info("Holdout training on sunny days is complete.")

    # partly cloudy day
    logging.info("Start holdout training on partly cloudy days ...")
    # get prediction and true labels
    partly_cloudy_pred, partly_cloudy_test, partly_cloudy_errors = \
            holdout_train(feats['partly_cloudy'],\
            labels['partly_cloudy'], parameters['PARTLY CLOUDY'])
    # write RMSE to log
    logging.info("RMSE errors of partly cloudy days: {}".\
            format(partly_cloudy_errors))
    print("RMSE errors of partly cloudy days: {}".\
            format(partly_cloudy_errors))
    # compare prediction results with true labels
    partly_cloudy_fig = utl.compare_pred_results(partly_cloudy_pred,\
            partly_cloudy_test, 'partly cloudy', style='b.')
    logging.info("Holdout training on partly cloudy days is complete.")

    # cloudy day
    logging.info("Start holdout training on cloudy days ...")
    # get prediction and true labels
    cloudy_pred, cloudy_test, cloudy_errors = holdout_train(\
            feats['cloudy'], labels['cloudy'], parameters['CLOUDY'])
    # write RMSE to log
    logging.info("RMSE errors of cloudy days: {}".format(cloudy_errors))
    print("RMSE errors of cloudy days: {}".format(cloudy_errors))
    # compare prediction results with true labels
    cloudy_fig = utl.compare_pred_results(cloudy_pred, cloudy_test,\
            'cloudy', style='b.')


    # save fig
    fig_path = parameters['SUNNY']['fig_path'] + '.png'
    sunny_fig.savefig(fig_path)

    fig_path = parameters['PARTLY CLOUDY']['fig_path'] + '.png'
    partly_cloudy_fig.savefig(fig_path)

    fig_path = parameters['CLOUDY']['fig_path'] + '.png'
    cloudy_fig.savefig(fig_path)

    if parameters['FLAG']['show_figs'] :
        plt.show()

    logging.info("Holdout training on cloudy days is complete.")

def weather_prediction(feats, labels, parameters, masks):
    ''' weather prediction wrapper for solar forecast '''

    # create results dictionary under each weather
    sunny_results = {}
    cloudy_results = {}
    partly_cloudy_results = {}

    # get parameters of SVR model under each weather types
    if parameters['SUNNY']['para_path'] is not None :
        logging.info('Loading given model on disk ...')
        sunny_model = utl.restore_model(\
                parameters['SUNNY']['para_path'])
    else :
        logging.info('Restore model on disk ...')
        sunny_model = utl.restore_model(\
                parameters['SUNNY']['name'] + '-grid-search')
    logging.info("Sunny model loaded is {}".format(sunny_model))

    if parameters['CLOUDY']['para_path'] is not None :
        logging.info('Loading given model on disk ...')
        cloudy_model = utl.restore_model(\
                parameters['CLOUDY']['para_path'])
    else :
        logging.info('Restore model on disk ...')
        cloudy_model = utl.restore_model(\
                parameters['CLOUDY']['name'] + '-grid-search')
    logging.info("Cloudy model loaded is {}".format(cloudy_model))

    if parameters['PARTLY CLOUDY']['para_path'] is not None :
        logging.info('Loading given model on disk ...')
        partly_cloudy_model = utl.restore_model(\
                parameters['PARTLY CLOUDY']['para_path'])
    else :
        logging.info('Restore model on disk ...')
        partly_cloudy_model = utl.restore_model(\
                parameters['PARTLY CLOUDY']['name'] + '-grid-search')
    logging.info("Partly cloudy model loaded is {}"\
                .format(partly_cloudy_model))

    # perform prediction based on weather types

    # sunny day
    logging.info("Start performing predictions on sunny days ...")
    sunny_pred = predict(feats['sunny'], sunny_model)
    sunny_results['prediction'] = sunny_pred
    logging.info("Weather prediction on sunny days is complete.")

    # cloudy day
    logging.info("Start performing predictions on cloudy days ...")
    cloudy_pred = predict(feats['cloudy'], cloudy_model)
    cloudy_results['prediction'] = cloudy_pred
    logging.info("Weather prediction on cloudy days is complete.")

    # partly cloudy day
    logging.info("Start performing predictions on partly cloudy days ...")
    partly_cloudy_pred = predict(feats['partly_cloudy'],partly_cloudy_model)
    partly_cloudy_results['prediction'] = partly_cloudy_pred
    logging.info("Weather prediction on partly cloudy days is complete.")

    if labels is not None :

        # computing errors
        sunny_errors = utl.compute_error(sunny_pred, labels['sunny'])
        sunny_results['rmse_errors'] = sunny_errors
        logging.info("RMSE errors of sunny days: {}".format(sunny_errors))

        cloudy_errors = utl.compute_error(cloudy_pred, labels['cloudy'])
        cloudy_results['rmse_errors'] = cloudy_errors
        logging.info("RMSE errors of cloudy days: {}".format(cloudy_errors))

        partly_cloudy_errors = utl.compute_error(partly_cloudy_pred,\
                                    labels['partly_cloudy'])
        partly_cloudy_results['rmse_errors'] = partly_cloudy_errors
        logging.info("RMSE errors of partly cloudy days: {}"\
                .format(partly_cloudy_errors))

        # comparison between prediction and true labels
        sunny_fig = utl.compare_pred_results(sunny_pred, labels['sunny'],\
                'sunny', style='k.')
        cloudy_fig = utl.compare_pred_results(cloudy_pred, labels['cloudy'],\
                'cloudy', style='k.')
        partly_cloudy_fig = utl.compare_pred_results(partly_cloudy_pred,\
                labels['partly_cloudy'], 'partly cloudy', style='k.')

        preds = {'sunny': sunny_pred, 'cloudy': cloudy_pred,\
                'partly_cloudy': partly_cloudy_pred}

        # plot predictions and measurements
        fig_pred_meas, preds_total = utl.compare_preds_labels(preds, labels, masks, )

        # save fig
        fig_path = parameters['SUNNY']['fig_path'] + '.png'
        sunny_fig.savefig(fig_path)

        fig_path = parameters['CLOUDY']['fig_path'] + '.png'
        cloudy_fig.savefig(fig_path)

        fig_path = parameters['PARTLY CLOUDY']['fig_path'] + '.png'
        partly_cloudy_fig.savefig(fig_path)

        fig_path = parameters['fig_folder'] + 'pred_vs_meas' + '.png'
        partly_cloudy_fig.savefig(fig_path)

        if parameters['FLAG']['show_figs'] :
            plt.show()

        return preds_total
