import h5py
import logging
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sklearn.metrics import mean_squared_error

pickles_folder = '../pickles/'

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

pylab.rcParams.update(params)


def read_mat(path):
    logging.info("Loading data ...")
    data_dict = {}
    fin = h5py.File(path, 'r')
    dataset = fin.items()
    for data in dataset:
        if type(data[1]) is h5py.Dataset:
            data_dict[data[0]] = data[1].value

    logging.info("Data loaded.")
    return data_dict

def normalize_data(data, supremum):

    logging.info("Start normalizing data ...")

    max_data = np.max(data)
    min_data = np.min(data)
    data_normalized = supremum * (data - min_data) / (max_data - min_data)

    logging.info("Normalizing is complete ...")

    return data_normalized

def data_preprocessing(dataset, month_selection):

    logging.info("Start preprocessing data ...")

    # keep track of (day/night and month) mask
    masks = {}

    # generate mask that only keep samples whose toa_irradiance > 0
    toa_mask = dataset['toa_irradiance'] > 0
    masks['toa mask'] = toa_mask

    # generate mask that only keep samples in selected months
    month_mask = np.zeros(dataset['month'].shape, dtype=bool)

    for month in month_selection :
        month_mask = month_mask | (dataset['month'] == month)

    masks['month mask'] = month_mask
    # sample mask
    preprocess_mask = toa_mask & month_mask
    masks['proc mask'] = preprocess_mask

    # only keep samples satisfying our requirements
    processed_data = {}
    exclude_list = ['day','month','hour']
    for name, data in dataset.items():
        if name not in exclude_list:
            processed_data[name] = data[preprocess_mask]
            logging.debug("{} has shape: {}".format(name, data.shape))

    # data rescaling
    processed_data['air_temperature'] = \
            normalize_data(processed_data['air_temperature'], 1)

    processed_data['relative_humidity'] = \
            normalize_data(processed_data['relative_humidity'], 1)

    processed_data['wind'] = \
            normalize_data(processed_data['wind'], 1)

    logging.info("Preprocessing complete.")

    return processed_data, masks

def regroup_data(preds, labels, masks, data_format=None):
    logging.info("Start regrouping data ...")

    # get size
    data_size = masks['toa mask'].shape

    # initialize new data vector
    labels_vec = np.zeros((data_size[0], data_size[1]))
    labels_vec[masks['proc mask']] = labels
    labels_vec[~masks['proc mask']] = np.nan

    preds_vec = np.zeros((data_size[0], data_size[1]))
    preds_vec[masks['proc mask']] = preds
    preds_vec[~masks['proc mask']] = np.nan

    # remove data entries for excluded months
    # reshape data vector into a data cube with shape (25 x 24 x days)
    # labels_vec = labels_vec[masks['month mask']].reshape((-1, 25, 24))
    # preds_vec = preds_vec[masks['month mask']].reshape((-1, 25, 24))

    labels_vec = labels_vec[masks['month mask']].reshape((-1, 24, 25))
    preds_vec = preds_vec[masks['month mask']].reshape((-1, 24, 25))

    # for debugging
    logging.debug('the shape of preds_vec is '.format(preds_vec.shape))

    logging.info("data regrouping is complete.")

    return preds_vec, labels_vec

def compare_daily_mean(preds_cube, labels_cube, sensor_selection=0):

    logging.info("Start comparing predicted daily mean and measured daily mean")

    # calculate daily mean of each sensor (nans are excluded)
    labels_mean = np.nanmean(labels_cube, axis=2)
    preds_mean = np.nanmean(preds_cube, axis=2)

    logging.debug('the shape of labels_mean is {}'.format(labels_mean.shape))
    logging.debug('the shape of preds_mean is {}'.format(preds_mean.shape))
    logging.info("daily mean computed.")

    selected_mean_preds = preds_mean[:,sensor_selection].flatten()
    selected_mean_labels = labels_mean[:,sensor_selection].flatten()

    print 'selected_mean_labels:\n', selected_mean_labels
    print 'selected_mean_preds:\n', selected_mean_preds

    print 'mean absolute error:\n', np.sum(np.absolute(\
            selected_mean_preds - selected_mean_labels)) / len(selected_mean_labels)

    rmse = compute_error(selected_mean_preds, selected_mean_labels)

    logging.info("Average daily mean error is {}".format(rmse))
    print "Average daily mean error is {}".format(rmse)

    fig1 = compare_pred_results(selected_mean_preds,\
            selected_mean_labels, 'daily_mean')
    fig2 = plot_preds_labels(selected_mean_preds,\
            selected_mean_labels)

    return fig1, fig2

def compute_error(y_pred, y_test):
    logging.info("Start computing prediction error ...")
    err = mean_squared_error(y_test, y_pred)
    logging.info("Prediction error computed.")
    return np.sqrt(err)

def compare_pred_results(y_pred, y_test, var_name, style=None, axis_min=None, axis_max=None):
    logging.info("Start comparing prediction results with true results ...")

    # get the maximum of y_test and y_pred
    if axis_max is None :
        axis_max = np.max([np.max(y_pred), np.max(y_test)])

    if axis_min is None :
        axis_min = np.min([np.min(y_pred), np.min(y_test)])

    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect('equal', adjustable='box')

    # plot function f(x) = x
    ax.plot([axis_min, axis_max], [axis_min, axis_max], 'r--',\
            label='reference line')

    # plot prediction results and true labels
    if style is not None :
        ax.plot(y_test, y_pred, style, label='prediction vs true labels')
    else :
        ax.plot(y_test, y_pred, 'b.', label='prediction vs true labels')
    plt.title("Comparison between predicted {} and true labels"\
            .format(var_name))
    plt.xlabel("True labels")
    plt.ylabel("Prediction results")
    plt.legend(loc=2)

    logging.info("Comparison generated.")
    return fig

def compare_preds_labels(preds, labels, masks, style='k-*'):
    logging.info("Start plotting prediction results and measurements")

    # combine all the predictions
    preds_total = np.zeros(masks['sunny'].shape)
    preds_total[masks['sunny']] = preds['sunny']
    preds_total[masks['cloudy']] = preds['cloudy']
    preds_total[masks['partly_cloudy']] = preds['partly_cloudy']

    # combine all labels
    labels_total = np.zeros(masks['sunny'].shape)
    labels_total[masks['sunny']] = labels['sunny']
    labels_total[masks['cloudy']] = labels['cloudy']
    labels_total[masks['partly_cloudy']] = labels['partly_cloudy']

    fig = plot_preds_labels(preds_total, labels_total, style)

    logging.info("Plot generated")

    return fig, preds_total

def plot_preds_labels(preds, labels, style='k-*'):

    fig = plt.figure()
    ax = fig.gca()
    ax.set_ylim(0,1000)
    plt.plot(preds, style, label='predicted results')
    plt.plot(labels, 'r-o', label='measured results')

    plt.title("predicted value and measured value")
    plt.xlabel("Time")
    plt.ylabel("Function value")
    plt.legend(loc=2)

    return fig

def plot_irradiance_drop(preds, meas, style='k-*'):
    logging.info("Start plotting predicted irradiance drop and measured \
        irradiance drop")

    fig = plt.figure()
    ax = fig.gca()
    ax.set_ylim(0,1000)
    plt.plot(preds, style, label='predicted irradiance drop')
    plt.plot(meas, 'r-o', label='measured irradiance drop')

    plt.title("predicted irradiance drop and measured irradiance drop")
    plt.xlabel("Time (h)")
    plt.ylabel("Irradiance drop")
    plt.legend(loc=2)

    logging.info("Predicted irradiance drop and measured irradiance drop are \
        plotted")

    return fig

def save_model(var,name):
    logging.info('Saving model on disk ...')
    with open(pickles_folder + name + '.pickle', 'wb') as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info('Model saved.')


def restore_model(name):
    logging.info('Recovering model, {} on disk ...'.format(name))
    val = None
    with open(pickles_folder + name + '.pickle', 'rb') as handle:
        val = pickle.load(handle)
    logging.info('Model recovered.')

    return val

def print_dict(dictionary):
    for key, val in dictionary.items():
        print str(key) + ':' + str(val) + '\n'

'''
if __name__ == '__main__':
    #------------------------------------------------------------
    # Function Test
    #------------------------------------------------------------
    path = './data/CESM_for_SVM.mat'
    data = read_mat(path)
    month = np.array(data['month'])
    day = np.array(data['day'])

    _, ax = plt.subplots()
    plt.plot(month, 'bo')
    plt.plot(day, 'ro')
    ax.grid(True)
    plt.show()
'''

