__docformat__ = 'restructedtext en'

import logging
logger = logging.getLogger(__name__)

import os
import sys
import timeit

import numpy
import scipy.io

def load_layer_params(params_file):
    ''' Loads the imagenet weights and bias according to name

    This function is modified from load_data in
    http://deeplearning.net/tutorial/code/logistic_sgd.py

    Author Francis Marcogliese fam2148

    '''

    #############
    # LOAD DATA #
    #############

    # Download the SVHN dataset if it is not present
    def check_dataset(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.getcwd(),
            "data",
            dataset
        )
        if (not os.path.isfile(new_path)):
            from six.moves import urllib
            origin = (
                'http://www.vlfeat.org/matconvnet/models/' + params_file
            )
            logger.info('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, new_path)
        return new_path

    params_file_path = check_dataset(params_file)

    logger.info('... loading params')

    # Load the dataset
    params_dataset = scipy.io.loadmat(params_file_path, variable_names="layers")
    #logger.info (numpy.array(weight_set['layers'][0][0][0][0]).shape)
    names = []
    types = {}
    weights = {}
    bias = {}
    filter_shape = {}
    pool_shape = {}
    for layer in params_dataset['layers'][0]:
        layer_name = layer[0][0][0][0].encode('ascii')
        names.append(layer_name)
        types[layer_name]= layer[0][0][1][0].encode('ascii')
        try:
            weights[layer_name] = layer[0][0][2][0][0].T
        except:
            weights[layer_name] = None
        try:
            bias[layer_name] = (layer[0][0][2][0][1].T)[0]
        except:
            bias[layer_name] = None
        try:
            filter_shape[layer_name] = tuple(numpy.fliplr(layer[0][0][3])[0].reshape(1,-1)[0])
        except:
            filter_shape[layer_name] = None
        try:
            pool_shape[layer_name] = tuple(layer[0][0][4][0].reshape(1,-1)[0])
        except:
            pool_shape[layer_name] = None

    # return names,types,weights,bias,filter_shape,pool_shape
    return { 'names': names, 'types': types, 'weights': weights, 'bias': bias, 'filter_shape': filter_shape, 'pool_shape': pool_shape }

