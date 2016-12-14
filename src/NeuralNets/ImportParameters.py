
from __future__ import print_function

__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy
import scipy.io

def load_data(current_Layer):
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
            os.path.split(__file__)[0],
            "data",
            dataset
        )
        if (not os.path.isfile(new_path)):
            from six.moves import urllib
            origin = (
                'http://www.vlfeat.org/matconvnet/models/' + dataset
            )
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, new_path)
        return new_path

    weight_dataset = check_dataset('imagenet-vgg-verydeep-19.mat')


    print('... loading data')

    # Load the dataset
    weight_set = scipy.io.loadmat(weight_dataset, variable_names="layers")
    #print (numpy.array(weight_set['layers'][0][0][0][0]).shape)

    for i in range(len(weight_set['layers'][0])):
        layer_name = weight_set['layers'][0][i][0][0][0][0]
        if current_Layer == layer_name:
            break

    print (layer_name)
    W = weight_set['layers'][0][i][0][0][2][0][0]
    b = weight_set['layers'][0][i][0][0][2][0][1]
    print (numpy.array(W).shape)
    print (numpy.array(b).shape)


    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.
    '''
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    '''
    print (W)
    #print (b)
    return W, b

load_data("conv1_1")