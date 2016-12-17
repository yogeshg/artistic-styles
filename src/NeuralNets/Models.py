"""
Source Code for Homework 3 of ECBM E4040, Fall 2016, Columbia University

Instructor: Prof. Zoran Kostic

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""
import sys, os
try:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
except:
    pass

import logging
from collections import defaultdict

import numpy
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.signal import pool

from Utils import shared_dataset, load_data, RmsProp, train_nn
from Utils import about
from Layers import LogisticRegression, HiddenLayer, LeNetConvLayer, DropoutHiddenLayer, drop
from ImportParameters import load_layer_params



class VGG_19():
    def __init__(self, rng, datasets, filter_shape, batch_size=1, learning_rate=0.1,
                    weights=None,bias=None,image_size=(3,224,224)):
        self.model_name = "VGG_ILSVRC_19_layers"
        self.layer_names = ["conv1_1","conv1_2","pool1","conv2_1","conv2_2","pool2"
                                ,"conv3_1","conv3_2","conv3_3","conv3_4","pool3","conv4_1","conv4_2","conv4_3","conv4_4","pool4"
                                ,"conv5_1","conv5_2","conv5_3","conv5_4","pool5","fc6","drop6","fc7","drop7","fc8","prob"]
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug('Initializing...')

        if( (weights is None) or (bias is None) ):
            weights = {}
            bias = {}
            for name in self.layer_names:
                weights[name]=None
                bias[name]=None

        d,w,h=image_size

        x = T.matrix('x')  # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels
        self.eval_sample = {x: np.random.random((batch_size, d*w*h)).astype(np.float32)}

        layer0_input = x.reshape((batch_size, d, w, h))

        self.conv1_1 = LeNetConvLayer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv1_1'],
            W_values=weights['conv1_1'],
            b_values=bias['conv1_1']
        )
        self.conv1_1_sample = self.conv1_1.output.eval( self.eval_sample )
        self.logger.debug('self.conv1_1_sample:'+about(self.conv1_1_sample))

        #new image dimensions
        d = filter_shape['conv1_1'][0]

        self.conv1_2 = LeNetConvLayer(
            rng,
            input=self.conv1_1.output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv1_2'],
            W_values=weights['conv1_2'],
            b_values=bias['conv1_2']
        )
        self.conv1_2_sample = self.conv1_2.output.eval( self.eval_sample )
        self.logger.debug('self.conv1_2_sample:'+about(self.conv1_2_sample))

        #new image dimensions
        d = filter_shape['conv1_2'][0]
        pool1_output = pool.pool_2d(
                input=self.conv1_2.output,
                ds=(2,2),
                ignore_border=True
            )

        #new image dimensions
        w = w/2
        h = h/2

        self.conv2_1 = LeNetConvLayer(
            rng,
            input=pool1_output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv2_1'],
            W_values=weights['conv2_1'],
            b_values=bias['conv2_1']
        )
        self.conv2_1_sample = self.conv2_1.output.eval( self.eval_sample )
        self.logger.debug('self.conv2_1_sample:'+about(self.conv2_1_sample))

        d = filter_shape['conv2_1'][0]

        self.conv2_2 = LeNetConvLayer(
            rng,
            input=self.conv2_1.output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv2_2'],
            W_values=weights['conv2_2'],
            b_values=bias['conv2_2']
        )
        self.conv2_2_sample = self.conv2_2.output.eval( self.eval_sample )
        self.logger.debug('self.conv2_2_sample:'+about(self.conv2_2_sample))

        d = filter_shape['conv2_2'][0]

        pool2_output   = pool.pool_2d(
                input=self.conv2_2.output,
                ds=(2,2),
                ignore_border=True
            )

        w = w/2
        h = h/2

        self.conv3_1 = LeNetConvLayer(
            rng,
            input=pool2_output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv3_1'],
            W_values=weights['conv3_1'],
            b_values=bias['conv3_1']
        )
        self.conv3_1_sample = self.conv3_1.output.eval( self.eval_sample )        #neweval_sample dimensions
        self.logger.debug('self.conv3_1_sample:'+about(self.conv3_1_sample))

        d = filter_shape['conv3_1'][0]

        self.conv3_2 = LeNetConvLayer(
            rng,
            input=self.conv3_1.output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv3_2'],
            W_values=weights['conv3_2'],
            b_values=bias['conv3_2']
        )
        self.conv3_2_sample = self.conv3_2.output.eval( self.eval_sample )
        self.logger.debug('self.conv3_2_sample:'+about(self.conv3_2_sample))

        #new image dimensions
        d = filter_shape['conv3_2'][0]

        self.conv3_3 = LeNetConvLayer(
            rng,
            input=self.conv3_2.output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv3_3'],
            W_values=weights['conv3_3'],
            b_values=bias['conv3_3']
        )
        self.conv3_3_sample = self.conv3_3.output.eval( self.eval_sample )
        self.logger.debug('self.conv3_3_sample:'+about(self.conv3_3_sample))

        d = filter_shape['conv3_3'][0]

        self.conv3_4 = LeNetConvLayer(
            rng,
            input=self.conv3_3.output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv3_4'],
            W_values=weights['conv3_4'],
            b_values=bias['conv3_4']
        )
        self.conv3_4_sample = self.conv3_4.output.eval( self.eval_sample )
        self.logger.debug('self.conv3_4_sample:'+about(self.conv3_4_sample))

        d = filter_shape['conv3_4'][0]

        pool3_output   = pool.pool_2d(
                input=self.conv3_4.output,
                ds=(2,2),
                ignore_border=True
            )

        #new image dimensions
        w = w/2
        h = h/2

        self.logger.debug('filter_shape[\'conv4_1\']:' + str(filter_shape['conv4_1']))
        self.conv4_1 = LeNetConvLayer(
            rng,
            input=pool3_output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv4_1'],
            W_values=weights['conv4_1'],
            b_values=bias['conv4_1']
        )
        self.conv4_1_sample = self.conv4_1.output.eval( self.eval_sample )
        self.logger.debug('self.conv4_1_sample:'+about(self.conv4_1_sample))

        #new image dimensions
        d = filter_shape['conv4_1'][0]

        self.conv4_2 = LeNetConvLayer(
            rng,
            input=self.conv4_1.output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv4_2'],
            W_values=weights['conv4_2'],
            b_values=bias['conv4_2']
        )
        self.conv4_2_sample = self.conv4_2.output.eval( self.eval_sample )
        self.logger.debug('self.conv4_2_sample:'+about(self.conv4_2_sample))

        #new image dimensions
        d = filter_shape['conv4_2'][0]

        self.conv4_3 = LeNetConvLayer(
            rng,
            input=self.conv4_2.output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv4_3'],
            W_values=weights['conv4_3'],
            b_values=bias['conv4_3']
        )
        self.conv4_3_sample = self.conv4_3.output.eval( self.eval_sample )
        self.logger.debug('self.conv4_3_sample:'+about(self.conv4_3_sample))

        d = filter_shape['conv4_3'][0]

        self.conv4_4 = LeNetConvLayer(
            rng,
            input=self.conv4_3.output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv4_4'],
            W_values=weights['conv4_4'],
            b_values=bias['conv4_4']
        )
        self.conv4_4_sample = self.conv4_4.output.eval( self.eval_sample )
        self.logger.debug('self.conv4_4_sample:'+about(self.conv4_4_sample))

        #new image dimensions
        d = filter_shape['conv4_4'][0]

        pool4_output   = pool.pool_2d(
                input=self.conv4_4.output,
                ds=(2,2),
                ignore_border=True
            )

        #new image dimensions
        w = w/2
        h = h/2

        self.logger.debug('filter_shape[\'conv5_1\']:' + str(filter_shape['conv5_1']))
        self.conv5_1 = LeNetConvLayer(
            rng,
            input=pool4_output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv5_1'],
            W_values=weights['conv5_1'],
            b_values=bias['conv5_1']
        )
        self.conv5_1_sample = self.conv5_1.output.eval( self.eval_sample )
        self.logger.debug('self.conv5_1_sample:'+about(self.conv5_1_sample))

        #new image dimensions
        d = filter_shape['conv5_1'][0]

        self.conv5_2 = LeNetConvLayer(
            rng,
            input=self.conv5_1.output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv5_2'],
            W_values=weights['conv5_2'],
            b_values=bias['conv5_2']
        )
        self.conv5_2_sample = self.conv5_2.output.eval( self.eval_sample )
        self.logger.debug('self.conv5_2_sample:'+about(self.conv5_2_sample))

        #new image dimensions
        d = filter_shape['conv5_2'][0]

        self.conv5_3 = LeNetConvLayer(
            rng,
            input=self.conv5_2.output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv5_3'],
            W_values=weights['conv5_3'],
            b_values=bias['conv5_3']
        )
        self.conv5_3_sample = self.conv5_3.output.eval( self.eval_sample )
        self.logger.debug('self.conv5_3_sample:'+about(self.conv5_3_sample))

        #new image dimensions
        d = filter_shape['conv5_3'][0]

        self.conv5_4 = LeNetConvLayer(
            rng,
            input=self.conv5_3.output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv5_4'],
            W_values=weights['conv5_4'],
            b_values=bias['conv5_4']
        )
        self.conv5_4_sample = self.conv5_4.output.eval( self.eval_sample )
        self.logger.debug('self.conv5_4_sample:'+about(self.conv5_4_sample))

        #new image dimensions
        d = filter_shape['conv5_4'][0]

        pool5_output   = pool.pool_2d(
                input=self.conv5_4.output,
                ds=(2,2),
                ignore_border=True
            )

        #new image dimensions
        w = w/2
        h = h/2

        fc6_input = pool5_output.flatten(2)

        self.fc6_input_sample = fc6_input.eval( self.eval_sample )
        self.logger.debug('self.fc6_input_sample:'+about(self.fc6_input_sample))
        get_reshape = lambda sh: (sh[0],np.prod( sh[1:] ))
        flatten_2 = lambda m: m.reshape(get_reshape(m.shape))

        self.fc6     = HiddenLayer(
            rng,
            input=fc6_input,
            n_in=int(d * w * h),
            n_out=4096,
            activation=T.nnet.relu,
            W_values=flatten_2(weights['fc6']).T,
            b_values=flatten_2(bias['fc6'])[:,0]
        )
        self.fc6_sample = self.fc6.output.eval( self.eval_sample )
        self.logger.debug('self.fc6_sample:'+about(self.fc6_sample))

        self.drop6   = drop(self.fc6.output, p=0.5)

        self.fc7     = HiddenLayer(
            rng,
            input=self.drop6 ,
            n_in=4096,
            n_out=4096,
            activation=T.nnet.relu,
            W_values=flatten_2(weights['fc7']).T,
            b_values=flatten_2(bias['fc7'])[:,0]
        )
        self.fc7_sample = self.fc7.output.eval( self.eval_sample )
        self.logger.debug('self.fc7_sample:'+about(self.fc7_sample))

        self.drop7   = drop(self.fc7.output, p=0.5)

        self.fc8     = HiddenLayer(                         ## CHECK if Prob LogisticRegression can be used instead of HiddenLayer
            rng,                                            ## CHECK if Prob LogisticRegression can be used instead of HiddenLayer
            input=self.drop7 ,                              ## CHECK if Prob LogisticRegression can be used instead of HiddenLayer
            n_in=4096,                                      ## CHECK if Prob LogisticRegression can be used instead of HiddenLayer
            n_out=1000,                                     ## CHECK if Prob LogisticRegression can be used instead of HiddenLayer
            activation=None,                                ## CHECK if Prob LogisticRegression can be used instead of HiddenLayer
            W_values=flatten_2(weights['fc8']).T,
            b_values=flatten_2(bias['fc8'])[:,0]
        )                                                   ## CHECK if Prob LogisticRegression can be used instead of HiddenLayer
        self.fc8_sample = self.fc8.output.eval( self.eval_sample )
        self.logger.debug('self.fc8_sample:'+about(self.fc8_sample))

        self.prob    = LogisticRegression(
            input=self.fc8.output,
            n_in=1000,
            n_out=1000,
            W_values=numpy.identity(1000).astype(np.float32),
            b_values=numpy.zeros(1000).astype(np.float32)
        )
        # the cost we minimize during training is the NLL of the model
        self.cost = self.prob.negative_log_likelihood(y)

        # create a function to compute the mistakes that are made by the model
        self.test_model = theano.function(
            [x,y],
            self.prob.errors(y),
            allow_input_downcast=True
        )
        print('Test model compiled...')

        self.validate_model = theano.function(
            [x,y],
            self.prob.errors(y),
            allow_input_downcast=True
        )
        print('Validate model compiled...')

        # create a list of all model parameters to be fit by gradient descent
        # params = np.sum([ (getattr(self, l, None)).params for l in self.layer_names ]);
        params = self.conv1_1.params+self.conv1_2.params+\
                self.conv2_1.params+self.conv2_2.params+\
                self.conv3_1.params+self.conv3_2.params+self.conv3_3.params+self.conv3_4.params+\
                self.conv4_1.params+self.conv4_2.params+self.conv4_3.params+self.conv4_4.params+\
                self.conv5_1.params+self.conv5_2.params+self.conv5_3.params+self.conv5_4.params+\
                self.fc6.params+self.fc7.params+self.fc8.params

        # create a list of gradients for all model parameters
        grads = T.grad(self.cost, params)

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
            ]

        self.train_model = theano.function(
            [x,y],
            self.cost,
            updates=updates,
            allow_input_downcast=True ## To allow float64 values to be changed to float32
        )

        self.x = x
        print('Train model compiled...')

    def __str__(self):
        layers_strs = [ l+":\t"+str(getattr(self, l, 'LAYER_NOT_SET')) for l in self.layer_names ]
        header_str = "{}:{} L2_sqr:{}".format(self.__class__.__name__, self.model_name, 1234 ) # self.L2_sqr.eval()
        return header_str+"\n\t"+("\n\t".join(layers_strs))
