"""
Source Code for Homework 3 of ECBM E4040, Fall 2016, Columbia University

Instructor: Prof. Zoran Kostic

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""
import sys, os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

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
from Layers import LogisticRegression, HiddenLayer, LeNetConvLayer, DropoutHiddenLayer, drop
from ImportParameters import load_layer_params

from collections import defaultdict

class VGG_19():
    def __init__(self, rng, datasets, filter_shape, batch_size=10, learning_rate=0.1,
                    Weights=None,bias=None,image_size=(3,224,224)):
        self.model_name = "VGG_ILSVRC_19_layers"
        self.layer_names = ["conv1_1","conv1_2","pool1","conv2_1","conv2_2","pool2"
                                ,"conv3_1","conv3_2","conv3_3","conv3_4","pool3","conv4_1","conv4_2","conv4_3","conv4_4","pool4"
                                ,"conv5_1","conv5_2","conv5_3","conv5_4","pool5","fc6","drop6","fc7","drop7","fc8","prob"]
        if((Weights == None) or (bias == None)):
            Weights = {}
            bias = {}
            for name in self.layer_names:
                Weights[name]=None
                bias[name]=None

        d,w,h=image_size

        x = T.matrix('x')  # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels


        layer0_input = x.reshape((batch_size, d, w, h))

        self.conv1_1 = LeNetConvLayer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv1_1'],
            W=Weights['conv1_1'],
            b=bias['conv1_1']
        )

        #new image dimensions
        d = filter_shape['conv1_1'][0]

        self.conv1_2 = LeNetConvLayer(
            rng,
            input=self.conv1_1.output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv1_2'],
            W=Weights['conv1_2'],
            b=bias['conv1_2']
        )

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
            W=Weights['conv2_1'],
            b=bias['conv2_1']
        )

        d = filter_shape['conv2_1'][0]

        self.conv2_2 = LeNetConvLayer(
            rng,
            input=self.conv2_1.output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv2_2'],
            W=Weights['conv2_2'],
            b=bias['conv2_2']
        )

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
            W=Weights['conv3_1'],
            b=bias['conv3_1']
        )
        #new image dimensions
        d = filter_shape['conv3_1'][0]
        
        self.conv3_2 = LeNetConvLayer(
            rng,
            input=self.conv3_1.output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv3_2'],
            W=Weights['conv3_2'],
            b=bias['conv3_2']
        )

        #new image dimensions
        d = filter_shape['conv3_2'][0]

        self.conv3_3 = LeNetConvLayer(
            rng,
            input=self.conv3_2.output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv3_3'],
            W=Weights['conv3_3'],
            b=bias['conv3_3']
        )

        d = filter_shape['conv3_3'][0]

        self.conv3_4 = LeNetConvLayer(
            rng,
            input=self.conv3_3.output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv3_4'],
            W=Weights['conv3_4'],
            b=bias['conv3_4']
        )

        d = filter_shape['conv3_4'][0]

        pool3_output   = pool.pool_2d(
                input=self.conv3_4.output,
                ds=(2,2),
                ignore_border=True
            )

        #new image dimensions
        w = w/2
        h = h/2

        self.conv4_1 = LeNetConvLayer(
            rng,
            input=pool3_output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv4_1'],
            W=Weights['conv4_1'],
            b=bias['conv4_1']
        )

        #new image dimensions
        d = filter_shape['conv4_1'][0]

        self.conv4_2 = LeNetConvLayer(
            rng,
            input=self.conv4_1.output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv4_2'],
            W=Weights['conv4_2'],
            b=bias['conv4_2']
        )

        #new image dimensions
        d = filter_shape['conv4_2'][0]

        self.conv4_3 = LeNetConvLayer(
            rng,
            input=self.conv4_2.output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv4_3'],
            W=Weights['conv4_3'],
            b=bias['conv4_3']
        )
        #new image dimensions
        d = filter_shape['conv4_3'][0]

        self.conv4_4 = LeNetConvLayer(
            rng,
            input=self.conv4_3.output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv4_4'],
            W=Weights['conv4_4'],
            b=bias['conv4_4']
        )

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

        self.conv5_1 = LeNetConvLayer(
            rng,
            input=pool4_output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv5_1'],
            W=Weights['conv5_1'],
            b=bias['conv5_1']
        )

        #new image dimensions
        d = filter_shape['conv5_1'][0]

        self.conv5_2 = LeNetConvLayer(
            rng,
            input=self.conv5_1.output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv5_2'],
            W=Weights['conv5_2'],
            b=bias['conv5_2']
        )

        #new image dimensions
        d = filter_shape['conv5_2'][0]

        self.conv5_3 = LeNetConvLayer(
            rng,
            input=self.conv5_2.output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv5_3'],
            W=Weights['conv5_3'],
            b=bias['conv5_3']
        )

        #new image dimensions
        d = filter_shape['conv5_3'][0]

        self.conv5_4 = LeNetConvLayer(
            rng,
            input=self.conv5_3.output,
            image_shape=(batch_size, d, w, h),
            filter_shape=filter_shape['conv5_4'],
            W=Weights['conv5_4'],
            b=bias['conv5_4']
        )

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

        self.fc6     = HiddenLayer(
            rng,
            input=fc6_input,
            n_in=d * w * h,
            n_out=4096,
            activation=T.nnet.relu,
            W=Weights['fc6'],
            b=bias['fc6']
        )
        self.drop6   = drop(self.fc6.output, p=0.5)

        self.fc7     = HiddenLayer(
            rng,
            input=self.drop6 ,
            n_in=4096,
            n_out=4096,
            activation=T.nnet.relu,
            W=Weights['fc7'],
            b=bias['fc7']
        )

        self.drop7   = drop(self.fc7.output, p=0.5)

        self.fc8     = HiddenLayer(                         ## CHECK if Prob LogisticRegression can be used instead of HiddenLayer
            rng,                                            ## CHECK if Prob LogisticRegression can be used instead of HiddenLayer
            input=self.drop6 ,                              ## CHECK if Prob LogisticRegression can be used instead of HiddenLayer
            n_in=4096,                                      ## CHECK if Prob LogisticRegression can be used instead of HiddenLayer
            n_out=1000,                                     ## CHECK if Prob LogisticRegression can be used instead of HiddenLayer
            activation=None,                                ## CHECK if Prob LogisticRegression can be used instead of HiddenLayer
            W=Weights['fc8'],                               ## CHECK if Prob LogisticRegression can be used instead of HiddenLayer
            b=bias['fc8']                                   ## CHECK if Prob LogisticRegression can be used instead of HiddenLayer
        )                                                   ## CHECK if Prob LogisticRegression can be used instead of HiddenLayer

        self.prob    = LogisticRegression(
            input=self.fc8.output,
            n_in=1000,
            n_out=1000,
            W=numpy.identity(1000),
            b=numpy.zeros(1000)
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
        params = np.sum([ (getattr(self, l, None)).params for l in self.layer_names ]);
        # params = self.conv1_1.params+self.conv1_2.params+\
        #         self.conv2_1.params+self.conv2_2.params+\
        #         self.conv3_1.params+self.conv3_2.params+self.conv3_3.params+self.conv3_4.params+\
        #         self.conv4_1.params+self.conv4_2.params+self.conv4_3.params+self.conv4_4.params+\
        #         self.conv5_1.params+self.conv5_2.params+self.conv5_3.params+self.conv5_4.params+\
        #         self.fc6.params+self.fc7.params+self.fc8.params

        # create a list of gradients for all model parameters
        grads = T.grad(cost, params)

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
            cost,
            updates=updates,
            allow_input_downcast=True ## To allow float64 values to be changed to float32
        )
        print('Train model compiled...')

    def __str__(self):
        layers_strs = [ l+":\t"+str(getattr(self, l, 'LAYER_NOT_SET')) for l in self.layer_names ]
        header_str = "{}:{} L2_sqr:{}".format(self.__class__.__name__, self.model_name, 1234 ) # self.L2_sqr.eval()
        return header_str+"\n\t"+("\n\t".join(layers_strs))

#Implement a convolutional neural network to achieve at least 80% testing accuracy on CIFAR-dataset
class MyLeNet():
  def __init__(self, rng, datasets,
                batch_size=10       ,
                learning_rate=0.1 ):

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    n_train_batches = train_set_x.shape[0]
    n_valid_batches = valid_set_x.shape[0]
    n_test_batches = test_set_x.shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size
    self.restore_freq = n_test_batches

    x = T.matrix('x')   # the data is presented as rasterized images

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvLayer
    # (32, 32) is the size of MNIST images.

    POOL_SIZE = (2,2)
    layer0_input = x.reshape((batch_size, 3, 32, 32))


    self.conv11 = LeNetConvLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(64, 3, 3, 3),
        W=load_layer_params(self.layer_names[0])[0],
        b=load_layer_params(self.layer_names[0])[1]
    )
    print(self.conv11)


    self.conv12 = LeNetConvLayer(
        rng,
        input=self.conv11.output,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(64, 64, 3, 3),
        W=load_layer_params(self.layer_names[1])[0],
        b=load_layer_params(self.layer_names[1])[1]
    )
    print(self.conv12)

    pool1_output = pool.pool_2d(
            input=self.conv12.output,
            ds=(2,2),
            ignore_border=True
        )

    self.conv21 = LeNetConvLayer(
        rng,
        input=pool1_output,
        image_shape=(batch_size, 64, 16, 16),
        filter_shape=(128, 64, 3, 3),
        W=load_layer_params(self.layer_names[3])[0],
        b=load_layer_params(self.layer_names[3])[1]
    )
    print(self.conv21)


    self.conv22 = LeNetConvLayer(
        rng,
        input=self.conv21.output,
        image_shape=(batch_size, 128, 16, 16),
        filter_shape=(128, 128, 3, 3),
        W=load_layer_params(self.layer_names[4])[0],
        b=load_layer_params(self.layer_names[4])[1]
    )
    print(self.conv22)

    pool2_output = pool.pool_2d(
            input=self.conv22.output,
            ds=(2,2),
            ignore_border=True
        )

    self.conv31 = LeNetConvLayer(
        rng,
        input=pool2_output,
        image_shape=(batch_size, 128, 8, 8),
        filter_shape=(256, 128, 3, 3),
        W=load_layer_params(self.layer_names[6])[0],
        b=load_layer_params(self.layer_names[6])[1]
    )
    print(self.conv31)

    upsample4_output = self.conv31.output.repeat(2, axis=2).repeat(2, axis=3)
    self.conv41 = LeNetConvLayer(
        rng,
        input=upsample4_output,
        image_shape=(batch_size, 256, 16, 16),
        filter_shape=(128, 256, 3, 3),
        W=load_layer_params(self.layer_names[7])[0],
        b=load_layer_params(self.layer_names[7])[1]
    )
    print(self.conv41)
    self.conv42 = LeNetConvLayer(
        rng,
        input=self.conv41.output,
        image_shape=(batch_size, 128, 16, 16),
        filter_shape=(128, 128, 3, 3),
    )
    print(self.conv42)

# #################################################
                        # 128x16x16      # 128x16x16
    add_inputs1 = self.conv22.output + self.conv42.output
    upsample5_output = add_inputs1.repeat(2, axis=2).repeat(2, axis=3)
    self.conv51 = LeNetConvLayer(
        rng,
        input=upsample5_output,
        image_shape=(batch_size, 128, 32, 32),
        filter_shape=(64, 128, 3, 3),
    )
    print(self.conv51)
    self.conv52 = LeNetConvLayer(
        rng,
        input=self.conv51.output,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(64, 64, 3, 3),
    )
    print(self.conv52)

# #################################################
                    # 64x32x32       +     64x32x32
    add_inputs2 = self.conv12.output + self.conv52.output
    self.conv6 = LeNetConvLayer(
        rng,
        input=add_inputs2,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(3, 64, 3, 3),
    )
    print(self.conv6)

    # the cost we minimize during training is the NLL of the model
    #       # 3*32*32 - 3*32*32
    cost = T.mean((x - self.conv6.output.flatten(2))**2)

    # create a function to compute the mistakes that are made by the model
    self.test_model = theano.function(
        [x],
        cost,
        allow_input_downcast=True ## To allow float64 values to be changed to float32
    )
    print('Test model compiled...')

    self.validate_model = theano.function(
        [x],
        cost,
        allow_input_downcast=True ## To allow float64 values to be changed to float32
    )
    print('Validate model compiled...')

    # create a list of all model parameters to be fit by gradient descent
    params = self.conv11.params+self.conv12.params+\
                self.conv21.params+self.conv22.params+\
                self.conv31.params+\
                self.conv41.params+self.conv42.params+\
                self.conv51.params+self.conv52.params+\
                self.conv6.params
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    # updates = [
    #     (param_i, param_i - learning_rate * grad_i)
    #     for param_i, grad_i in zip(params, grads)
    # ]

    r = RmsProp()
    updates = r.getUpdates(params, grads)

    self.train_model = theano.function(
        [x],
        cost,
        updates=updates,
        allow_input_downcast=True ## To allow float64 values to be changed to float32
    )
    self.x = x
    self.x_corrupted = x_corrupted
    self.restore_count = 0
    print('Train model compiled...')

  def restore(self, inp):
    self.restore_count+=1
    check = not self.restore_count%self.restore_freq
    if(check):
        cor = self.x_corrupted.eval({self.x: inp.astype(np.float32)})
        out = self.conv6.output.eval({self.x: inp.astype(np.float32)})
        print out.shape
        label = str(self.restore_count+self.restore_freq)
        plot16(out,'./imgs2-reconstructed-'+label+'.png')
        plot16(cor,'./imgs2-corrupted-'+label+'.png')
        plot16(inp,'./imgs2-original-'+label+'.png')
    return check

  def test_model_restore(self, inp):
    self.restore(inp)
    return self.test_model(inp)

  def __str__(self):
    return 'MyLeNet\n'+str(self.conv11)+'\n'+str(self.conv12)+'\n'+str(self.conv21)+'\n'+str(self.conv22)+'\n'+str(self.conv31)+'\n'+str(self.conv41)+'\n'+str(self.conv42)+'\n'+str(self.conv51)+'\n'+str(self.conv52)+'\n'+str(self.conv6)+'\n'
