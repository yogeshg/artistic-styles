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
from ImportParameters.py import load_params

class VGG_19():
    def __init__(self, rng, datasets, batch_size=10, learning_rate=0.1):
        self.model_name = "VGG_ILSVRC_19_layers"
        self.layer_names = ["conv1_1","conv1_2","pool1","conv2_1","conv2_2","pool2"
                                ,"conv3_1","conv3_2","conv3_3","conv3_4","pool3","conv4_1","conv4_2","conv4_3","conv4_4","pool4"
                                ,"conv5_1","conv5_2","conv5_3","conv5_4","pool5","fc6","drop6","fc7","drop7","fc8","prob"]
        self.conv1_1 = None
        self.conv1_2 = None
        self.pool1   = None
        self.conv2_1 = None
        self.conv2_2 = None
        self.pool2   = None
        self.conv3_1 = None
        self.conv3_2 = None
        self.conv3_3 = None
        self.conv3_4 = None
        self.pool3   = None
        self.conv4_1 = None
        self.conv4_2 = None
        self.conv4_3 = None
        self.conv4_4 = None
        self.pool4   = None
        self.conv5_1 = None
        self.conv5_2 = None
        self.conv5_3 = None
        self.conv5_4 = None
        self.pool5   = None
        self.fc6     = None
        self.drop6   = None
        self.fc7     = None
        self.drop7   = None
        self.fc8     = None
        self.prob    = None
        return

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
        W=load_params(self.layer_names[0])[0],
        b=load_params(self.layer_names[0])[1]
    )
    print(self.conv11)


    self.conv12 = LeNetConvLayer(
        rng,
        input=self.conv11.output,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(64, 64, 3, 3),
        W=load_params(self.layer_names[1])[0],
        b=load_params(self.layer_names[1])[1]
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
        W=load_params(self.layer_names[3])[0],
        b=load_params(self.layer_names[3])[1]
    )
    print(self.conv21)


    self.conv22 = LeNetConvLayer(
        rng,
        input=self.conv21.output,
        image_shape=(batch_size, 128, 16, 16),
        filter_shape=(128, 128, 3, 3),
        W=load_params(self.layer_names[4])[0],
        b=load_params(self.layer_names[4])[1]
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
        W=load_params(self.layer_names[6])[0],
        b=load_params(self.layer_names[6])[1]
    )
    print(self.conv31)

    upsample4_output = self.conv31.output.repeat(2, axis=2).repeat(2, axis=3)
    self.conv41 = LeNetConvLayer(
        rng,
        input=upsample4_output,
        image_shape=(batch_size, 256, 16, 16),
        filter_shape=(128, 256, 3, 3),
        W=load_params(self.layer_names[7])[0],
        b=load_params(self.layer_names[7])[1]
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


