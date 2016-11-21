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

from hw3_utils import shared_dataset, load_data, RmsProp
from hw3_nn import LogisticRegression, HiddenLayer, train_nn,LeNetConvPoolLayer 
from hw2c import DropoutHiddenLayer

#Problem 3 
#Implement a convolutional neural network to achieve at least 80% testing accuracy on CIFAR-dataset
# def MY_lenet():
#     return
class MyLeNet():
  def __init__(self, rng, datasets,
                batch_size=10       ,
                learning_rate=0.1 ):

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (32, 32) is the size of MNIST images.

    POOL_SIZE = (2,2)
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    self.conv11 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(32, 3, 3, 3)
    )
    print(self.conv11)

    self.conv12 = LeNetConvPoolLayer(
        rng,
        input=self.conv11.output,
        image_shape=(batch_size, 32, 32, 32),
        filter_shape=(32, 32, 3, 3),
    )
    print(self.conv12)

    pool1_output = pool.pool_2d(
            input=self.conv12.output,
            ds=(2,2),
            ignore_border=True
        )

    self.conv21 = LeNetConvPoolLayer(
        rng,
        input=pool1_output,
        image_shape=(batch_size, 32, 16, 16),
        filter_shape=(64, 32, 3, 3)
    )
    print(self.conv21)


    self.conv22 = LeNetConvPoolLayer(
        rng,
        input=self.conv21.output,
        image_shape=(batch_size, 64, 16, 16),
        filter_shape=(64, 64, 3, 3),
    )
    print(self.conv22)

    pool2_output = pool.pool_2d(
            input=self.conv22.output,
            ds=(2,2),
            ignore_border=True
        )

    self.conv31 = LeNetConvPoolLayer(
        rng,
        input=pool2_output,
        image_shape=(batch_size, 64,  8,  8),
        filter_shape=(128,64, 3, 3)
    )
    print(self.conv31)


    self.conv32 = LeNetConvPoolLayer(
        rng,
        input=self.conv31.output,
        image_shape=(batch_size, 128, 8,  8),
        filter_shape=(128,128,3, 3),
    )
    print(self.conv32)

    pool3_output = pool.pool_2d(
            input=self.conv32.output,
            ds=(2,2),
            ignore_border=True
        )

    training_enabled = T.iscalar('training_enabled') # pseudo boolean for switching between training and prediction

    self.hidden3 = DropoutHiddenLayer(
        rng,
        input=pool3_output.flatten(2),
        is_train=training_enabled,
        n_in=128*4*4,
        n_out=1024,
        p=0.5
    )
    print(self.hidden3)

    self.hidden4 = DropoutHiddenLayer(
        rng,
        input=self.hidden3.output,
        is_train=training_enabled,
        n_in=1024,
        n_out=1024,
        p=0.5
    )
    print(self.hidden4)

    self.hidden5 = DropoutHiddenLayer(
        rng,
        input=self.hidden4.output,
        is_train=training_enabled,
        n_in=1024,
        n_out=1024,
        p=0.5
    )
    print(self.hidden5)

    self.hidden6 = DropoutHiddenLayer(
        rng,
        input=self.hidden5.output,
        is_train=training_enabled,
        n_in=1024,
        n_out=200,
        p=0.5
    )
    print(self.hidden6)

    # classify the values of the fully-connected sigmoidal layer
    self.softmax6 = LogisticRegression(
        input=self.hidden6.output,
        n_in=200,
        n_out=10)
    print(self.softmax6)

    # the cost we minimize during training is the NLL of the model
    cost = self.softmax6.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    self.test_model = theano.function(
        [x,y],
        self.softmax6.errors(y),
        allow_input_downcast=True, ## To allow float64 values to be changed to float32
        givens={
            training_enabled: numpy.cast['int32'](0)
        }
    )
    print('Test model compiled...')

    self.validate_model = theano.function(
        [x,y],
        self.softmax6.errors(y),
        allow_input_downcast=True, ## To allow float64 values to be changed to float32
        givens={
            training_enabled: numpy.cast['int32'](0)
        }
    )
    print('Validate model compiled...')

    # create a list of all model parameters to be fit by gradient descent
    params = self.conv11.params + self.conv12.params + self.conv21.params + self.conv22.params + \
                self.conv31.params + self.conv32.params + \
                self.hidden3.params + self.hidden4.params + self.hidden5.params + self.hidden6.params + \
                self.softmax6.params

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
        [x,y],
        cost,
        updates=updates,
        allow_input_downcast=True, ## To allow float64 values to be changed to float32
        givens={
            training_enabled: numpy.cast['int32'](1)
        }

    )
    print('Train model compiled...')

  def __str__(self):
    return 'MyLeNet\n'+str(self.conv11)+'\n'+str(self.conv12)+str(self.conv21)+'\n'+str(self.conv22)+'\n'+str(self.hidden3)+'\n'+str(self.hidden4)+'\n'+str(self.hidden5)+'\n'+str(self.hidden6)+'\n'+str(self.softmax6)

import math
from PIL import Image
import random

# def image2vector(im):
#     return im.transpose(2,0,1).flatten()
def vector2image(v):
    return np.reshape(v,(3,32,32)).transpose(1,2,0)

def vector2pil(v):
    return Image.fromarray(np.uint8(255*np.reshape(v,(3,32,32)).transpose(1,2,0)))
def pil2vector(p):
    return (np.array(p.getdata()).T.flatten()/255.)

def rotateTranslate(image, angle, new_center = None, yMirror=1):
    angle = -angle/180.0*math.pi
    x,y = image.getbbox()[2:4]
    x = x/2
    y = y/2
    nx,ny = x,y 
    if new_center:
        (dx,dy) = new_center
        nx+=dx
        ny+=dy
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = yMirror * cosine
    b = yMirror * sine
    c = (yMirror * (x-nx*a-ny*b))
    d = -sine
    e = cosine
    f = y-nx*d-ny*e
    return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=Image.BICUBIC)

#Problem 2.1
#Write a function to add translations
def translate_image(inp, t):
    return pil2vector(rotateTranslate(vector2pil(inp),0,t))

#Problem 2.2
#Write a function to add roatations
def rotate_image(inp, r):
    return pil2vector(rotateTranslate(vector2pil(inp),r))

#Problem 2.3
#Write a function to flip images
def flip_image(inp):
    return pil2vector(rotateTranslate(vector2pil(inp),0,(-64,0),-1))

## plot16(temp, './imgs-flip-sample.png')
def plot16(arr, filename):
        # fig = plt.figure()
        # for i in range(1,16+1):
        #     print i,
        #     ax = fig.add_subplot(4,4,i)
        #     ax.imshow(vector2image( arr[i] ))
        # fig.savefig(filename)
        return

MAX_ROTATE = 5 # Degrees
MAX_TRANSLATE = 2.5 # Pixels
MAX_NOISE = 0.01 # for [0,1]

def test_lenet( batch_size=10       ,
                nkerns=[32,64]      ,
                nhidden=[4096, 512] ,
                n_epochs=200        ,
                learning_rate=0.1   ,
                rotation=False, translation=False, flipping=False, noise=None):
    print 'test_lenet:', locals()
    rng = numpy.random.RandomState(23455)

    datasets = load_data(theano_shared=False)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    N = train_set_x.shape[0]
    print N

    if( flipping ):
        temp = [flip_image(train_set_x[i]) for i in range(train_set_x.shape[0])]
        train_set_x = np.concatenate([train_set_x, temp])
        train_set_y = np.concatenate([train_set_y, train_set_y])
        plot16(temp, './imgs-flip-aug.png')
        plot16(train_set_x, './imgs-flip-orig.png')

    if( rotation ):
        temp = [rotate_image(train_set_x[i], (MAX_ROTATE)-random.random()*(2*MAX_ROTATE)) for i in range(int(train_set_x.shape[0]))]
        train_set_x = np.concatenate([train_set_x, temp])
        train_set_y = np.concatenate([train_set_y, train_set_y])
        plot16(temp, './imgs-rotate-aug.png')
        plot16(train_set_x, './imgs-rotate-orig.png')

    if( translation ):
        temp = [translate_image(train_set_x[i], (MAX_TRANSLATE-random.random()*(2*MAX_TRANSLATE),MAX_TRANSLATE-random.random()*(2*MAX_TRANSLATE))) for i in range(train_set_x.shape[0])]
        train_set_x = np.concatenate([train_set_x, temp])
        train_set_y = np.concatenate([train_set_y, train_set_y])
        plot16(temp, './imgs-translate-aug.png')
        plot16(train_set_x, './imgs-translate-orig.png')

    if( noise ):
        temp = [noise_injection(train_set_x[i], magnitude=MAX_NOISE, method=noise) for i in range(train_set_x.shape[0])]
        train_set_x = np.concatenate([train_set_x, temp])
        train_set_y = np.concatenate([train_set_y, train_set_y])
        plot16(temp, './imgs-'+str(noise)+'Noise-aug.png')
        plot16(train_set_x, './imgs-'+str(noise)+'Noise-orig.png')

    # compute number of minibatches for training, validation and testing
    # n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    # n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    # n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches = train_set_x.shape[0]
    n_valid_batches = valid_set_x.shape[0]
    n_test_batches = test_set_x.shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    datasets[0] = (train_set_x, train_set_y)
    datasets[1] = (valid_set_x, valid_set_y)
    datasets[2] = (test_set_x, test_set_y  )
 
    myLeNet = MyLeNet(rng, datasets, batch_size=batch_size, learning_rate=learning_rate)
    print myLeNet
    train_nn(myLeNet.train_model, myLeNet.validate_model, myLeNet.test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            datasets, batch_size,
            verbose = True)

#Implement a convolutional neural network with the translation method for augmentation
def test_lenet_translation(**kwargs):
    kwargs['translation']=True
    return test_lenet(**kwargs)

#Problem 2.2
#Implement a convolutional neural network with the rotation method for augmentation
def test_lenet_rotation(**kwargs):
    kwargs['rotation']=True
    return test_lenet(**kwargs)

#Problem 2.3
#Implement a convolutional neural network with the flip method for augmentation
def test_lenet_flip(**kwargs):
    kwargs['flipping']=True
    return test_lenet(**kwargs)

#Problem 2.4
#Write a function to add noise, it should at least provide Gaussian-distributed and uniform-distributed noise with zero mean
def noise_injection(inp, method='normal', magnitude=MAX_NOISE):
    if (method=='uniform'):
        err = numpy.random.uniform(low=0.0, high=magnitude, size=inp.shape)
    else :
        err = numpy.random.normal(loc=0.0, scale=magnitude, size=inp.shape)
    out = inp + err
    out = (out - out.min())/(out.max() - out.min())
    return out
#Implement a convolutional neural network with the augmentation of injecting noise into input
def test_lenet_inject_noise_input(**kwargs):
    if(not kwargs.has_key('noise')):
        kwargs['noise']='normal'
    return test_lenet(**kwargs)
    

#Problem4
#Implement the convolutional neural network depicted in problem4 
def MY_CNN():
    return

if __name__ == '__main__': 

    test_lenet(batch_size=512, n_epochs=1000, flipping=True, translation=True)
    test_lenet(batch_size=512, n_epochs=1000, flipping=True)

    # test_lenet_rotation(batch_size=256, n_epochs=500)
    # test_lenet_translation(batch_size=256, n_epochs=500)
    # test_lenet(batch_size=256, n_epochs=500)
    # test_lenet_flip(batch_size=256, n_epochs=500)

    # test_lenet(batch_size=20, n_epochs=10, flipping=True)
    # test_lenet(batch_size=20, n_epochs=10, rotation=True)
    # test_lenet(batch_size=20, n_epochs=10, translation=True)

