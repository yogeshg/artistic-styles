"""
Source Code for Homework 3 of ECBM E4040, Fall 2016, Columbia University

Instructor: Prof. Zoran Kostic

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""
import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample

from hw3_utils import shared_dataset, load_data
from hw3_nn import LogisticRegression, HiddenLayer, LeNetConvPoolLayer, train_nn

#Problem 1
#Implement the convolutional neural network architecture depicted in HW3 problem 1
#Reference code can be found in http://deeplearning.net/tutorial/code/convolutional_mlp.py
class MyLeNet():
  def __init__(self, rng, datasets,
                batch_size=10       ,
                nkerns=[32,64]      ,
                nhidden=[4096, 512] ,
                learning_rate=0.1 ):

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    index = T.lscalar()  # index to a [mini]batch    
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
    IMG_DIM = 32
    IMG_CHANNELS = 3
    FILT_SIZE = 3
    POOL_SIZE = (2,2)
    layer0_input = x.reshape((batch_size, IMG_CHANNELS, IMG_DIM, IMG_DIM))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (32-3+1 , 32-3+1) = (30, 30)
    # maxpooling reduces this further to (30/2, 30/2) = (15, 15)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 15, 15)
    self.layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, IMG_CHANNELS, IMG_DIM, IMG_DIM),
        filter_shape=(nkerns[0], IMG_CHANNELS, FILT_SIZE, FILT_SIZE),
        poolsize=POOL_SIZE
    )
    print(self.layer0)

    LAYER0_OUT_DIM = 15
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    self.layer1 = LeNetConvPoolLayer(
        rng,
        input=self.layer0.output,
        image_shape=(batch_size, nkerns[0], LAYER0_OUT_DIM, LAYER0_OUT_DIM),
        filter_shape=(nkerns[1], nkerns[0], FILT_SIZE, FILT_SIZE),
        poolsize=POOL_SIZE
    )
    print(self.layer1)

    LAYER1_OUT_DIM = 6
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = self.layer1.output.flatten(2)
    # construct a fully-connected sigmoidal layer
    self.layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * LAYER1_OUT_DIM * LAYER1_OUT_DIM,
        n_out=nhidden[0],
        activation=T.tanh
    )
    print(self.layer2)

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    # construct a fully-connected sigmoidal layer
    self.layer3 = HiddenLayer(
        rng,
        input=self.layer2.output,
        n_in=nhidden[0],
        n_out=nhidden[1],
        activation=T.tanh
    )
    print(self.layer3)

    # classify the values of the fully-connected sigmoidal layer
    self.layer4 = LogisticRegression(input=self.layer3.output, n_in=nhidden[1], n_out=10)
    print(self.layer4)

    # the cost we minimize during training is the NLL of the model
    cost = self.layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    self.test_model = theano.function(
        [index],
        self.layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    print('Test model compiled...')

    self.validate_model = theano.function(
        [index],
        self.layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    print('Validate model compiled...')

    # create a list of all model parameters to be fit by gradient descent
    params = self.layer4.params + self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params

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
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    print('Train model compiled...')

  def __str__(self):
    return 'MyMLP\n'+str(self.layer0)+'\n'+str(self.layer1)+'\n'+str(self.layer2)+'\n'+str(self.layer3)+'\n'+str(self.layer4)

def test_lenet( batch_size=10       ,
                nkerns=[32,64]      ,
                nhidden=[4096, 512] ,
                n_epochs=200        ,
                learning_rate=0.1   ):

    rng = numpy.random.RandomState(23455)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    myLeNet = MyLeNet(rng, batch_size=batch_size, nkerns=nkerns, nhidden=nhidden, learning_rate=learning_rate, datasets=datasets )
    train_nn(myLeNet.train_model, myLeNet.validate_model, myLeNet.test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            verbose = True)
 

#Problem 2.1
#Write a function to add translations
def translate_image():
    return
#Implement a convolutional neural network with the translation method for augmentation
def test_lenet_translation():
    return


#Problem 2.2
#Write a function to add roatations
def rotate_image():
    return
#Implement a convolutional neural network with the rotation method for augmentation
def test_lenet_rotation():
    return

#Problem 2.3
#Write a function to flip images
def flip_image():
    return
#Implement a convolutional neural network with the flip method for augmentation
def test_lenet_flip():
    return
    
    
#Problem 2.4
#Write a function to add noise, it should at least provide Gaussian-distributed and uniform-distributed noise with zero mean
def noise_injection():
    return
#Implement a convolutional neural network with the augmentation of injecting noise into input
def test_lenet_inject_noise_input():
    return
    
#Problem 3 
#Implement a convolutional neural network to achieve at least 80% testing accuracy on CIFAR-dataset
def MY_lenet():
    return

#Problem4
#Implement the convolutional neural network depicted in problem4 
def MY_CNN():
    return

