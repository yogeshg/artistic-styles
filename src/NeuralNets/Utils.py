"""
Source Code for Homework 3 of ECBM E4040, Fall 2016, Columbia University

This code contains implementation of several utility funtions for the homework.

Instructor: Prof.  Zoran Kostic

"""
import os
import timeit
import inspect
import sys
import re

import numpy
import scipy.io
import tarfile
import theano
import theano.tensor as T

def floatX(X):
    return numpy.asarray(X, dtype=theano.config.floatX)

def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(numpy.asarray(X, dtype=dtype), name=name)

def shared_zeros(shape, dtype=theano.config.floatX, name=None):
    return sharedX(numpy.zeros(shape), dtype=dtype, name=name)

WORDBOUNDS_REGEX = re.compile(r'\s+')

def about(x):
    strx = str(x)
    strx = WORDBOUNDS_REGEX.sub(' ', strx)
    lenx = len(strx)
    if(lenx > 70):
        strx=strx[:70]+'...'+str(lenx)

    return str(type(x))+'\n'+strx

# RMS prop algo
class RmsProp():
    def __init__(self,rho=0.9, lr = 0.001, epsilon = 1e-6):
        self.rho = rho
        self.lr = lr
        self.epsilon = epsilon
        return

    def getUpdates(self, params, grads):
        accumulators = [shared_zeros(p.get_value().shape) for p in params]
        updates = []
    
        for p, gr, ac in zip(params, grads, accumulators):
            ac2 = self.rho * ac + (1 - self.rho) * gr ** 2
            p2 = p - self.lr * gr / T.sqrt(ac2 + self.epsilon)
            updates.extend([(ac, ac2), (p, p2)])

        return updates


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

def load_data(ds_rate=None, theano_shared=True):
    ''' Loads the SVHN dataset

    :type ds_rate: float
    :param ds_rate: downsample rate; should be larger than 1, if provided.

    :type theano_shared: boolean
    :param theano_shared: If true, the function returns the dataset as Theano
    shared variables. Otherwise, the function returns raw data.
    '''
    if ds_rate is not None:
        assert(ds_rate > 1.)

    # Download the CIFAR-10 dataset if it is not present
    def check_dataset(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.getcwd(),
            "..",
            "data",
            dataset
        )
        #f_name = new_path.replace("src/../data/%s"%dataset, "data/") 
        f_name = os.path.join(
            os.getcwd(),
            "..",
            "data"
        )
        if (not os.path.isfile(new_path)):
            from six.moves import urllib
            origin = (
                'https://www.cs.toronto.edu/~kriz/' + dataset
            )
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, new_path) 
             
        tar = tarfile.open(new_path)
        file_names = tar.getnames()
        for file_name in file_names:
            tar.extract(file_name,f_name)
        tar.close()              
        
        return f_name
    
    f_name=check_dataset('cifar-10-matlab-00100.tar.gz')
    
    train_batches=os.path.join(f_name,'cifar-10-batches-mat/data_batch_1.mat')
    print train_batches
    
    # Load data and convert data format
    train_batches=['data_batch_1.mat','data_batch_2.mat','data_batch_3.mat','data_batch_4.mat','data_batch_5.mat']
    train_batch=os.path.join(f_name,'cifar-10-batches-mat',train_batches[0])
    train_set=scipy.io.loadmat(train_batch)
    print 'loaded a matrix of shape', train_set['data'].shape
    train_set['data']=train_set['data']/255.
    for i in range(4):
        train_batch=os.path.join(f_name,'cifar-10-batches-mat',train_batches[i+1])
        temp=scipy.io.loadmat(train_batch)
        print 'loaded a matrix of shape', temp['data'].shape
        train_set['data']=numpy.concatenate((train_set['data'],temp['data']/255.),axis=0)
        train_set['labels']=numpy.concatenate((train_set['labels'].flatten(),temp['labels'].flatten()),axis=0)
    
    test_batches=os.path.join(f_name,'cifar-10-batches-mat/test_batch.mat')
    test_set=scipy.io.loadmat(test_batches)
    print 'loaded a matrix of shape', test_set['data'].shape
    test_set['data']=test_set['data']/255.
    test_set['labels']=test_set['labels'].flatten()
    
    train_set=(train_set['data'],train_set['labels'])
    test_set=(test_set['data'],test_set['labels'])
    

    # Downsample the training dataset if specified
    train_set_len = len(train_set[1])
    if ds_rate is not None:
        train_set_len = int(train_set_len // ds_rate)
        train_set = [x[:train_set_len] for x in train_set]

    # Extract validation dataset from train dataset
    valid_set = [x[-(train_set_len//5):] for x in train_set]
    train_set = [x[:-(train_set_len//5)] for x in train_set]

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    if theano_shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    else:
        rval = [train_set, valid_set, test_set]

    return rval


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

MAX_ROTATE = 5 # Degrees
MAX_TRANSLATE = 2.5 # Pixels
MAX_NOISE = 0.01 # for [0,1]

# function to add translations
def translate_image(inp, t):
    return pil2vector(rotateTranslate(vector2pil(inp),0,t))

# function to add roatations
def rotate_image(inp, r):
    return pil2vector(rotateTranslate(vector2pil(inp),r))

# function to flip images
def flip_image(inp):
    return pil2vector(rotateTranslate(vector2pil(inp),0,(-64,0),-1))

def plot8(arr, filename):
        fig = plt.figure()
        for i in range(1,8+1):
            print i,
            ax = fig.add_subplot(4,2,i)
            ax.imshow(vector2image( arr[i] ))
        fig.savefig(filename)
        return

def plot16(arr, filename):
        fig = plt.figure()
        for i in range(1,16+1):
            # print i,
            ax = fig.add_subplot(4,4,i)
            ax.imshow(vector2image( arr[i] ))
        fig.savefig(filename)
        return

def noise_injection(inp, method='normal', magnitude=MAX_NOISE):
    if (method=='uniform'):
        err = numpy.random.uniform(low=0.0, high=magnitude, size=inp.shape)
    else :
        err = numpy.random.normal(loc=0.0, scale=magnitude, size=inp.shape)
    out = inp + err
    out = (out - out.min())/(out.max() - out.min())
    return out

def train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            datasets, batch_size,
            verbose = True):
    """
    Wrapper function for training and test THEANO model

    :type train_model: Theano.function
    :param train_model:

    :type validate_model: Theano.function
    :param validate_model:

    :type test_model: Theano.function
    :param test_model:

    :type n_train_batches: int
    :param n_train_batches: number of training batches

    :type n_valid_batches: int
    :param n_valid_batches: number of validation batches

    :type n_test_batches: int
    :param n_test_batches: number of testing batches

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to

    """
    print( 'train_nn: '+str(locals()))
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.85  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model(getBatch( minibatch_index, batch_size, train_set_x))

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(getBatch(i,batch_size,valid_set_x))
                                        for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                if verbose:
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(getBatch(i,batch_size,test_set_x))
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation error of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    sys.stderr.write(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))

