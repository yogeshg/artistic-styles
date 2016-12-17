"""
Source Code for Homework 3 of ECBM E4040, Fall 2016, Columbia University

This code contains implementation of some basic components in neural network.

Instructor: Prof. Zoran Kostic

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""

import logging
import timeit
import inspect
import sys
import numpy as np
# from theano.tensor.nnet import conv
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
# from theano.tensor.signal import pool

from Utils import about

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out,W_values=None,b_values=None):

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug('Initializing...')

        assert (type (input) in (theano.tensor.dtensor4, theano.tensor.TensorVariable)), type (input)
        assert (type (n_in) == int ), type(n_in)
        assert (type (n_out) == int ), type(n_out)
        assert (type (W_values) in (type(None), np.ndarray)), type (W_values)
        assert (type (b_values) in (type(None), np.ndarray)), type (b_values)

        self.logger.debug( 'input' + about(input) )
        self.logger.debug( 'W_values' + about(W_values) )
        self.logger.debug( 'b_values' + about(b_values) )

        self.input = input

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W_values is None:
            W_values = np.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                )
        # W = theano.shared(value=W_values, name='W', borrow=True)
        W = W_values

        if b_values is None:
            # initialize the biases b as a vector of n_out 0s
            b_values = np.zeros((n_out,),dtype=theano.config.floatX)
        # b = theano.shared( value=b_values, name='b', borrow=True)
        b = b_values

        self.W=W
        self.b=b

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

        self.L1 = abs(self.W).sum()
        self.L2_sqr = (self.W**2).sum()

    def __str__(self):
        return "Layer:{} L2_sqr:{}".format(self.__class__.__name__, self.L2_sqr.eval())

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W_values=None, b_values=None,
                 activation=T.tanh):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug('Initializing...')

        assert (type (rng) == np.random.RandomState), type (rng)
        assert (type (input) in (theano.tensor.dtensor4, theano.tensor.TensorVariable)), type (input)
        assert (type (n_in) == int ), type(n_in)
        assert (type (n_out) == int ), type(n_out)
        try:
            assert (type (activation) in (theano.Op, type(None))), type (activation)
        except Exception, e:
            self.logger.exception(e)
            
        assert (type (W_values) in (type(None), np.ndarray)), type (W_values)
        assert (type (b_values) in (type(None), np.ndarray)), type (b_values)

        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W_values is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

        W = theano.shared(value=W_values, name='W', borrow=True)

        if b_values is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]
        self.L1 = abs(self.W).sum()
        self.L2_sqr = (self.W**2).sum()

    def __str__(self):
        return "Layer:{} L2_sqr:{}".format(self.__class__.__name__, self.L2_sqr.eval())


class LeNetConvLayer(object):
    """Convolutional Layer"""

    def __init__(self, rng, input, filter_shape, image_shape,W_values=None,b_values=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug('Initializing...')

        assert (type (rng) == np.random.RandomState), type (rng)
        assert (type (input) in (theano.tensor.dtensor4, theano.tensor.TensorVariable)), type (input)
        assert (type (filter_shape) in (tuple, list)), type (filter_shape)
        assert (len (filter_shape) == 4), len (filter_shape)
        assert (type (image_shape) in (tuple, list)), type (image_shape)
        assert (len (image_shape) == 4), len (image_shape)
        assert (type (W_values) in (type(None), np.ndarray)), type (W_values)
        assert (type (b_values) in (type(None), np.ndarray)), type (b_values)

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))

        self.logger.debug('W_values: '+about(W_values))
        if type(W_values)==type(None):
            W_values = np.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                    )
        W = theano.shared( W_values, borrow=True)

        self.logger.debug('b_values: '+about(b_values))
        if type(b_values)==type(None):
            # the bias is a 1D tensor -- one bias per output feature map
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)

        self.W = W
        self.b = b
        
        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode='half'
        )

        self.output = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
        self.L1 = abs(self.W).sum()
        self.L2_sqr = (self.W**2).sum()

    def __str__(self):
        return "Layer:{} L2_sqr:{}".format(self.__class__.__name__, self.L2_sqr.eval())

def drop(input, p=0.5):
    """
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout is applied

    :type p: float or double between 0. and 1.
    :param p: p probability of NOT dropping out a unit, therefore (1.-p) is the drop rate.

    """
    rng = np.random.RandomState(1234)
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
    return input * mask

class DropoutHiddenLayer(object):
    def __init__(self, rng, is_train, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, p=0.5):
        """
        Hidden unit activation is given by: activation(dot(input,W) + b)

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type is_train: theano.iscalar
        :param is_train: indicator pseudo-boolean (int) for switching between training and prediction

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer

        :type p: float or double
        :param p: probability of NOT dropping out a unit
        """
        self.input = input

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)


        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b

        output = activation(lin_output)

        # multiply output and drop -> in an approximation the scaling effects cancel out
        train_output = drop(output,p)

        #is_train is a pseudo boolean theano variable for switching between training and prediction
        self.output = T.switch(T.neq(is_train, 0), train_output, p*output)

        self.L1 = abs(W).sum()
        self.L2_sqr = (W**2).sum()

        # parameters of the model
        self.params = [self.W, self.b]

        self._is_train = is_train
        self._p = p

    def __str__ (self):
        it = 'undecided'
        p = 'undecided'
        try:
            it = self._is_train.eval()
            p = self._p
        except:
            pass
        return 'DropoutHiddenLayer (is_train:{},p:{})'.format(it,p)+str(self.L1.eval())+' '+str(self.L2_sqr.eval())
