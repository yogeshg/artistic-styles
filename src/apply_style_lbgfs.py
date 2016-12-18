import logging
logging.basicConfig(level = logging.INFO)

import matplotlib.pyplot as plt
from NeuralNets.Models import VGG_19
from NeuralNets.ImportParameters import load_layer_params
from imageProcess import preprocess_image,deprocess_image,np2pil
from ASLoss import total_loss

import numpy as np
import scipy
import theano
import theano.tensor as T

import NeuralNets.Utils as u
from PIL import Image

import archiver
import json

def train_style(alpha, beta, content_image_path, style_image_path, blank_image_path,
                style_layers = ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1'],
                content_layers = ['conv4_2'], n_epochs=10, learning_rate=0.000001,
                optimizer='Adam'):

    vgg19_params = 'imagenet-vgg-verydeep-19.mat'
    image_shape = (3,224,224)

    hyperparameters = locals()

    rng = np.random.RandomState(23455)

    print 'loading parameters...'

    p = load_layer_params(vgg19_params)

    print 'creating vgg19...'

    v = VGG_19(rng, None, p['filter_shape'], weights=p['weights'], bias=p['bias'])

    style_values = np.reshape(preprocess_image(style_image_path), (1, np.prod(image_shape))) # (1,3,224,224)
    content_values = np.reshape(preprocess_image(content_image_path), (1, np.prod(image_shape)))  # (1,3,224,224)
    style_values = style_values.astype( np.float32 )
    content_values = content_values.astype( np.float32 )

    content_conv_4_2 = v.conv4_2.output.eval({v.x : content_values})
    style_conv1_1 = v.conv1_1.output.eval({v.x: style_values})
    style_conv2_1 = v.conv2_1.output.eval({v.x: style_values})
    style_conv3_1 = v.conv3_1.output.eval({v.x: style_values})
    style_conv4_1 = v.conv4_1.output.eval({v.x: style_values})
    style_conv5_1 = v.conv5_1.output.eval({v.x: style_values})
    print('content_conv_4_2: ' + str(content_conv_4_2.shape))
    print('style_conv1_1: ' + str(style_conv1_1.shape))
    print('style_conv2_1: ' + str(style_conv2_1.shape))
    print('style_conv3_1: ' + str(style_conv3_1.shape))
    print('style_conv4_1: ' + str(style_conv4_1.shape))
    print('style_conv5_1: ' + str(style_conv5_1.shape))

    style_activations = {
        'conv1_1': style_conv1_1,
        'conv2_1': style_conv2_1,
        'conv3_1': style_conv3_1,
        'conv4_1': style_conv4_1,
        'conv5_1': style_conv5_1
    }
    content_activations = {
        'conv4_2': content_conv_4_2
    }

    loss = total_loss(style_activations, content_activations, v,
                      style_layers, content_layers,
                      alpha, beta, p['filter_shape'])

    # loss : symbolic
    # grad = T.grad(loss, v.x)
    # grad = T.nnet.relu(T.grad(loss, v.x))
    # grad = T.grad(T.nnet.relu(loss), v.x)
    # grad = T.grad((loss), T.nnet.relu(v.x))
    grad = T.grad(loss, v.x)

    # updates = [
    #     (v.x, v.x - learning_rate * grad)
    # ]

    blank_values = np.reshape(preprocess_image(blank_image_path), (1, 3 * 224 * 224)).astype(np.float32)  # (1,3,224,224)
    blank_sh = theano.shared(blank_values)
    # grad_sh = theano.shared(np.zeros_like(blank_values))

    # updates = [
    #     (blank_sh, blank_sh - learning_rate * grad),
    #     (grad_sh, grad)
    # ]

    #if optimizer=='Adam':
    #    a = u.Adam(learning_rate=learning_rate)
    #    updates = a.getUpdates(blank_sh,grad)
    #else:
    #    updates = [
    #        (blank_sh, blank_sh - learning_rate * grad)
    #    ]

    givens = {v.x: blank_sh}

    #train_model = theano.function([], loss, updates=updates, givens=givens)

    loss_fct = theano.function([], loss,  givens=givens)
    grad_fct = theano.function([], grad,  givens=givens)

    def loss_fct_py(x0):
        x0 = (x0.reshape((1, 3* 224* 224))).astype(np.float32)
        blank_sh.set_value(x0)
        return loss_fct()

    def grad_fct_py(x0):
        x0 = (x0.reshape((1, 3* 224* 224))).astype(np.float32)
        blank_sh.set_value(x0)
        return grad_fct()

    x0 = blank_sh.get_value()

    for i in range(n_epochs):
        print(i)
        new_im, losses, _ = scipy.optimize.fmin_l_bfgs_b(loss_fct_py, x0.flatten(), fprime=grad_fct_py, maxfun=40)
        print new_im.shape
        blank_sh.set_value(new_im)
        x0 = blank_sh.get_value()
        #print losses
        o = blank_sh.get_value()
        '''
        try:
            p = np2pil(deprocess_image(o.reshape((1, 3, 224, 224)))[0])
            # p = np2pil( o.reshape((1,3,224,224))[0])
            # p = np2pil(deprocess_image(np.reshape(o,(1,3,224,224)))[0])
            filepath = archiver.getFilePath('gen_{:04d}.jpg'.format(i))
            p.save(filepath)
            print(filepath + '   saved')
        except Exception, e:
            print o.shape
            print e
'''




    print('... training')

    return None

train_style(0, 1, 'test_images/thais.JPG', 'test_images/starry_night_google.jpg', 'test_images/thais.JPG',
                style_layers = ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1'],
                content_layers = ['conv4_2'], n_epochs=100,learning_rate=10)
