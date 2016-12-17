import logging
logging.basicConfig(level = logging.INFO)

import matplotlib.pyplot as plt
from NeuralNets.Models import VGG_19
from NeuralNets.ImportParameters import load_layer_params
from imageProcess import preprocess_image
from ASLoss import total_loss

import numpy as np
import theano
import theano.tensor as T

import NeuralNets.Utils as u
from PIL import Image

def train_style(alpha, beta, content_image_path, style_image_path, blank_image_path,
                style_layers = ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1'],
                content_layers = ['conv4_2'], n_epochs=10,learning_rate=0.000001):



    rng = np.random.RandomState(23455)

    print 'loading parameters...'

    p = load_layer_params('imagenet-vgg-verydeep-19.mat')

    print 'creating vgg19...'

    v = VGG_19(rng, None, p['filter_shape'], weights=p['weights'], bias=p['bias'])

    style_values = np.reshape(preprocess_image(style_image_path), (1, 3 * 224 * 224)) # (1,3,224,224)
    content_values = np.reshape(preprocess_image(content_image_path), (1, 3 * 224 * 224))  # (1,3,224,224)
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
    grad = T.grad((loss), (v.x))

    # blank = np.reshape(preprocess_image(blank_image_path), (1, 3 * 224 * 224))  # (1,3,224,224)   

    # updates = [
    #     (v.x, v.x - learning_rate * grad)
    # ]

    blank_values = np.reshape(preprocess_image(blank_image_path), (1, 3 * 224 * 224)).astype(np.float32)  # (1,3,224,224)
    blank_sh = theano.shared(blank_values)
    grad_sh = theano.shared(np.zeros_like(blank_values))

    updates = [
        (blank_sh, blank_sh + learning_rate * grad),
        (grad_sh, grad)
    ]
    givens = { v.x : blank_sh }

    train_model = theano.function([], loss, updates=updates, givens=givens)

    print('... training')

    for i in range(n_epochs):
        # print sum(blank)
        loss = train_model()
        print (loss)
        o = blank_sh.get_value()
        g = grad_sh.get_value()
        print 'magnitude of gradient:', np.sum(g**2)
        print 'min of gradient:', g.min()
        print 'max of gradient:', g.max()
        print 'mean of gradient:', g.mean()
        try:
            p = u.vector2pil( o[0].reshape((3,224,224)) )
            p.save('./data/gen_'+str(i)+'.jpg')
            print('./data/gen_'+str(i)+'.jpg'+'   saved')
        except Exception, e:
            print o.shape
            print e
    return loss

train_style(0, 1, 'test_images/thais.JPG', 'test_images/starry_night_google.jpg', 'test_images/thais.JPG',
                style_layers = ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1'],
                content_layers = ['conv4_2'], n_epochs=10,learning_rate=0.0)


