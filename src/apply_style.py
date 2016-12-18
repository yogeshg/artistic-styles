import logging
logging.basicConfig(level = logging.INFO)

import matplotlib.pyplot as plt
from NeuralNets.Models import VGG_19
from NeuralNets.ImportParameters import load_layer_params
from imageProcess import preprocess_image,deprocess_image,np2pil
from ASLoss import total_loss

import numpy as np
import theano
import theano.tensor as T

import NeuralNets.Utils as u
from PIL import Image

import archiver
import json
def white_noise(shape=(1,3,224,224)):
    image = np.random.randint(0,255,size=shape)
    image[:,0,:,:] = image[:,0, :, :] - 103.939
    image[:,1,:,:] = image[:,1, :, :] - 116.779
    image[:,2,:,:] = image[:,2, :, :] - 123.68
    image = image[:, ::-1,:, :]
    return image

def train_style(alpha, beta, content_image_path, style_image_path, blank_image_path=None,
                style_layers = ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1'],
                content_layers = ['conv4_2'], n_epochs=10, learning_rate=0.000001,
                optimizer='Adam'):

    vgg19_params = 'imagenet-vgg-verydeep-19.mat'
    #image_shape = (3,224,224)

    hyperparameters = locals()

    rng = np.random.RandomState(23455)

    print 'loading parameters...'

    p = load_layer_params(vgg19_params)

    print 'creating vgg19...'

    #v = VGG_19(rng, None, p['filter_shape'], weights=p['weights'], bias=p['bias'],)
    style_image,style_shape = preprocess_image(style_image_path,resize=False)
    style_values = np.reshape(style_image, (style_shape[0], np.prod(style_shape[1:])))
    content_image,content_shape = preprocess_image(content_image_path,resize=False)
    content_values = np.reshape(content_image,(content_shape[0], np.prod(content_shape[1:])))
    style_values = style_values.astype( np.float32 )
    content_values = content_values.astype( np.float32 )
    v_style = VGG_19(rng, None, p['filter_shape'], weights=p['weights'], bias=p['bias'],image_size=style_shape)
    v = VGG_19(rng,None,p['filter_shape'], weights=p['weights'], bias=p['bias'],image_size=content_shape)
    content_conv_4_2 = v.conv4_2.output.eval({v.x : content_values})
    style_conv1_1 = v_style.conv1_1.output.eval({v_style.x: style_values})
    style_conv2_1 = v_style.conv2_1.output.eval({v_style.x: style_values})
    style_conv3_1 = v_style.conv3_1.output.eval({v_style.x: style_values})
    style_conv4_1 = v_style.conv4_1.output.eval({v_style.x: style_values})
    style_conv5_1 = v_style.conv5_1.output.eval({v_style.x: style_values})
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
    if blank_image_path==None:
        blank_values = np.reshape(white_noise(content_shape),(content_shape[0], np.prod(content_shape[1:]))).astype(np.float32)
    else:
        blank_image,blank_image_shape = preprocess_image(blank_image_path,resize=False)
        blank_values = np.reshape(blank_image, (content_shape[0], np.prod(content_shape[1:]))).astype(np.float32)  # (1,3,224,224)
    blank_sh = theano.shared(blank_values)
    # grad_sh = theano.shared(np.zeros_like(blank_values))

    # updates = [
    #     (blank_sh, blank_sh - learning_rate * grad),
    #     (grad_sh, grad)
    # ]

    if optimizer=='Adam':
        a = u.Adam(learning_rate=learning_rate)
        updates = a.getUpdates(blank_sh,grad)
    else:
        updates = [
            (blank_sh, blank_sh - learning_rate * grad)
        ]
    givens = { v.x : blank_sh }

    train_model = theano.function([], loss, updates=updates, givens=givens)

    print('... training')

    archiver.cleanDir(archiver.CURRDIR)
    with open(archiver.getFilePath('properties.json'), 'w') as f:
        json.dump(hyperparameters, f)

    try:
        for i in range(n_epochs):
            # print sum(blank)
            loss = train_model()
            print '%.3e' % loss
            o = blank_sh.get_value()
            # g = grad_sh.get_value()
            #print 'magnitude of gradient:', np.sum(g**2)
            #print 'min of gradient:', g.min()
            #print 'max of gradient:', g.max()
            #print 'mean of gradient:', g.mean()
            try:
                p = np2pil( deprocess_image(o.reshape(content_shape))[0])
                #p = np2pil( o.reshape((1,3,224,224))[0])
                #p = np2pil(deprocess_image(np.reshape(o,(1,3,224,224)))[0])
                filepath = archiver.getFilePath('gen_{:04d}.jpg'.format(i))
                p.save( filepath )
                print(filepath+'   saved')
            except Exception, e:
                print o.shape
                print e

    except Exception, e:
        print e
    finally:
        archiver.archiveDir(archiver.CURRDIR)


    return loss

train_style(0.02, 2e-4, 'test_images/tubingen.jpg', 'test_images/starry_night_google.jpg', blank_image_path='test_images/tubingen.jpg',
                style_layers = ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1'],
                content_layers = ['conv4_2'], n_epochs=100,learning_rate=10)
