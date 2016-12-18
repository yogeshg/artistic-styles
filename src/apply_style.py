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
                optimizer='Adam',resize=True,lbfgs_maxfun=20):

    vgg19_params = 'imagenet-vgg-verydeep-19.mat'
    #image_shape = (3,224,224)

    hyperparameters = locals()

    rng = np.random.RandomState(23455)

    print 'loading parameters...'

    p = load_layer_params(vgg19_params)

    print 'creating vgg19...'

    #v = VGG_19(rng, None, p['filter_shape'], weights=p['weights'], bias=p['bias'],)
    style_image,style_shape = preprocess_image(style_image_path,resize=resize)
    style_values = np.reshape(style_image, (style_shape[0], np.prod(style_shape[1:])))
    content_image,content_shape = preprocess_image(content_image_path,resize=resize)
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
        blank_image,blank_image_shape = preprocess_image(blank_image_path,resize=resize)
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
        givens = { v.x : blank_sh }
        train_model = theano.function([], loss, updates=updates, givens=givens)
    elif optimizer =='GD':
        updates = [
            (blank_sh, blank_sh - learning_rate * grad)
        ]
        givens = { v.x : blank_sh }
        train_model = theano.function([], loss, updates=updates, givens=givens)
    elif optimizer =='l-bfgs':
        blank_values=np.reshape(blank_values,(content_shape)).astype(np.float32)
        blank_sh = theano.shared(blank_values)
        givens = {v.x: blank_sh.flatten(2)}
        x0 = blank_sh.get_value().astype(np.float32)
        loss_fct = theano.function([], loss,  givens=givens)
        grad_fct = theano.function([], grad,  givens=givens)
        loss_grad_fct = theano.function([],[loss,grad],givens=givens)

    def loss_grad_fct_py(x1):
        x1 = (x1.reshape(content_shape)).astype(np.float32)
        blank_sh.set_value(x1)
        out = loss_grad_fct()
        loss_val = out[0]
        grad_val = np.ones(np.prod(content_shape[1:]))-1+(np.array(out[1:]).flatten())
        return loss_val,grad_val
    def loss_fct_py(x1):
        x1 = (x1.reshape(content_shape)).astype(np.float32)
        blank_sh.set_value(x1)
        return loss_fct()
    def grad_fct_py(x1):
        x1 = (x1.reshape(content_shape)).astype(np.float32)
        blank_sh.set_value(x1)
        #print np.array(grad_fct()).flatten().shape
        #print np.ones(3*224*224).shape
        #print type(np.array(grad_fct()).flatten())
        #print type(np.ones(3*224*224))
        temp2 = np.ones(np.prod(content_shape[1:]))-1+(np.array(grad_fct()).flatten())
        #return np.array(grad_fct()).flatten()
        #return np.ones(3*224*224)
        return temp2

    #https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py
    class loss_grad_eval():
        def __init__(self):
            self.loss_val = None
            self.grad_val = None
        def loss(self,x1):
            loss_val,grad_val = loss_grad_fct_py(x1)
            self.loss_val = loss_val
            self.grad_val = grad_val
            return self.loss_val
        def grad(self,x1):
            assert self.loss_val is not None
            grad_val = np.copy(self.grad_val)
            self.loss_val = None
            self.grad_val = None
            return grad_val

    print('... training')

    archiver.cleanDir(archiver.CURRDIR)
    with open(archiver.getFilePath('properties.json'), 'w') as f:
        json.dump(hyperparameters, f)

    try:
        for i in range(n_epochs):
            # print sum(blank)
            if optimizer=='l-bfgs':
                e = loss_grad_eval()
                new_im, loss, temp = scipy.optimize.fmin_l_bfgs_b(e.loss, x0.flatten(), fprime=e.grad, maxfun=lbfgs_maxfun)
                new_im = np.reshape(new_im, content_shape)
                blank_sh.set_value(new_im.astype(np.float32))
                x0 = blank_sh.get_value().astype(np.float32)
            else:
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

train_style(0.2, 2e-4, 'test_images/tubingen_small.jpg', 'test_images/shipwreck.jpg', blank_image_path=None,
                style_layers = ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1'],
                content_layers = ['conv4_2'], n_epochs=10,learning_rate=10,resize=False,optimizer='l-bfgs',lbfgs_maxfun=20)
