import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-8s %(levelname)-6s %(message)s')
logger = logging.getLogger(__file__)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from NeuralNets.Models import VGG_19
from NeuralNets.ImportParameters import load_layer_params
from imageProcess import preprocess_image,deprocess_image,np2pil
from ASLoss import total_loss

import numpy as np
import scipy
import theano
theano.config.floatX='float32'
import theano.tensor as T

import NeuralNets.Utils as u
from PIL import Image

import archiver
import json

from util import about 

def white_noise(shape=(1,3,224,224)):
    image = np.random.randint(0,255,size=shape)
    image[:,0,:,:] = image[:,0, :, :] - 103.939
    image[:,1,:,:] = image[:,1, :, :] - 116.779
    image[:,2,:,:] = image[:,2, :, :] - 123.68
    image = image[:, ::-1,:, :]
    return image

def default_mask_val(mask_shape):
    mask_val = np.ones((mask_shape)).astype(np.float32)
    num_chanels=mask_shape[1]
    w=mask_shape[2]
    h=mask_shape[3]
    for i in range(num_chanels):
        mask_val[:,i,0:w/2,:] = 0
    return mask_val

def train_style(alpha, beta, content_image_path, style_image_path, style_image_path2, blank_image_path=None,
                style_layers = ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1'],
                content_layers = ['conv4_2'], n_epochs=10, learning_rate=0.000001,
                optimizer='Adam',resize=True,resize_shape=(224,224),style_scale=1.0,lbfgs_maxfun=20,pool2d_mode='max',vgg_train=True):


    vgg19_params = 'imagenet-vgg-verydeep-19.mat'

    hyperparameters = locals()

    rng = np.random.RandomState(23455)

    logger.info('loading imagenet parameters...')

    p = load_layer_params(vgg19_params)

    logger.info('loading images...')
    style_shape = tuple([int(style_scale*x) for x in resize_shape])
    style_image,style_shape = preprocess_image(style_image_path,resize=resize,shape=style_shape)
    style_values = np.reshape(style_image, (style_shape[0], np.prod(style_shape[1:])))

    content_shape = tuple([int(x) for x in resize_shape])
    content_image,content_shape = preprocess_image(content_image_path,resize=resize,shape=content_shape)

    content_values = np.reshape(content_image,(content_shape[0], np.prod(content_shape[1:])))
    
    style_values = style_values.astype( np.float32 )
    content_values = content_values.astype( np.float32 )

    logger.info('creating style Neural Network...')

    v_style = VGG_19(rng, None, p['filter_shape'], weights=p['weights'], bias=p['bias'],image_size=style_shape,pool2d_mode=pool2d_mode,train=vgg_train)

    logger.info('calculating style activations...')

    style_activations={}
    for s_layer in style_layers:
        activation = getattr(v_style,s_layer).output.eval({v_style.x_image:style_values})
        style_activations[s_layer] = activation

    del v_style

    v_style2 = VGG_19(rng, None, p['filter_shape'], weights=p['weights'], bias=p['bias'],image_size=style_shape,pool2d_mode=pool2d_mode,train=vgg_train)
    style_activations2={}
    for s_layer in style_layers:
        activation = getattr(v_style2,s_layer).output.eval({v_style2.x_image:style_values})
        style_activations2[s_layer] = activation
    del v_style2

    logger.info('creating content Neural Network...')

    v = VGG_19(rng,None,p['filter_shape'], weights=p['weights'], bias=p['bias'],image_size=content_shape,pool2d_mode=pool2d_mode,train=vgg_train)

    logger.info('calculating content activations...')

    content_activations={}
    for c_layer in content_layers:
        activation = getattr(v,c_layer).output.eval({v.x_image : content_values})
        content_activations[c_layer] = activation


    style_loss1, content_loss1= total_loss(style_activations, content_activations, v,
                      style_layers, content_layers,
                      alpha, beta, p['filter_shape'])

    style_loss2, content_loss2= total_loss(style_activations2, content_activations, v,
                      style_layers, content_layers,
                      alpha, beta, p['filter_shape'])
    

    loss = style_loss1 + content_loss1
    loss2 = style_loss2 + content_loss2
    # loss : symbolic
    # grad = T.grad(loss, v.x)
    # grad = T.nnet.relu(T.grad(loss, v.x))
    # grad = T.grad(T.nnet.relu(loss), v.x)
    # grad = T.grad((loss), T.nnet.relu(v.x))

    grad1 = T.grad(loss, v.x)
    grad2 = T.grad(loss2, v.x)

    mask_val= None
    if(mask_val is None):
        mask_val = default_mask_val(content_shape)
    mask_val = np.reshape(mask_val,(content_shape[0], np.prod(content_shape[1:])))

    logger.debug(about(content_image))
    logger.debug(about(content_values))
    logger.debug(about(v.x_image))
    logger.debug(about(grad1))
    logger.debug(about(mask_val))
    logger.debug(about(grad2))
    logger.debug(about(1-mask_val))
    grad = grad1*mask_val+grad2*(1-mask_val)
    logger.debug(about(grad))
    raise ValueError("break")

    if blank_image_path==None:
        blank_values = np.reshape(white_noise(content_shape),(content_shape[0], np.prod(content_shape[1:]))).astype(np.float32)
    else:
        blank_image,blank_image_shape = preprocess_image(blank_image_path,resize=resize,shape=content_shape[2:])
        blank_values = np.reshape(blank_image, (content_shape[0], np.prod(content_shape[1:]))).astype(np.float32) 
    blank_sh = theano.shared(blank_values)

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

    logger.info('generating image...')

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

            print 'loss =','%.3e' % loss
            #content_loss = loss_c.eval({v.x:blank_sh.get_value().reshape((1,np.prod(content_shape[1:])))})
            #content_loss_div_alpha = content_loss/alpha
            #style_loss_div_beta = (loss-content_loss)/beta
            #print 'content loss =','%.3e'%content_loss_div_alpha
            #print 'style loss =','%.3e'%style_loss_div_beta
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

if __name__ == '__main__':
    train_style(0.2, 5e-5, 'test_images/tubingen_small.jpg', 'test_images/starry_night_google.jpg', 'test_images/starry_night_google.jpg',
                blank_image_path=None,
                style_layers = ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1'],
                content_layers = ['conv4_2'], n_epochs=10,learning_rate=10,resize=True,resize_shape=(250,250),style_scale=1.111111,
                optimizer='l-bfgs',lbfgs_maxfun=40,pool2d_mode='max',vgg_train=False)
