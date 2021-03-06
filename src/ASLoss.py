"""
Functions for computing the loss functions for style and content and combining them for the artistic style project

Author Richard Godden rg3047

"""
import timeit
import inspect
import sys
import numpy
import theano
import theano.tensor as T

import logging
logger = logging.getLogger(__file__)

'''
Content representation
'''

def getContentLoss(Fl,Pl):
    loss = 0.5*(T.sum(T.pow(Fl-Pl,2)))
    return loss

'''
Style representation
'''

def gram_matrix(Input):
    assert Input.ndim==3, Input.ndim
    X = Input
    X_flat = T.reshape(X,(X.shape[0],X.shape[1]*X.shape[2]))
    gram = T.dot(X_flat, X_flat.T)
    return gram

def getStyleLoss(Fl,al,N,M,wl):
    Al = gram_matrix(al)
    Gl = gram_matrix(Fl)
    El = ((1/(2.*N*M))**2)*(T.sum(T.pow(Gl-Al,2)))
    return wl*El

# TODO @RG, return a dictionary with keys 'content_loss', 'style_loss', 'total_loss'
# TODO @RG mask will be np matrix, with mask values (think of it right now as left half 0 and right half all 1
# we may want to apply this before or after calculating the gradient -- need to discuss
def total_loss (style_activations, content_activations, v,
                      style_layers, content_layers,
                      alpha, beta, filter_shape, mask=None):
    # def total_loss(style_image,content_image,vgg,style_layers,content_layer,alpha,beta,filter_shape):
    # loss = T.scalar('loss')   # should be intialised(?) to 0
    totalLoss = 0
    styleLoss = 0
    contentLoss = 0
    # add content loss

    assert 'conv4_2' == content_layers[0]
    Fl = v.conv4_2.output                 # depends on v.x
    Pl = content_activations['conv4_2']

    contentLoss = alpha * getContentLoss(Fl, Pl)
    totalLoss += contentLoss
    # Fl = v.conv1_1.output
    # al = style_activations['conv1_1']
    for layer in style_layers:
        Fl = getattr(v, layer, None).output[0]
        al = style_activations[layer][0]
        logger.debug('Fl: '+str(Fl))
        logger.debug('al: '+str(al.shape))
        wl = 1./float(len(style_layers))
        N =filter_shape[layer][-3]
        M =filter_shape[layer][-1]*filter_shape[layer][-2]
        styleLoss += beta*getStyleLoss(Fl,al,N,M,wl)
    totalLoss += styleLoss
    return styleLoss, contentLoss
    

# def gram_matrix_numpy(Input):
#     assert Input.ndim==3
#     X = Input
#     X_flat = np.reshape(X,(X.shape[0],X.shape[1]*X.shape[2]))
#     gram = np.dot(X_flat, X_flat.T)
#     return gram

# def getContentLoss_numpy(Fl,Pl):
#     loss = 0.5*(np.sum(np.power(Fl-Pl,2)))
#     return loss

# def getStyleLoss_numpy(Fl,al,N,M,wl):
#     Al = gram_matrix_numpy(al)
#     Gl = gram_matrix_numpy(Fl)
#     El = ((1/(2.*N*M))**2)*(np.sum(np.power(Gl-Al,2)))
#     return wl*El

# def total_loss_numpy(style_image,content_image,generated_image,vgg,style_layers,content_layer,alpha,beta):
#     loss = 0
#     # add content loss
#     get_layer = theano.function([vgg.x], getattr(vgg,layer).output, allow_input_downcast=True)
#     Fl = get_layer(generated_image)[0]
#     Pl = get_layer(content_image)[0]
#     loss+=alpha * getContentLoss_numpy(Fl,Pl)
#     print loss
#     for layer in style_layers:
#         get_layer = theano.function([vgg.x], getattr(vgg,layer).output, allow_input_downcast=True)
#         Fl = get_layer(generated_image)[0]
#         al = get_layer(style_image)[0]
#         wl = 0.2
#         N =Fl.shape[0]
#         M =Fl.shape[-1]*Fl.shape[-2]
#         loss += beta*getStyleLoss_numpy(Fl,al,N,M,wl)
#         print loss
#     return loss
