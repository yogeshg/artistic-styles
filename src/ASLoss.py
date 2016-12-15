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
    assert Input.ndim==3

    F_flat = T.reshape(X,(X.shape[0],X.shape[1]*X.shape[2]))
    gram = T.dot(X_flat, X_flat.T)
    return gram

def getStyleLoss(Fl,al,N,M,wl):
    Al = gram_matrix(al)
    Gl = gram_matrix(Fl)
    El = ((1/(2.*N*M))**2)*(T.sum(T.pow(Gl-Al,2)))
    return wl*loss


def total_loss(generated_vgg,style_vgg,content_vgg,style_layers,
                content_layer,alpha,beta,style_weights,filter_shape):
    loss = T.scalar('loss')
    # add content loss
    Fl = getattr(generated_vgg,content_layer).output
    Pl = getattr(content_vgg,content_layer).output
    loss+=alpha * getContentLoss(Fl,Pl)
    for layer in style_layers:
        Fl = getattr(generated_vgg,layer).output
        al = getattr(style_vgg,layer).output
        wl = style_weights[layer]
        N =filter_shape[layer][1]
        m = Fl.get_value().shape
        M =m[-1]*m[-2]
        loss += beta*getStyleLoss(Fl,al,N,M,wl)
