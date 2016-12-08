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

    X = Input.dimshuffle([2, 0, 1])
    F_flat = T.reshape(X,(X.shape[0],X.shape[1]*X.shape[2]))
    gram = T.dot(X_flat, X_flat.T)
    return gram

def getStyleLoss(Fl,al,N,M,wl):
    Al = gram_matrix(al)
    Gl = gram_matrix(Fl)
    El = ((1/(2.*N*M))**2)*(T.sum(T.pow(Gl-Al,2)))
    return wl*loss


def total_loss(F,a,P,style_layers,
                content_layer,alpha,beta,style_weights):
    loss = T.scalar('loss')
    # add content loss
    Fl = F[content_layer]
    Pl = P[content_layer]
    loss+=alpha * getContentLoss(Fl,Pl)
    for layer in style_layers:
        Fl = F[layer]
        al = a[layer]
        wl = style_weights[layer]
        N =...
        M =...
        loss += beta*getStyleLoss(Fl,al,N,M,wl)
        
