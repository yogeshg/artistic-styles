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
    X = Input
    X_flat = T.reshape(X,(X.shape[0],X.shape[1]*X.shape[2]))
    gram = T.dot(X_flat, X_flat.T)
    return gram

def getStyleLoss(Fl,al,N,M,wl):
    Al = gram_matrix(al)
    Gl = gram_matrix(Fl)
    El = ((1/(2.*N*M))**2)*(T.sum(T.pow(Gl-Al,2)))
    return wl*El

def total_loss(style_image,content_image,generated_image,vgg,style_layers,content_layer,alpha,beta,filter_shape):
    loss = T.scalar('loss')
    # add content loss
    Fl = getattr(vgg,content_layer).output.eval({v.x:generated_image})[0]
    Pl = getattr(vgg,content_layer).output.eval({v.x:style_image})[0]
    loss+=alpha * getContentLoss(Fl,Pl)
    for layer in style_layers:
        Fl = getattr(vgg,layer).output.eval({v.x:generated_image})[0]
        al = getattr(vgg,layer).output.eval({v.x:style_image})[0]
        wl = 0.2
        N =filter_shape[-3]
        M =filter_shape[-1]*shape_shape[-2]
        loss += beta*getStyleLoss(Fl,al,N,M,wl)
    return loss

def gram_matrix_numpy(Input):
    assert Input.ndim==3
    X = Input
    X_flat = np.reshape(X,(X.shape[0],X.shape[1]*X.shape[2]))
    gram = np.dot(X_flat, X_flat.T)
    return gram

def getContentLoss_numpy(Fl,Pl):
    loss = 0.5*(np.sum(np.power(Fl-Pl,2)))
    return loss

def getStyleLoss_numpy(Fl,al,N,M,wl):
    Al = gram_matrix_numpy(al)
    Gl = gram_matrix_numpy(Fl)
    El = ((1/(2.*N*M))**2)*(np.sum(np.power(Gl-Al,2)))
    return wl*El

def total_loss_numpy(style_image,content_image,generated_image,vgg,style_layers,content_layer,alpha,beta):
    loss = 0
    # add content loss
    Fl = getattr(vgg,content_layer).output.eval({v.x:generated_image})[0]
    Pl = getattr(vgg,content_layer).output.eval({v.x:content_image})[0]
    loss+=alpha * getContentLoss_numpy(Fl,Pl)
    print loss
    for layer in style_layers:
        Fl = getattr(vgg,layer).output.eval({v.x:generated_image})[0]
        al = getattr(vgg,layer).output.eval({v.x:style_image})[0]
        wl = 0.2
        N =Fl.shape[0]
        M =Fl.shape[-1]*Fl.shape[-2]
        loss += beta*getStyleLoss_numpy(Fl,al,N,M,wl)
        print loss
    return loss
