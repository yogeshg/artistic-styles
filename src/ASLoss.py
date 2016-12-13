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

from PIL import Image

def preprocess_image(paths):
    # if single given not list of strings then convert to list of strings
    if isinstance(paths, basestring):
        paths = [paths]
        
    images = []
    for path in paths:
        image = Image.open(path,'r').convert('RGB')
        w,h = image.size
        # resize so smallest dimenison = 256 (preserve aspect ratio)
        if h<w:
            image = image.resize((w*256/h,256))
        else:
            image = image.resize((256,h*256/w))
        # crop the images to 224x224
        #get the new shape of the image
        w,h = image.size
        #get the bounds of the box
        right = w//2 + 112
        left =w//2 - 112
        top = h//2 - 112
        bottom = h//2 + 112
        image = image.crop((left,top,right,bottom))
        im = numpy.asarray(image)
        imcopy = numpy.zeros(im.shape)
        imcopy[:,:,0] = im[:, :, 0] - 103.939
        imcopy[:,:,1] = im[:, :, 1] - 116.779
        imcopy[:,:,2] = im[:, :, 2] - 123.68
        #RGB -> BGR
        imcopy = imcopy[:, :, ::-1]
        #put channels first
        imcopy = numpy.rollaxis(imcopy,2,0)
        #add dimension to make it a 4d image (for theano tensor)
        imcopy = numpy.expand_dims(imcopy,axis=0)
        #store it in images array
        if len(images)==0:
            images = imcopy
        else:
            images = numpy.append(images,imcopy,axis=0)
    return images

def deprocess_image(image_array):
    # put channels last
    assert image_array.ndim==4
    image_array = numpy.rollaxis(image_array,1,4)
    #BGR -> RGB
    image_array = image_array[:,:, :, ::-1]
    #add mean channel
    image_array_copy = numpy.zeros(image_array.shape)
    image_array_copy[:,:,:,0] = image_array[:,:, :, 0] + 103.939
    image_array_copy[:,:,:,1] = image_array[:,:, :, 1] + 116.779
    image_array_copy[:,:,:,2] = image_array[:,:, :, 2] + 123.68
    #convert to int between 0 and 254
    image_array_copy = numpy.clip(image_array_copy, 0, 255).astype('uint8')
    return image_array_copy
