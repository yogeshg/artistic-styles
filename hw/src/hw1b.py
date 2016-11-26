import os
from os import walk
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.linalg as linalg

from PIL import Image

import theano
import theano.tensor as T

import logging
logging.basicConfig(level=logging.DEBUG)

from hw1a import IMG_LOCATION_FORMAT, NUM_IMAGES, IMAGE_SIZE, IMAGE_DATA_TYPE
from hw1a import plot_top_16, getImageFileNames, getImagesRaw, convertRawImages2blockMatrix

from gradient_descent import GradientDescent


'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''
VALIDATE_EPSILON = 10
EPSILON = 1e-50
MAX_ITERATIONS = 500

def reconstructed_image(D,c,num_coeffs,X_mean,im_num):
    '''
    This function reconstructs an image given the number of
    coefficients for each image specified by num_coeffs
    '''
    
    '''
        Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the image blocks.
        n represents the maximum dimension of the PCA space.
        m is (number of images x n_blocks**2)

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)

    im_num: Integer
        index of the image to visualize

    X_mean: np.ndarray
        a matrix representing the mean block.

    num_coeffs: Integer
        an integer that specifies the number of top components to be
        considered while reconstructing
    '''
    
    c_im = c[:num_coeffs,im_num]
    D_im = D[:,:num_coeffs]

    X_approx = np.dot(c_im.T, D_im.T).reshape(X_mean.shape) + X_mean
    logging.debug('shape of single approximated image matrix is:' +str(X_approx.shape))

    #TODO: Enter code below for reconstructing the image
    #......................
    #......................
    #X_recon_img = ........
    return X_approx

def plot_reconstructions(D,c,num_coeff_array,X_mean,im_num):
    '''
    Plots 9 reconstructions of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        num_coeff_array: Iterable
            an iterable with 9 elements representing the number_of coefficients
            to use for reconstruction for each of the 9 plots
        
        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        X_mean: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        im_num: Integer
            index of the image to visualize
    '''
    f, axarr = plt.subplots(3,3)
    for i in range(3):
        for j in range(3):
            plt.axes(axarr[i,j])
            plt.imshow(reconstructed_image(D,c,num_coeff_array[i*3+j],X_mean,im_num)
                        , cmap=cm.Greys_r
                        , interpolation='none')
            
    f.savefig('output/hw1b_{0}.png'.format(im_num))
    plt.close(f)

    
def main():
    '''
    Read here all images(grayscale) from Fei_256 folder and collapse 
    each image to get an numpy array Ims with size (no_images, height*width).
    Make sure the images are read after sorting the filenames
    '''
    X_raw = getImagesRaw( getImageFileNames() )
    Ims = convertRawImages2blockMatrix( X_raw, 256 )

    Ims = Ims.astype(np.float32)
    X_mn = np.mean(Ims, 0)
    X = Ims - np.repeat(X_mn.reshape(1, -1), Ims.shape[0], 0)

    '''
    Use theano to perform gradient descent to get top 16 PCA components of X
    Put them into a matrix D with decreasing order of eigenvalues

    If you are not using the provided AMI and get an error "Cannot construct a ufunc with more than 32 operands" :
    You need to perform a patch to theano from this pull(https://github.com/Theano/Theano/pull/3532)
    Alternatively you can downgrade numpy to 1.9.3, scipy to 0.15.1, matplotlib to 1.4.2
    '''
    
    gd1 = GradientDescent( n=256*256, eta=0.5, numEig=16)
    w,v = gd1.getAllEigenValuesAndVectors(X, epsilon=EPSILON, max_iteraions=MAX_ITERATIONS, validateEpsilon=VALIDATE_EPSILON)

    D = v
    c = np.dot(D.T, X.T)
    logging.info('shape of c:%s, D:%s', str(c.shape), str(D.shape))
        
    for i in range(0, 200, 10):
        plot_reconstructions(D=D, c=c, num_coeff_array=[1, 2, 4, 6, 8, 10, 12, 14, 16], X_mean=X_mn.reshape((256, 256)), im_num=i)

    plot_top_16(D.T, 256, 'output/hw1b_top16_256.png')


if __name__ == '__main__':
    main()
    
    