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

'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def reconstructed_image(D,c,num_coeffs,X_mean,n_blocks,im_num):
    '''
    This function reconstructs an image X_recon_img given the number of
    coefficients for each image specified by num_coeffs
    '''
    
    '''
        Parameters
    ---------------
    c: np.ndarray
        an n x m matrix representing the coefficients of all the image blocks.
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
        

    n_blocks: Integer
        number of blocks comprising the image in each direction.
        For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4
    '''
    c_im = c[:num_coeffs,n_blocks*n_blocks*im_num:n_blocks*n_blocks*(im_num+1)]
    D_im = D[:,:num_coeffs]

    X_approx = np.dot(c_im.T, D_im.T)
    logging.debug('shape of single approximated image matrix is:' +str(X_approx.shape))
    # logging.debug('the matrix is\n'+str(X_approx))
    blockSize = int(D.shape[0]**0.5)
    n_pixels_img = blockSize * n_blocks

    X_recon_img = np.zeros((n_pixels_img, n_pixels_img), dtype=np.dtype('float32'))
    b = 0
    for (slice_i, slice_j) in getBlockSlices(n_blocks, blockSize):
        X_recon_img[slice_i, slice_j] = X_approx[b,:].reshape(blockSize, blockSize)
        b+=1

    X_recon_img + np.tile( X_mean, (n_blocks,n_blocks) );
    
    #TODO: Enter code below for reconstructing the image X_recon_img
    #......................
    #......................
    #X_recon_img = ........
    return X_recon_img

def getBlockSlices(numBlocks, blockSize):
    for b in range(numBlocks**2):
        block_i = b%numBlocks
        block_j = b/numBlocks
        slice_i = slice(block_i*blockSize, (block_i+1)*blockSize)
        slice_j = slice(block_j*blockSize, (block_j+1)*blockSize)
        yield(slice_i, slice_j)

def plot_reconstructions(D,c,num_coeff_array,X_mean,n_blocks,im_num):
    '''
    Plots 9 reconstructions of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        num_coeff_array: Iterable
            an iterable with 9 elements representing the number of coefficients
            to use for reconstruction for each of the 9 plots
        
        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        n_blocks: Integer
            number of blocks comprising the image in each direction.
            For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4

        X_mean: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        im_num: Integer
            index of the image to visualize
    '''
    logging.info('plotting reconstruction for image %d blocks %d', im_num, n_blocks)
    f, axarr = plt.subplots(3,3)
    for i in range(3):
        for j in range(3):
            plt.axes(axarr[i,j])
            plt.imshow(reconstructed_image(D,c,num_coeff_array[i*3+j],X_mean,n_blocks,im_num), cmap=cm.Greys_r)
            
    f.savefig('output/hw1a_{0}_im{1}.png'.format(n_blocks, im_num))
    plt.close(f)
    
    
    
def plot_top_16(D, blockSize, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image block of shape (blockSize, blockSize)

    Parameters
    -------------
    D: np.ndarray
        N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    blockSize: Integer
        The height and width of each block

    imname: string
        name of file where image will be saved.
    '''
    #TODO: Obtain top 16 components of D and plot them
    
    logging.info('plotting top 16 components for block size %d', blockSize)
    f, axarr = plt.subplots(4,4)
    for i in range(4):
        for j in range(4):
            plt.axes(axarr[i,j])
            plt.imshow(D[i*4+j,:].reshape((blockSize, blockSize)), cmap=cm.Greys_r)
            
    f.savefig(imname)
    plt.close(f)

IMG_LOCATION_FORMAT = './Fei_256/image{i}.jpg'
NUM_IMAGES = 200

def getImageFileNames(num_images=NUM_IMAGES):
    return [ IMG_LOCATION_FORMAT.format(i=i) for i in range(num_images)]

IMAGE_SIZE = 256
IMAGE_DATA_TYPE = np.dtype('uint8')

def getImagesRaw(imageFiles, dtype=IMAGE_DATA_TYPE):
    numImgs = len(imageFiles)
    X = np.zeros( (numImgs,IMAGE_SIZE, IMAGE_SIZE), dtype=dtype )
    for i in range( numImgs ):
        im = Image.open(imageFiles[i])
        X[i,:,:] = np.matrix( im )
    return X

def convertRawImages2blockMatrix( X, blockSize ):
    numImgs = len(X)
    numCols = blockSize**2
    numBlocks = IMAGE_SIZE / blockSize
    numRows = numImgs * (numBlocks**2)
    logging.debug('creating a matrix of size (%d,%d) and type %s',numRows, numCols, str(IMAGE_DATA_TYPE))
    mat = np.zeros(( numRows, numCols ), dtype=IMAGE_DATA_TYPE)
    i = 0
    for imgId in range(numImgs):
        for (slice_i, slice_j) in getBlockSlices(numBlocks, blockSize):
            mat[i,:] = X[imgId,slice_i,slice_j].flatten()
            i+=1
    return mat 

def main():
    '''
    Read here all images(grayscale) from Fei_256 folder
    into an numpy array Ims with size (no_images, height, width).
    Make sure the images are read after sorting the filenames
    '''
    #TODO: Read all images into a numpy array of size (no_images, height, width)
    X_raw = getImagesRaw( getImageFileNames() )
    
    blockSizes = [8, 32, 64]
    num_coeffs = [range(1, 10, 1), range(3, 30, 3), range(5, 50, 5)]

    for blockSize, nc in zip(blockSizes, num_coeffs):
        print blockSize, nc
        '''
        Divide here each image into non-overlapping blocks of shape (blockSize, blockSize).
        Flatten each block and arrange all the blocks in a
        (no_images*n_blocks_in_image) x (blockSize*blockSize) matrix called X
        ''' 
        
        X = convertRawImages2blockMatrix( X_raw, blockSize )
        logging.debug('shape of matrix of all images is:'+str(X.shape))
        # logging.debug('the matrix is\n'+str(X))
        
        X_mean = np.mean(X, 0)
        X = X - np.repeat(X_mean.reshape(1, -1), X.shape[0], 0)

        '''
        Perform eigendecomposition on X^T X and arrange the eigenvectors
        in decreasing order of eigenvalues into a matrix D
        '''
        
        w,v = np.linalg.eig(np.dot(X.T,X))
        idx = np.argsort( -w )
        D = v[:,idx]
        
        c = np.dot(D.T, X.T)
        
        for i in range(0, NUM_IMAGES, 10):
            plot_reconstructions(D=D, c=c, num_coeff_array=nc, X_mean=X_mean.reshape((blockSize, blockSize)), n_blocks=int(256/blockSize), im_num=i)

        plot_top_16(D, blockSize, imname='output/hw1a_top16_{0}.png'.format(blockSize))


if __name__ == '__main__':
    main()
    
    
