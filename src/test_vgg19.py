from NeuralNets.Models import VGG_19
from NeuralNets.ImportParameters import load_layer_params
import numpy as np

rng = np.random.RandomState(23455)

print 'loading parameters...'

p = load_layer_params('imagenet-vgg-verydeep-19.mat')

print 'creating vgg19...'

v = VGG_19( rng, None, p['filter_shape'], p['pool_shape'])


