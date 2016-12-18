import logging
logging.basicConfig(level = logging.DEBUG)
import matplotlib.pyplot as plt

from NeuralNets.Models import VGG_19
from NeuralNets.ImportParameters import load_layer_params

from ILSVRC12 import ILSVRC12

import numpy as np

rng = np.random.RandomState(23455)

print 'loading parameters...'

p = load_layer_params('imagenet-vgg-verydeep-19.mat')

print 'creating vgg19...'


image_shape = (3,224,224)
batch_size=4

v = VGG_19( rng, None, p['filter_shape'], weights=p['weights'], bias=p['bias'], batch_size=batch_size)

imagenet = ILSVRC12()
for i in range(5):
    x,y = imagenet.getImageBatch(i,batch_size)
    x = x.reshape(batch_size, x.size/batch_size)
    error = v.train_model(x,y)
    print error

