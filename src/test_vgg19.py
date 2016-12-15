from NeuralNets.Models import VGG_19
from NeuralNets.ImportParameters import load_layer_params
from imageProcess import preprocess_image

import numpy as np

rng = np.random.RandomState(23455)

print 'loading parameters...'

p = load_layer_params('imagenet-vgg-verydeep-19.mat')

print 'creating vgg19...'

v = VGG_19( rng, None, p['filter_shape'])
i = np.reshape(preprocess_image('test_images/thais.JPG'),(1,3*224*224)) # (1,3,224,224)

o = v.conv1_1.output.eval( {v.x: i}  )
print o.shape

import cPickle

with open('output.dat','w') as f:
    cPickle.dump(o, f)
