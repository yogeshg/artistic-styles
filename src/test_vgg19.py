import logging
logging.basicConfig(level = logging.DEBUG)

import matplotlib.pyplot as plt
from NeuralNets.Models import VGG_19
from NeuralNets.ImportParameters import load_layer_params
from imageProcess import preprocess_image

import numpy as np

rng = np.random.RandomState(23455)

print 'loading parameters...'

p = load_layer_params('imagenet-vgg-verydeep-19.mat')

print 'creating vgg19...'

v = VGG_19( rng, None, p['filter_shape'], weights=p['weights'], bias=p['bias'])
i = np.reshape(preprocess_image('test_images/thais.JPG'),(1,3*224*224)) # (1,3,224,224)
i = i.astype(np.float32)

o1 = v.conv1_1.output.eval({v.x:i})
o2 = v.conv2_1.output.eval({v.x:i})
o3 = v.conv3_1.output.eval({v.x:i})
o4 = v.conv4_1.output.eval({v.x:i})
o5 = v.conv5_1.output.eval({v.x:i})

print o1[0].shape
print o2[0].shape
print o3[0].shape
print o4[0].shape
print o5[0].shape

plt.imsave('./data/conv1_1.jpg', o1[0][0])
plt.imsave('./data/conv2_1.jpg', o2[0][0])
plt.imsave('./data/conv3_1.jpg', o3[0][0])
plt.imsave('./data/conv4_1.jpg', o4[0][0])
plt.imsave('./data/conv5_1.jpg', o5[0][0])



