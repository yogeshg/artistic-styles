from NeuralNets.Models import VGG_19
from NeuralNets.ImportParameters import load_layer_params
from imageProcess import preprocess_image
from ASLoss import total_loss

import numpy as np
import theano
import theano.tensor as T

#test_images/starry_night_google.jpg'
#'test_images/thais.JPG'

def train_style(content_image_path, style_image_path, blank_image_path):
    rng = np.random.RandomState(23455)

    print 'loading parameters...'

    p = load_layer_params('imagenet-vgg-verydeep-19.mat')

    print 'creating vgg19...'

    v = VGG_19( rng, None, p['filter_shape'])

    style = np.reshape(preprocess_image(style_image_path), (1, 3 * 224 * 224))  # (1,3,224,224)

    s1 = v.conv1_1.output.eval({v.x: style})
    s2 = v.conv2_1.output.eval({v.x: style})
    s3 = v.conv3_1.output.eval({v.x: style})
    s4 = v.conv4_1.output.eval({v.x: style})
    s5 = v.conv5_1.output.eval({v.x: style})

    content = np.reshape(preprocess_image(content_image_path), (1, 3 * 224 * 224))  # (1,3,224,224)

    c1 = v.conv4_2.output.eval({v.x: content})

    blank = np.reshape(preprocess_image(blank_image_path), (1, 3 * 224 * 224))  # (1,3,224,224)

    loss = total_loss()

    grad = T.grad(loss, blank)

    updates =

    train_model = theano.function(
        [],
        cost,
        updates=updates,
        givens={
            x: blank
        }
    )

    print('... training')

