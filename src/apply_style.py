from NeuralNets.Models import VGG_19
from NeuralNets.ImportParameters import load_layer_params
from imageProcess import preprocess_image
from ASLoss import total_loss

import numpy as np
import theano
import theano.tensor as T

def train_style(alpha, beta, content_image_path, style_image_path, blank_image_path,
                style_layers = ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1'],
                content_layer = 'conv4_2', n_epochs=10):

    rng = np.random.RandomState(23455)

    print 'loading parameters...'

    p = load_layer_params('imagenet-vgg-verydeep-19.mat')

    print 'creating vgg19...'

    v = VGG_19( rng, None, p['filter_shape'])

    style = np.reshape(preprocess_image(style_image_path), (1, 3 * 224 * 224))  # (1,3,224,224)

    content = np.reshape(preprocess_image(content_image_path), (1, 3 * 224 * 224))  # (1,3,224,224)

    blank = np.reshape(preprocess_image(blank_image_path), (1, 3 * 224 * 224))  # (1,3,224,224)

    loss = total_loss(style, content, blank, v, style_layers, content_layer, alpha, beta, p['filter_shape'])

    grad = T.grad(loss, blank)

    updates = [
        (blank, blank - learning_rate * grad)
    ]

    train_model = theano.function(
        [blank],
        loss,
        updates=updates,
        givens={
            x: blank
        }
    )

    print('... training')

    for i in range(n_epochs):
        loss = train_model(blank)
        print (loss)

    return loss


train_style(0.5, 0.5, 'test_images/thais.JPG', 'test_images/starry_night_google.jpg', 'test_images/whitenoise.jpeg',
                style_layers = ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1'],
                content_layer = 'conv4_2', n_epochs=10)
