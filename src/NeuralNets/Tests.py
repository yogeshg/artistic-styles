import random
import numpy as np

from Utils import load_data, rotate_image, translate_image, flip_image, plot16, noise_injection, train_nn
from Models import MyLeNet

def test_lenet( batch_size=10       ,
                n_epochs=200        ,
                learning_rate=0.1   ,
                rotation=False, translation=False, flipping=False, noise=None):
    # print 'test_lenet:', locals()
    rng = np.random.RandomState(23455)

    datasets = load_data(theano_shared=False)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    N = train_set_x.shape[0]
    print N

    if( flipping ):
        temp = [flip_image(train_set_x[i]) for i in range(train_set_x.shape[0])]
        train_set_x = np.concatenate([train_set_x, temp])
        train_set_y = np.concatenate([train_set_y, train_set_y])
        plot16(temp, './imgs-flip-aug.png')
        plot16(train_set_x, './imgs-flip-orig.png')

    if( rotation ):
        temp = [rotate_image(train_set_x[i], (MAX_ROTATE)-random.random()*(2*MAX_ROTATE)) for i in range(int(train_set_x.shape[0]))]
        train_set_x = np.concatenate([train_set_x, temp])
        train_set_y = np.concatenate([train_set_y, train_set_y])
        plot16(temp, './imgs-rotate-aug.png')
        plot16(train_set_x, './imgs-rotate-orig.png')

    if( translation ):
        temp = [translate_image(train_set_x[i], (MAX_TRANSLATE-random.random()*(2*MAX_TRANSLATE),MAX_TRANSLATE-random.random()*(2*MAX_TRANSLATE))) for i in range(train_set_x.sha/e[0])]
        train_set_x = np.concatenate([train_set_x, temp])
        train_set_y = np.concatenate([train_set_y, train_set_y])
        plot16(temp, './imgs-translate-aug.png')
        plot16(train_set_x, './imgs-translate-orig.png')

    if( noise ):
        temp = [noise_injection(train_set_x[i], magnitude=MAX_NOISE, method=noise) for i in range(train_set_x.shape[0])]
        train_set_x = np.concatenate([train_set_x, temp])
        train_set_y = np.concatenate([train_set_y, train_set_y])
        plot16(temp, './imgs-'+str(noise)+'Noise-aug.png')
        plot16(train_set_x, './imgs-'+str(noise)+'Noise-orig.png')

    # compute number of minibatches for training, validation and testing
    # n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    # n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    # n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches = train_set_x.shape[0]
    n_valid_batches = valid_set_x.shape[0]
    n_test_batches = test_set_x.shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    datasets[0] = (train_set_x, train_set_y)
    datasets[1] = (valid_set_x, valid_set_y)
    datasets[2] = (test_set_x, test_set_y  )
 
    myLeNet = MyLeNet(rng, datasets, batch_size=batch_size, learning_rate=learning_rate)
    print myLeNet
    train_nn(myLeNet.train_model, myLeNet.validate_model, myLeNet.test_model_restore,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            datasets, batch_size,
            verbose = True)


