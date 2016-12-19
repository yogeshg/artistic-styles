import sys, os
try:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
except:
    pass

import logging
logging.basicConfig(level = logging.INFO)
import timeit
import cPickle

import numpy as np
import matplotlib.pyplot as plt

from NeuralNets.Models import VGG_19
from NeuralNets.ImportParameters import load_layer_params
from ILSVRC12 import ILSVRC12
import archiver

rng = np.random.RandomState(23455)

print 'loading parameters...'

p = load_layer_params('imagenet-vgg-verydeep-19.mat')

print 'creating vgg19...'


image_shape = (3,224,224)
batch_size=16

v = VGG_19( rng, None, p['filter_shape'], batch_size=batch_size)

imagenet = ILSVRC12()
# for i in range(5):
#     x,y = imagenet.getImageBatch(i,batch_size)
#     x = x.reshape(batch_size, x.size/batch_size)
#     error = v.train_model(x,y)
#     print error

n_epochs = 1000
verbose = True

n_total_batches = imagenet.num_data / batch_size
n_train_batches = int(n_total_batches * 0.6)
n_test_batches = int((n_total_batches - n_train_batches) / 2)
n_valid_batches = int(n_total_batches - n_train_batches - n_test_batches)

print('batch_size:', batch_size)
print('n_epochs:', n_epochs)
print('verbose:', verbose)
print('n_total_batches:', n_total_batches)
print('n_train_batches:', n_train_batches)
print('n_test_batches:', n_test_batches)
print('n_valid_batches:', n_valid_batches)

best_validation_loss = np.inf
best_iter = 0
test_score = 0.

start_time = timeit.default_timer()
epoch = 0
done_looping = False

archiver.cleanDir(archiver.CURRDIR)

while (epoch < n_epochs) and (not done_looping):
    try:
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)

            x,y = imagenet.getImageBatch(minibatch_index, batch_size)
            x = x.reshape(batch_size, x.size/batch_size)
            cost_ij = v.train_model(x,y)

            if (iter + 1) % n_train_batches == 0:

                # compute zero-one loss on validation set
                vl = []
                for validation_index in range(n_valid_batches):
                    x,y = imagenet.getImageBatch( n_train_batches+n_test_batches+validation_index , batch_size)
                    x = x.reshape(batch_size, x.size/batch_size)
                    vl.append(v.validate_model(x,y))

                this_validation_loss = np.mean(vl)

                if verbose:
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    # if this_validation_loss < best_validation_loss * improvement_threshold:
                    #     patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    filepath = archiver.getFilePath('vgg_{:04d}.cPickle'.format(iter))
                    with open(filepath, 'w') as f:
                        cPickle.dump(v.params, f)

                    tl = []
                    for test_index in range(n_test_batches):
                        x,y = imagenet.getImageBatch( n_train_batches+test_index, batch_size)
                        x = x.reshape(batch_size, x.size/batch_size)
                        tl.append(v.test_model(x,y))

                    # test it on the test set
                    test_score = np.mean(tl)

                    if verbose:
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1,
                               n_train_batches,
                               test_score * 100.))
    except Exception, e:
        logging.error(e)
        break

archiver.archiveDir(archiver.CURRDIR)
            # if patience <= iter:
            #     done_looping = True
            #     break

end_time = timeit.default_timer()


# Print out summary
print('Optimization complete.')
print('Best validation error of %f %% obtained at iteration %i, '
      'with test performance %f %%' %
      (best_validation_loss * 100., best_iter + 1, test_score * 100.))
sys.stderr.write(('The training process ran for %.2fm' % ((end_time - start_time) / 60.)))

