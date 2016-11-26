
# coding: utf-8

# In[5]:

import scipy


# In[51]:

def createSampleSets(mat_path, decreasing_sizes, sample_path):
    print 'reading from:', mat_path
    test_set = scipy.io.loadmat(mat_path)
    sampleSize = test_set['y'].shape[0]
    print 'size of matrix:', sampleSize
    for sampleSize in decreasing_sizes:
        test_set['X'] = test_set['X'][:,:,:,slice(sampleSize)]
        test_set['y'] = test_set['y'][slice(sampleSize),:]
        sampleSize = test_set['y'].shape[0]
        samplePath1 = sample_path.format(sampleSize)
        print 'saving to:', samplePath1
        scipy.io.savemat(samplePath1, test_set)


# In[52]:

createSampleSets(mat_path = '../data/test_32x32.mat',
                 decreasing_sizes = reversed((10, 100, 500, 1000, None)),
                 sample_path = '../data/test_32x32_{:05d}.mat')


# In[53]:

createSampleSets(mat_path = '../data/train_32x32.mat',
                 decreasing_sizes = reversed((10, 100, 500, 1000, 5000, 10000, None)),
                 sample_path = '../data/train_32x32_{:05d}.mat')


# In[54]:

createSampleSets(mat_path = '../data/extra_32x32.mat',
                 decreasing_sizes = reversed((10, 100, 500, 1000, 5000, 10000, None)),
                 sample_path = '../data/extra_32x32_{:05d}.mat')


# In[ ]:



