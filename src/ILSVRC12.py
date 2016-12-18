import logging

import pandas as pd
import numpy as np

from NeuralNets.Utils import about
import imageProcess as ip

def onehot(i, n):                                     
    r = np.zeros(n)
    r[i] = 1
    return r

class ILSVRC12():
    def __init__(self, selected_files='./ILSVRC12_selected_files.csv'):
        self.df = pd.read_csv(selected_files)
        idx = range(len(self.df))
        np.random.seed(123)
        np.random.shuffle(idx)
        self.df = self.df.loc[idx,:]
        label2enum = dict(zip(list(self.df.label.unique()),range(100)))
        self.paths = list(self.df.path)
        # self.labels = list(self.df.label.map(label2enum).map(lambda x: onehot(x,10)))
        self.labels = list(self.df.label.map(label2enum))
        self.logger = logging.getLogger(self.__class__.__name__)

    def getImageBatch(self, batch_num, batch_size):
        paths = self.paths[batch_num*batch_size:(batch_num+1)*batch_size]
        labels = self.labels[batch_num*batch_size:(batch_num+1)*batch_size]
        x,sh = ip.preprocess_image(paths)
        x = x.astype(np.float32)
        y = np.array(labels).astype(np.float32)
        self.logger.info('x: '+about(x))
        self.logger.info('y: '+about(y))
        return (x,y)

