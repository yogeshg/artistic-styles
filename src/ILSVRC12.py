import pandas as pd
import numpy as np

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
        self.labels = list(self.df.label.map(label2enum).map(lambda x: onehot(x,10)))

    def getNimages(self, n_data):
        paths = self.paths[:n_data]
        labels = self.labels[:n_data]
        x,sh = ip.preprocess_image(paths)
        x = x.astype(np.float32)
        y = np.array(labels).astype(np.float32)
        return (x,y)

