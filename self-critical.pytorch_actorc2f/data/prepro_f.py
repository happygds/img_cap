import h5py
import cPickle as pickle 
import numpy as np


def transfer_npz(h5name, name2idx, pre_folder):
    h5file = h5py.File(h5name, 'r')
    for k, v in h5file.iteritems():
        idx = name2idx[k]
        np.savez_compressed(pre_folder + str(idx), feat=v)


if __name__ == '__main__':
    validh5file = '../../data/test/test.bufeatures.h5'
    name2idx = pickle.load(file('./name2idx.pkl', 'r'))
    pre_folder = './aitalk/'
    transfer_npz(validh5file, name2idx, pre_folder)
