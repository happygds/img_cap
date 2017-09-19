import h5py
import numpy as np
import pickle
import math
import os
from multiprocessing.dummy import Pool as ThreadPool

pool = ThreadPool(8)

splits = 7

ori_feats_path = './data/train/train.features.h5'
features = h5py.File(ori_feats_path, 'r')
num = int(math.ceil(float(len(features)) / float(splits)))
assert num == 30000, '{}({}) != 30000'.format(num, type(num))

with open('./data/train/train.file.names.pkl', 'rb') as f:
    fnames = pickle.load(f)
    fnames = np.asarray([os.path.split(fname)[1] for fname in fnames])

split_feats_tmp = np.zeros((num, 121, 1536), dtype=np.float32)
# split the names to 7 files
for s in range(splits):
    split_fnames = fnames[s * num:(s + 1) * num]
    split_feats = split_feats_tmp[:len(split_fnames)]
    for j, s_fname in enumerate(split_fnames):
        split_feats[j] = features[s_fname][:]

    print('split {} finished'.format(s))
    with h5py.File('./data/train/train_split{}.features.h5'.format(s), 'w') as f:
        f.create_dataset("features", data=split_feats, chunks=True)

features.close()
