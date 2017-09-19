import h5py
import numpy as np
import time
import pickle
import os
from multiprocessing.dummy import Pool as ThreadPool

pool = ThreadPool(8)

tmp1 = time.time()
features = h5py.File('./data/train/train_split0.features.h5', 'r')['features'][:]
print(type(features), features.shape)
tmp2 = time.time()
print('time used {} s'.format(tmp2 - tmp1))


# with open('./data/train/train.file.names.pkl', 'rb') as f:
#     fnames = pickle.load(f)

# inds = np.random.randint(len(fnames), size=64)
# inds = sorted(list(inds))
# inds = fnames[inds]
# inds = [os.path.split(x)[1] for x in inds]
# # inds = inds[:500] + inds[:200] + inds[400:700]
# # print(inds)
# data2 = np.zeros((len(inds), 121, 1536))

# # tmp1 = time.time()
# # data1 = np.expand_dims(features[inds[0]][:], axis=0)
# # for _, ind in enumerate(inds[::1][1:]):
# #     tmp = np.expand_dims(features[ind][:], axis=0)
# #     data1 = np.concatenate([data1, tmp], axis=0)
# tmp2 = time.time()
# # print('time used {} s'.format(tmp2 - tmp1))

# names = inds
# # def read(names):
# #     for i, name in enumerate(names):
# #         data2[i] = features[name][:]
# #     return None
# data2 = np.asarray(pool.map(lambda x: features[x][:], names))
# # read(names)
# # data2 = np.asarray([features[name][:] for _, name in enumerate(names)])
# # data2 = np.take(data2, sort_inds, axis=0)
# tmp3 = time.time()
# print('time used {} s'.format(tmp3 - tmp2))
# # assert np.all(data1 == data2)
# # assert np.any(data1 == data2)

# print(data2.shape)
# pool.close()
# features.close()
