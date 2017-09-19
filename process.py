# encoding=utf-8
import caffe
import numpy as np
import caffe
import cv2
import h5py
import os
caffe.set_device(0)
caffe.set_mode_gpu()

train_image_folder = '/media/dl/expand/ai_challenger_caption_train_20170902/caption_train_images_20170902/'
valid_image_folder = '/media/dl/expand/ai_challenger_caption_train_20170902/caption_validation_images_20170902/'
data_path = '/media/dl/expand/data/'


def generate_ftoid(image_folder):
    filelist = os.popen('ls ' + image_folder).readlines()
    filenames = [image[:-1] for image in filelist]
    return filenames


model_path = '/media/dl/expand/data/'
model_def = model_path + 'deploy_inception-resnet-v2.prototxt.txt'
model_weights = model_path + 'inception-resnet-v2.caffemodel'

mean_value = np.array([128.0, 128.0, 128.0])
std = np.array([128.0, 128.0, 128.0])

net = caffe.Net(model_def, model_weights, caffe.TEST)


def image_preprocess(img):
    b, g, r = cv2.split(img)
    return cv2.merge([(b - mean_value[0]) / std[0], (g - mean_value[1]) / std[1], (r - mean_value[2]) / std[2]])


def generate_feats(prefix, outfilename):
    if prefix == 'train':
        image_folder = train_image_folder
    else:
        image_folder = valid_image_folder
    image_paths = generate_ftoid(image_folder)
    h5f = h5py.File(outfilename, mode='a')
    keys = set(h5f.keys())
    image_list = [image_folder + x for x in image_paths if x not in keys]
    print len(image_list), '-------------'
    n_examples = len(image_list)
    if n_examples == 0:
        return
    batch_size = 10
    count = 0
    for start, end in zip(range(0, n_examples, batch_size),
                          range(batch_size, n_examples + batch_size, batch_size)):
        count += 1
        image_batch_file = image_list[start:end]
        tmp = map(lambda x: image_preprocess(cv2.resize(cv2.imread(x),
                                                        (395, 395), interpolation=cv2.INTER_CUBIC)), image_batch_file)
        image_batch = np.asarray(tmp).astype(np.float32)
        print image_batch.shape
        n_sampled = image_batch.shape[0]
        image_batch = np.transpose(image_batch, (0, 3, 1, 2))
        if n_sampled != batch_size:
            print 'batch size small'
            image_batch_tmp = np.zeros((batch_size, 395, 395, 3))
            image_batch_tmp[0:n_sampled, :, :, :] = image_batch
            image_batch = image_batch_tmp
        net.blobs['data'].data[:] = image_batch
        net.forward()
        feats = net.blobs['conv6_1x1'].data
        print feats.shape
        feats = np.resize(feats, (feats.shape[0], feats.shape[1], feats.shape[2] * feats.shape[3]))
        print feats.shape
        feats = np.transpose(feats, (0, 2, 1))
        if n_sampled != batch_size:
            feats = feats[0:n_sampled, :, :]
        for _, (img_pth, feat) in enumerate(zip(image_batch_file, feats)):
            fname = os.path.split(img_pth)[1]
            print fname, feat.shape
            h5f[fname] = feat

        print ("Processed %d %s features.." % (end, prefix))


generate_feats('train', '/media/dl/expand/data/train/train.features.h5')
# generate_feats('validation', '/media/dl/expand/data/val/val.features.h5')
