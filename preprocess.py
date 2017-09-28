from scipy import ndimage
from collections import Counter
from core.utils import *

import numpy as np
import pandas as pd
import h5py
import os
import json
import caffe
import cv2
train_image_path = '/media/dl/expand/ai_challenger_caption_train_20170902/caption_train_images_20170902/'
val_image_path = '/media/dl/expand/ai_challenger_caption_train_20170902/caption_validation_images_20170902/'
test_image_folder = '/media/dl/expand/ai_challenger_caption_train_20170902/caption_test1_images_20170923/'
caffe.set_device(0)
caffe.set_mode_gpu()

model_path = '/media/dl/expand/data/'
model_def = model_path + 'deploy_inception-resnet-v2.prototxt.txt'
model_weights = model_path + 'inception-resnet-v2.caffemodel'

mean_value = np.array([128.0, 128.0, 128.0])
std = np.array([128.0, 128.0, 128.0])

# net = caffe.Net(model_def, model_weights, caffe.TEST)


def image_preprocess(img):
    b, g, r = cv2.split(img)
    return cv2.merge([(b - mean_value[0]) / std[0], (g - mean_value[1]) / std[1], (r - mean_value[2]) / std[2]])


def _process_caption_data(caption_file, image_dir, max_length, train=True):
    with open(caption_file) as f:
        annotations = json.load(f)

    # data is a list of dictionary which contains 'captions', 'file_name' and 'image_id' as key.
    data = []
    for annotation in annotations:
        annotation['file_name'] = os.path.join(image_dir, annotation['file_name'])
        data += [annotation]

    # convert to pandas dataframe (for later visualization or debugging)
    caption_data = pd.DataFrame.from_dict(data)
    caption_data.sort_values(by='image_id', inplace=True)
    caption_data = caption_data.reset_index(drop=True)

    if train:
        del_idx = []
        for i, caption in enumerate(caption_data['caption']):
            if len(caption.split(" ")) > max_length:
                del_idx.append(i)

        # delete captions if size is larger than max_length
        print "The number of captions before deletion: %d" % len(caption_data)
        caption_data = caption_data.drop(caption_data.index[del_idx])
        caption_data = caption_data.reset_index(drop=True)
        print "The number of captions after deletion: %d" % len(caption_data)
    return caption_data


def _build_vocab(annotations, threshold=1):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations['caption']):
        words = caption.split(' ')  # caption contrains only lower-case words
        for w in words:
            counter[w] += 1

        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))

    vocab = [word for word in counter if counter[word] >= threshold]
    print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))
    v_count = [0, 0, 0, 0] + [counter[word] for word in counter]
    v_count = np.asarray(v_count)

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2, u'<EOP>': 3}
    idx = 4
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print "Max length of caption: ", max_len
    return word_to_idx, v_count


def _build_caption_vector(annotations, word_to_idx, max_length=25):
    n_examples = len(annotations)
    captions = np.ndarray((n_examples, max_length + 2)).astype(np.int32)

    for i, caption in enumerate(annotations['caption']):
        words = caption.split(" ")  # caption contrains only lower-case words
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
        cap_vec.append(word_to_idx['<END>'])

        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>'])

        captions[i, :] = np.asarray(cap_vec[:(max_length + 2)])
    print "Finished building caption vectors"
    return captions


def _build_file_names(annotations):
    image_file_names = []
    id_to_idx = {}
    idx = 0
    image_ids = annotations['image_id']
    file_names = annotations['file_name']
    for image_id, file_name in zip(image_ids, file_names):
        if image_id not in id_to_idx:
            id_to_idx[image_id] = idx
            image_file_names.append(file_name)
            idx += 1

    file_names = np.asarray(image_file_names)
    return file_names, id_to_idx


def _build_image_idxs(annotations, id_to_idx):
    image_idxs = np.ndarray(len(annotations), dtype=np.int32)
    image_ids = annotations['image_id']
    for i, image_id in enumerate(image_ids):
        image_idxs[i] = id_to_idx[image_id]
    return image_idxs


def main():
    # maximum length of caption(number of word). if caption is longer than max_length, deleted.
    max_length = 25
    # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    word_count_threshold = 1

    # about 80000 images and 400000 captions for train dataset
    train_dataset = _process_caption_data(
        caption_file='./data/train/train_captions.json',
        image_dir=train_image_path,
        max_length=max_length)

    # about 40000 images and 200000 captions
    val_dataset = _process_caption_data(
        caption_file='./data/validation/validation_captions.json',
        image_dir=val_image_path,
        max_length=max_length, train=False)
#
    # # about 4000 images and 20000 captions for val / test dataset

    print 'Finished processing caption data'

    save_pickle(train_dataset, 'data/train/train.annotations.pkl')
    save_pickle(val_dataset, 'data/val/val.annotations.pkl')
    #save_pickle(val_dataset[val_cutoff:test_cutoff].reset_index(drop=True), 'data/test/test.annotations.pkl')

    for split in ['train', 'val']:
        annotations = load_pickle('data/%s/%s.annotations.pkl' % (split, split))

        if split == 'train':
            word_to_idx, word_counts = _build_vocab(annotations=annotations, threshold=word_count_threshold)
            save_pickle(word_to_idx, 'data/%s/word_to_idx.pkl' % split)
            save_pickle(word_counts, 'data/%s/word_counts.pkl' % split)

        captions = _build_caption_vector(annotations=annotations, word_to_idx=word_to_idx, max_length=max_length)
        save_pickle(captions, 'data/%s/%s.captions.pkl' % (split, split))

        file_names, id_to_idx = _build_file_names(annotations)
        save_pickle(file_names, 'data/%s/%s.file.names.pkl' % (split, split))

        image_idxs = _build_image_idxs(annotations, id_to_idx)
        save_pickle(image_idxs, 'data/%s/%s.image.idxs.pkl' % (split, split))

        # prepare reference captions to compute bleu scores later
        image_ids = set('none')
        feature_to_captions = {}
        i = -1
        for caption, image_id in zip(annotations['caption'], annotations['image_id']):
            if image_id not in image_ids:
                image_ids.add(image_id)
                i += 1
                feature_to_captions[i] = []
            # caption = ' '.join([x.encode('utf-8') for x in caption.split(' ')])
            tmp = {}
            caption = unicode(caption.encode('utf-8'), 'utf-8')
            tmp['caption'] = caption
            feature_to_captions[i].append(tmp)
        save_pickle(feature_to_captions, 'data/%s/%s.references.pkl' % (split, split))
        print feature_to_captions.items()[0]
        print "Finished building %s caption dataset" % split

    '''
    for split in ['train', 'val', 'test']:
        anno_path = 'data/%s/%s.annotations.pkl' % (split, split)
        save_path = 'data/%s/%s.features.h5' % (split, split)
        annotations = load_pickle(anno_path)
        image_path = list(annotations['file_name'].unique())
        n_examples = len(image_path)

        h5f = h5py.File(save_path, 'w')
        count = 0

        for start, end in zip(range(0, n_examples, batch_size),
                              range(batch_size, n_examples + batch_size, batch_size)):
            count += 1
            image_batch_file = image_path[start:end]
            tmp = map(lambda x: image_preprocess(cv2.resize(cv2.imread(x), (395, 395), interpolation=cv2.INTER_CUBIC)), image_batch_file)
            image_batch = np.asarray(tmp).astype(np.float32)
            print image_batch.shape
            image_batch = np.transpose(image_batch, (0, 3, 1, 2))
            net.blobs['data'].data[:] = image_batch
            net.forward()
            feats = net.blobs['conv6_1x1'].data
            print feats.shape
            feats = np.resize(feats, (feats.shape[0], feats.shape[1], feats.shape[2] * feats.shape[3]))
            print feats.shape
            feats = np.transpose(feats, (0, 2, 1))
            for _, (img_pth, feat) in enumerate(zip(image_batch_file, feats)):
                fname = os.path.split(img_pth)[1]
                print fname, feat.shape
                h5f[fname] = feat

            print ("Processed %d %s features.." % (end, split))

        print ("Saved %s.." % (save_path))
    '''


if __name__ == "__main__":
    main()
