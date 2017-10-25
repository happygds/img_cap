# coding=utf-8
import json
import jieba
import cPickle as pickle
import os
import hashlib
import numpy as np
import thulac
thu1 = thulac.thulac(seg_only=True)

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' % path)


split_list = ['val']

for split in split_list:
    base_path = '/media/dl/expand/'
    with open(os.path.join(base_path, './data/{}/{}.file.names.pkl'.format(split, split)), 'rb') as f:
        fnames = pickle.load(f)
        fnames = np.asarray([os.path.split(fname)[1] for fname in fnames])
        fnames = set(fnames)

    if split == 'val':
        split_name = 'validation'
    else:
        split_name = split

    with open(os.path.join(
            base_path, 'ai_challenger_caption_train_20170902/caption_{}_annotations_20170902.json'.format(split_name)),
            'r') as f:
        data = json.load(f)
    # ensure that there are five captions for each image
    for _, sample in enumerate(data):
        assert len(sample['caption']) == 5

    # using jieba to cut words
    base_dict = {}

    # read captions for convertion processing
    count_id = 1
    info_list = []
    for i, sample in enumerate(data):
        if i % 10000 == 0:
            print('processing {} samples'.format(i))
        if sample['image_id'] in fnames:
            file_name = sample['image_id'][:-4]
            file_hash = int(int(hashlib.sha256(file_name).hexdigest(), 16) % sys.maxint)
            # if len(sample['caption']) > 0:
            for _, cap in enumerate(sample['caption']):
                # text = thu1.cut(cap.encode('utf-8'), text=True)
                # cap_list = unicode(text, 'utf-8')
                cap_list = jieba.cut(cap, cut_all=False)
                cut_cap = ' '.join(cap_list)
                info_list.append([file_name, file_hash, count_id, cut_cap])
                count_id += 1

    # convert the information into a dict
    base_dict['annotations'] = []
    base_dict['images'] = []
    for _, info in enumerate(info_list):
        annot_dict = {}
        annot_dict['caption'] = info[3]
        annot_dict['id'] = info[2]
        annot_dict['image_id'] = info[1]
        base_dict['annotations'].append(annot_dict)

        img_dict = {}
        img_dict['file_name'] = info[0]
        img_dict['id'] = info[1]
        base_dict['images'].append(img_dict)

    base_dict['info'] = {
        "contributor": "He Zheng",
        "description": "CaptionEval",
        "url": "https://github.com/AIChallenger/AI_Challenger.git",
        "version": "1",
        "year": 2017
    }
    base_dict['licenses'] = [{
        "url": "https://challenger.ai"
    }]
    base_dict['type'] = "captions"

    # save result
    with open(os.path.join(base_path, 'caption_eval/{}_ref.json'.format(split)), 'w') as f:
        json.dump(base_dict, f, indent=4, separators=(',', ':'))

    with open(os.path.join(base_path, 'caption_eval/{}_ref.json'.format(split)), 'r') as f:
        ref_json = json.load(f)['annotations']
        ref_hashes = np.asarray([x['image_id'] for x in ref_json], dtype='int64')
        ref_json = np.asarray(ref_json)

    ref = {}
    for i, fname in enumerate(fnames):
        fname = os.path.split(fname)[1][:-4]
        ref_caps = []
        fhash = int(int(hashlib.sha256(fname).hexdigest(), 16) % sys.maxint)
        ref_ids = ref_json[ref_hashes == fhash]
        for _, ref_id in enumerate(ref_ids):
            assert fhash == ref_id['image_id'], "{} not equal {}".format(fhash, ref_id['image_id'])
            tmp = {}
            tmp['caption'] = ref_id['caption']
            ref_caps.append(tmp)
        ref[i] = ref_caps

    save_pickle(ref, os.path.join(base_path, 'data/%s/%s.references.pkl' % (split, split)))
