#!/usr/bin/python
#-*-coding:utf-8 -*-
import thulac
import jieba
import json
import cPickle as pickle
import os
from core.bleu import score_all
import hashlib
import pdb
import sys
from core.utils import save_pickle
import numpy as np
reload(sys)
sys.setdefaultencoding('utf-8')

split_list = ['train', 'val']
for split in split_list:
    with open('./data/{}/{}.file.names.pkl'.format(split, split), 'rb') as f:
        fnames = pickle.load(f)

    # with open('./caption_eval/{}_ref.json'.format(split), 'r') as f:
    #     ref_json = json.load(f)['annotations']
    #     ref_hashes = np.asarray([x['image_id'] for x in ref_json], dtype='int64')
    #     ref_json = np.asarray(ref_json)

    # ref = {}
    # for i, fname in enumerate(fnames):
    #     fname = os.path.split(fname)[1][:-4]
    #     ref_caps = []
    #     fhash = int(int(hashlib.sha256(fname).hexdigest(), 16) % sys.maxint)
    #     ref_ids = ref_json[ref_hashes == fhash]
    #     for _, ref_id in enumerate(ref_ids):
    #         assert fhash == ref_id['image_id'], "{} not equal {}".format(fhash, ref_id['image_id'])
    #         tmp = {}
    #         tmp['caption'] = ref_id['caption']
    #         ref_caps.append(tmp)
    #     ref[i] = ref_caps

    # save_pickle(ref, 'data/%s/%s.references.pkl' % (split, split))

with open('./caption_eval/val_result.json', 'r') as f:
    result = json.load(f)


reference_path = os.path.join('./data/val/val.references.pkl')
with open(reference_path, 'rb') as f:
    ref = pickle.load(f)


hypo = {}
refe = {}
for i, fname in enumerate(fnames):
    fname = os.path.split(fname)[1][:-4]
    assert fname == result[i]['image_id'], "{} not equal {}".format(fname, result[i]['image_id'])
    caption = result[i]['caption']
    caption = jieba.cut(caption, cut_all=False)
    caption = ' '.join(caption)
    hypo[i] = [{'caption': caption}]
    # refe[i] = [{'caption': cap} for cap in ref[i]]
    refe[i] = ref[i]
    # print caption, refe[i][1]['caption']

# compute bleu score
final_scores = score_all(refe, hypo)
print final_scores
