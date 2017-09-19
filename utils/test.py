# coding=utf-8
import cPickle
from collections import namedtuple
import json
import numpy as np
import os

base_path = '/media/dl/expand/ai_challenger_caption_train_20170902'
word_info = namedtuple('word_info', ['word', 'count'])


with open(os.path.join(base_path, 'words_dict.pkl'), 'r+') as f:
    words_dict = cPickle.load(f)
    print(len(words_dict))
words_dict = words_dict[:1000:40] + words_dict[1000::200]

for i, word in enumerate(words_dict):
    print('the word {} occured {} times'.format(word.word, word.count))
    # lq = len(unicode(word.word, 'utf-8'))
    # print(lq)

with open(os.path.join(base_path, 'caption_train_annotations_cut.json'), 'r') as f:
    data = json.load(f)

cap_lens = []
for _, sample in enumerate(data):
    caps = sample['caption']
    cap_len = [len(x.split(' ')) for x in caps]
    cap_lens.extend(cap_len)
cap_lens = np.asarray(cap_lens).ravel()
print('the max length of cutted captions is {}, \n min is {}, mean is {}, std is {} |'.format(
    np.max(cap_lens), np.min(cap_lens), np.mean(cap_lens), np.std(cap_lens)))

# compute the 64-words proportion
cap_prop = (cap_lens <= 25).mean() * 100.
print('the captions of length <= 25 occupy {}%\ of the training set |'.format(cap_prop))
cap_prop = (cap_lens <= 32).mean() * 100.
print('the captions of length <= 32 occupy {}%\ of the training set |'.format(cap_prop))
