# coding=utf-8
import cPickle
from collections import namedtuple
import json
import numpy as np
import os

base_path = '/media/dl/expand/ai_challenger_caption_train_20170902'
word_info = namedtuple('word_info', ['word', 'count'])


with open('/media/dl/expand/data/train/word_to_idx.pkl', 'rb') as f:
    words_dict = cPickle.load(f)

with open('/media/dl/expand/data/train/word_counts.pkl', 'rb') as f:
    words_count = cPickle.load(f)

words_list = []
print(type(words_dict.keys()))
for i, key in enumerate(words_dict.keys()):
    count = words_count[i]
    words_list.append(word_info(key.encode('utf-8'), count))

print(words_list[:10])
words_list = sorted(words_list, key=lambda x: -x.count)
print(words_list[:10])

for i, (word, count) in enumerate(words_list[:1000:10] + words_list[1000::200]):
    print('word {} occurred {} times'.format(word, count))

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
