# coding=utf-8
import json
import cPickle
from collections import namedtuple
import os

word_info = namedtuple('word_info', ['word', 'count'])

base_path = '/media/dl/expand/ai_challenger_caption_train_20170902/'

with open(os.path.join(base_path, 'caption_train_annotations_cut.json'), 'r') as f:
    data = json.load(f)
# ensure that there are five captions for each image
for _, cap in enumerate(data):
    assert len(cap['caption']) == 5

words_list = []
for i, sample in enumerate(data):
    caps = sample['caption']
    for _, cap in enumerate(caps):
        words_list.extend(cap.split(' '))
words_set = list(set(words_list))
words_dict = []
for word in words_set:
    word_count = words_list.count(word)
    words_dict.append(word_info(word.encode('utf-8'), word_count))
    # print('the word {} occured {} times'.format(word.encode('utf-8'), word_count))
print('Total number of words is {}'.format(len(words_set)))

# sort along the number of each word
print(words_dict[:10])
words_dict = sorted(words_dict, key=lambda x: -x.count)
print(words_dict[:10])

with open(os.path.join(base_path, 'words_dict.pkl'), 'w') as f:
    cPickle.dump(words_dict, f)
