
# encoding=utf-8
import json
import jieba
import sys
import os
import cPickle as pickle
from collections import Counter
import jieba.posseg as pseg
reload(sys)
sys.setdefaultencoding('utf-8')


folder_p = {'train': 'caption_train_images_20170902',
            'val': 'caption_validation_images_20170902', 'test': 'caption_test_images_20170902'}


def prepro_cap(jsonfile):
    with open(jsonfile, 'r') as f:
        caps_ori = json.load(f)
    flags = {}
    for img_id, cap_ori in enumerate(caps_ori):
        for sentenid, sentence in enumerate(cap_ori['caption']):
            # cut words
            text = pseg.cut(sentence.encode('utf-8'))
            if len(sentence) == 0:
                continue
            else:
                for word, flag in text:
                    if flag not in flags.keys():
                        flags[flag] = Counter([word])
                    else:
                        flags[flag].update([word])

    # print the most common 10 elements for each flag
    for key, item in flags.iteritems():
        print('the word flag {} '.format(key), item.most_common(10))
    return flags


if __name__ == '__main__':
    # flags = prepro_cap('./caption_train_annotations_20170902.json')
    # pickle.dump(flags, file('./flags.pkl', 'w'))
    flags = pickle.load(file('./flags.pkl', 'r'))
    num = 0
    for key, item in flags.iteritems():
        for word, count in item.items():
            if count > 3:
                num += 1
    print(num)
