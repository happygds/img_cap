# encoding=utf-8
import json
import jieba
import sys
import os
import cPickle as pickle
reload(sys)
sys.setdefaultencoding('utf-8')


folder_p = {'train': 'caption_train_images_20170902',
            'val': 'caption_validation_images_20170902', 'test': 'caption_test_images_20170902'}


def prepro_cap(jsonfile, split, pre_list, startidx):
    with open(jsonfile, 'r') as f:
        caps_ori = json.load(f)
    annotations = []
    images = []
    for img_id, cap_ori in enumerate(caps_ori):
        # image_id = img_id + startidx
        img_item = {}
        image = {}
        # img_item['filepath'] = folder_p[split]
        img_item['filename'] = cap_ori['image_id']
        # img_item['imgid'] = image_id
        # img_item['split'] = split
        # image['file_name'] = cap_ori['image_id']
        # image['id'] = image_id
        sentences = []
        # sid_list = []
        for sentenid, sentence in enumerate(cap_ori['caption']):
            # annotation = {}
            # sid = image_id * 5 + sentenid
            # sent_item = {}
            # cut words
            # text = jieba.cut(sentence.encode('utf-8'), cut_all=False)
            # text = [i for i in text]
            # if len(text) == 0:
            #     continue
            # sent_item['raw'] = sentence
            # sent_item['imgid'] = image_id
            # sent_item['sentid'] = sid
            # sid_list.append(sid)
            sentences.append(sentence)
            # annotation['image_id'] = image_id
            # annotation['id'] = sid
            # annotation['caption'] = ' '.join(text)
            # annotations.append(annotation)
        img_item['sentences'] = sentences
        # img_item['cocoid'] = image_id
        # img_item['sentids'] = sid_list
        pre_list.append(img_item)
        images.append(image)
    return pre_list, annotations, images


if __name__ == '__main__':
    save_path = './dataset_val_ai.json'
    # pre_list, val_ano, val_img = prepro_cap('./caption_train_annotations_20170902.json', 'train', [], 0)
    pre_list, val_ano, val_img = prepro_cap('./caption_validation_annotations_20170902.json', 'val', [], 0)

    # name2idx = {image['filename']: image['imgid'] for image in pre_list}
    # idx2name = [image['filename'] for image in pre_list]
    data_cap = {}
    data_cap['images'] = pre_list
    data_cap['dataset'] = 'ai'
    with open(save_path, 'w') as f:
        json.dump(data_cap, f, ensure_ascii=False, indent=4, separators=(',', ':'))
