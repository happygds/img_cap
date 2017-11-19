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
        image_id = img_id + startidx
        img_item = {}
        image = {}
        img_item['filepath'] = folder_p[split]
        img_item['filename'] = cap_ori['image_id']
        img_item['imgid'] = image_id
        img_item['split'] = split
        image['file_name'] = cap_ori['image_id']
        image['id'] = image_id
        sentences = []
        sid_list = []
        for sentenid, sentence in enumerate(cap_ori['caption']):
            annotation = {}
            sid = image_id * 5 + sentenid
            sent_item = {}
            # cut words
            sentence = sentence.replace('\n', '').replace(' ', '').strip()
            text = jieba.cut(sentence.encode('utf-8'), cut_all=False)
            text = [i for i in text]
            if len(text) == 0:
                continue
            sent_item['tokens'] = text
            sent_item['raw'] = sentence
            sent_item['imgid'] = image_id
            sent_item['sentid'] = sid
            sid_list.append(sid)
            sentences.append(sent_item)
            annotation['image_id'] = image_id
            annotation['id'] = sid
            annotation['caption'] = ' '.join(text)
            annotations.append(annotation)
        img_item['sentences'] = sentences
        img_item['cocoid'] = image_id
        img_item['sentids'] = sid_list
        pre_list.append(img_item)
        images.append(image)
    return pre_list, annotations, images


def prepro_test(imgpth, split, pre_list, startidx):
    # with open(imgpkl, 'r') as f:
    #     imgnames = pickle.load(f)
    #     imgnames = [os.path.split(imgname)[1] for imgname in imgnames]
    imgnames = os.listdir(imgpth)
    for img_id, imagename in enumerate(imgnames):
        image_id = img_id + startidx
        img_item = {}
        img_item['filepath'] = folder_p[split]
        img_item['filename'] = imagename
        img_item['imgid'] = image_id
        img_item['split'] = split
        sentences = []
        sid_list = []
        for sentenid, sentence in enumerate(['的']):
            sid = image_id * 5 + sentenid
            sent_item = {}
            # cut words
            sentence = sentence.replace('\n', '').replace(' ', '').strip()
            text = jieba.cut(sentence.encode('utf-8'), cut_all=False)
            text = [i for i in text]
            if len(text) == 0:
                continue
            sent_item['tokens'] = text
            sent_item['raw'] = sentence
            sent_item['imgid'] = image_id
            sent_item['sentid'] = sid
            sid_list.append(sid)
            sentences.append(sent_item)
        img_item['sentences'] = sentences
        img_item['cocoid'] = image_id
        img_item['sentids'] = sid_list
        pre_list.append(img_item)
    return pre_list


if __name__ == '__main__':
    save_path = './dataset_ai.json'
    pre_list, train_ano, train_img = prepro_cap('./caption_train_annotations_20170902.json', 'train', [], 0)
    pre_list, val_ano, val_img = prepro_cap('./caption_validation_annotations_20170902.json', 'val', pre_list, len(pre_list))
    pre_list = prepro_test('../../ai_challenger_caption_train_20170902/caption_test1_images_20170923', 'test', pre_list, len(pre_list))
    name2idx = {}
    idx2name = []
    for i in xrange(210000):
        name2idx[pre_list[i]['filename'] + '_train'] = pre_list[i]['imgid']
        idx2name.append(pre_list[i]['filename'] + '_train')
    for i in xrange(210000, 240000):
        name2idx[pre_list[i]['filename'] + '_val'] = pre_list[i]['imgid']
        idx2name.append(pre_list[i]['filename'] + '_val')
    for i in xrange(240000, 270000):
        name2idx[pre_list[i]['filename'] + '_test'] = pre_list[i]['imgid']
        idx2name.append(pre_list[i]['filename'] + '_test')
    data_cap = {}
    data_cap['images'] = pre_list
    data_cap['dataset'] = 'ai'
    with open(save_path, 'w') as f:
        json.dump(data_cap, f, ensure_ascii=False, indent=4, separators=(',', ':'))

    pickle.dump(name2idx, file('./name2idx.pkl', 'w'))
    pickle.dump(idx2name, file('./idx2name.pkl', 'w'))

    # licenses = [{u'url': u'http://creativecommons.org/licenses/by-nc-sa/2.0/', u'id': 1, u'name': u'Attribution-NonCommercial-ShareAlike License'}, {u'url': u'http://creativecommons.org/licenses/by-nc/2.0/', u'id': 2, u'name': u'Attribution-NonCommercial License'}, {u'url': u'http://creativecommons.org/licenses/by-nc-nd/2.0/', u'id': 3, u'name': u'Attribution-NonCommercial-NoDerivs License'}, {u'url': u'http://creativecommons.org/licenses/by/2.0/', u'id': 4, u'name': u'Attribution License'}, {u'url': u'http://creativecommons.org/licenses/by-sa/2.0/', u'id': 5, u'name': u'Attribution-ShareAlike License'}, {u'url': u'http://creativecommons.org/licenses/by-nd/2.0/', u'id': 6, u'name': u'Attribution-NoDerivs License'}, {u'url': u'http://flickr.com/commons/usage/', u'id': 7, u'name': u'No known copyright restrictions'}, {u'url': u'http://www.usa.gov/copyright.shtml', u'id': 8, u'name': u'United States Government Work'}]
    # info = {u'description': u'This is stable 1.0 version of the 2014 MS COCO dataset.', u'url': u'http://mscoco.org', u'version': u'1.0', u'year': 2014, u'contributor': u'Microsoft COCO group', u'date_created': u'2015-01-27 09:11:52.357475'}
    # # with open('./val_ref.json', 'w') as f:
    # print len(val_ano)
    #     #json.dump({'annotations':val_ano, 'images': val_img, 'type':'captions', 'licenses': licenses, 'info': info}, f, ensure_ascii=False, indent=4, separators=(',', ':'))

    # with open('./train_ref.json', 'w') as f:
    #     print len(train_ano)
    #     json.dump({'annotations':train_ano, 'images': train_img, 'type':'captions', 'licenses': licenses, 'info': info}, f, ensure_ascii=False, indent=4, separators=(',', ':'))
