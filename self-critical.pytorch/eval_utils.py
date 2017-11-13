from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import cPickle as pickle
import json
from json import encoder
import random
import string
import time
import pdb
import os
import sys
import misc.utils as utils
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append("./caption_eval/coco-caption/")
from pycxtools.coco import COCO
from pycxevalcap.eval import COCOEvalCap


idx2names = pickle.load(open('./data/idx2name.pkl'))


def compute_m1(json_predictions_file, reference_file):
    """Compute m1_score"""
    m1_score = {}
    m1_score['error'] = 0
    try:
        coco = COCO(reference_file)
        coco_res = coco.loadRes(json_predictions_file)

        # create coco_eval object.
        coco_eval = COCOEvalCap(coco, coco_res)

        # evaluate results
        coco_eval.evaluate()
    except Exception:
        m1_score['error'] = 1
    else:
        # print output evaluation scores
        for metric, score in coco_eval.eval.items():
            print('%s: %.3f' % (metric, score))
            m1_score[metric] = score
    return m1_score

def language_eval(dataset, preds, model_id, split):
    annFile = "./data/val_ref.json"

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('./eval_results'):
        os.mkdir('./eval_results')
    cache_path = os.path.join('./eval_results/', model_id + '_' + split + '.json')

    # coco = COCO(annFile)
    # valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    # preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds), len(preds)))
    json.dump(preds, open(cache_path, 'w'),
              ensure_ascii=False, indent=4, separators=(',', ':'))  # serialize to temporary json file. Sigh, COCO API...

    print(cache_path)
    out = compute_m1(cache_path, annFile)

    # cocoRes = coco.loadRes(cache_path)
    # cocoEval = COCOEvalCap(coco, cocoRes)
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # cocoEval.evaluate()

    # # create output dictionary
    # out = {}
    # for metric, score in cocoEval.eval.items():
    #     out[metric] = score

    # imgToEval = cocoEval.imgToEval
    # for p in preds_filt:
    #     image_id, caption = p['image_id'], p['caption']
    #     imgToEval[image_id]['caption'] = caption
    # with open(cache_path, 'w') as outfile:
    #     json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)
    print(out)

    return out


def eval_split(model, crit, loader, eval_kwargs={}, model_id=None):
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    caption_model = eval_kwargs.get('caption_model', 'c2ftopdown')

    if model_id is None:
        model_id = eval_kwargs['id']

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    if caption_model == 'c2ftopdown' or caption_model == 'c2fada':
        predictions_fine = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None:
            # forward the model to get loss
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
            tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks = tmp

            loss = crit(model(fc_feats, att_feats, labels), labels[:, 1:], masks[:, 1:]).data[0]
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        fc_feats, att_feats = tmp
        # forward the model to also get generated samples for each image
        if caption_model == 'c2ftopdown' or caption_model == 'c2fada':
            seq, _, seq_fine, _ = model.sample(fc_feats, att_feats, eval_kwargs)
        else:
            seq, _ = model.sample(fc_feats, att_feats, eval_kwargs)

        # set_trace()
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        if caption_model == 'c2ftopdown' or caption_model == 'c2fada':
            sents_fine = utils.decode_sequence(loader.get_vocab(), seq_fine)
        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if caption_model == 'c2ftopdown' or caption_model == 'c2fada':
                entry_fine = {'image_id': data['infos'][k]['id'], 'caption': sents_fine[k]}
                predictions_fine.append(entry_fine)
            if eval_kwargs.get('dump_path', 0) == 1:
                entry_fine['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + \
                    '" vis/imgs/img' + str(len(predictions)) + '.jpg'  # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                if int(entry['image_id']) % 200 == 0:
                    print('image %s: %s' % (entry['image_id'], entry['caption']))
                    if caption_model == 'c2ftopdown' or caption_model == 'c2fada':
                        print('image %s: %s' % (entry['image_id'], entry_fine['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()
            predictions_fine.pop()

        if verbose:
            if ix0 % 2000 == 0:
                print('evaluating validation preformance... %d/%d (%f)' % (ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    for i, pred in enumerate(predictions):
        name = idx2names[pred['image_id']]
        cap = ''.join(pred['caption'].split(' ')).replace(' ', '')
        predictions[i]['image_id'] = name.split('.')[0]
        predictions[i]['caption'] = cap

    if caption_model == 'c2ftopdown' or caption_model == 'c2fada':
        for i, pred in enumerate(predictions_fine):
            name = idx2names[pred['image_id']]
            cap = ''.join(pred['caption'].split(' ')).replace(' ', '')
            predictions_fine[i]['image_id'] = name.split('.')[0]
            predictions_fine[i]['caption'] = cap

    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, str(model_id), split)
        if caption_model == 'c2ftopdown' or caption_model == 'c2fada':
            lang_stats_fine = language_eval(dataset, predictions_fine, str(model_id) + '_fine', split)

    # Switch back to training mode
    model.train()
    if caption_model == 'c2ftopdown' or caption_model == 'c2fada':
        return loss_sum / loss_evals, predictions, [lang_stats, lang_stats_fine]
    else:
        return loss_sum / loss_evals, predictions, lang_stats
