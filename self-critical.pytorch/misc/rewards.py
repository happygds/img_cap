from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import misc.utils as utils
from collections import OrderedDict
import torch
from torch.autograd import Variable

import sys
sys.path.append("/media/dl/expand/caption_eval/coco_caption")
from pycxevalcap.ciderD.ciderD import CiderD
from pycxevalcap.bleu.bleu import Bleu
from pycxevalcap.rouge.rouge import Rouge
from pycxevalcap.meteor.meteor import Meteor
from pycxevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import jieba
import json
import pdb
import cPickle as pickle
import os
reload(sys)
sys.setdefaultencoding('utf-8')
tokenizer = PTBTokenizer()


itow = json.load(open('./data/aitalk.json', 'r'))['ix_to_word']
wtoi = {w: i for i, w in itow.items()}
vocb = set([w for i, w in itow.items()])


def init_cider_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD(sigma=4., df=pickle.load(open(os.path.join('data', cached_tokens + '.p'), 'r')))


def evaluate_captions_mix(gts, res, tokenize=False):
    if tokenize:
        for i, gt in gts.iteritems():
            gts[i] = [{'caption': x} for x in gt]
            res[i] = [{'caption': res[i][0]}]
        # print(res[0], gts[0])
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (CiderD_scorer, "CIDEr"),
        # (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]
    final_scores = {}
    for scorer, method in scorers:
        if type(method) == list:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        else:
            score, scores = scorer.compute_score(gts, res)
        if type(score) == list:
            for m, s in zip(method, scores):
                final_scores[m] = np.asarray(s)
                assert len(s) == len(res)
        else:
            final_scores[method] = np.asarray(scores)
            assert len(scores) == len(res)
    return final_scores['CIDEr'] + 1.8 * final_scores['ROUGE_L'] \
        + 0.8 * final_scores['Bleu_4'] + 0.7 * final_scores['Bleu_3'] + \
        0.3 * final_scores['Bleu_2'] + 0.3 * final_scores['Bleu_1']
    # return final_scores['METEOR']


def array_to_str(arr):
    # original version
    out = ''
    out_jieba = ''
    for i in range(len(arr)):
        if arr[i] == 0:
            break
        out += str(arr[i]) + ' '
        out_jieba += itow[str(arr[i])]

    # jieba version
    out_jieba = out_jieba.strip()
    out_jieba = jieba.cut(out_jieba.encode('utf-8'), cut_all=False)

    out_jieba = [str(wtoi[w]) if w in vocb else '0' for w in out_jieba]
    if len(out_jieba) > 0:
        p = 1.0 - out_jieba.count('0') * 1. / float(len(out_jieba))
    else:
        p = 1.

    # if len(out_jieba) > 0 and (out_jieba[-1] == str(wtoi[u'\u5728']) or out_jieba[-1] == str(wtoi[u'\u7684']) or out_jieba[-1] == str(wtoi[u'\u7ed9'])):
    #     out_jieba += [out_jieba[-1], ] * (len(arr) - len(out_jieba))
    #     out += str(arr[i]) + ' '

    # # if doesnot have un-recognized word, use the jieba cut results
    # if '0' not in out_jieba:
    out = ' '.join(out_jieba)

    if len(out) == 0:
        out = '0'
        # out_words = '.'
    assert len(out) > 0

    return out.strip(), p


def get_self_critical_reward(model, fc_feats, att_feats, data, gen_result, only_cider=True):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])

    # get greedy decoding baseline
    greedy_res, _ = model.sample(Variable(fc_feats.data, volatile=True), Variable(att_feats.data, volatile=True))

    res = OrderedDict()

    gen_result = gen_result.cpu().numpy()
    greedy_res = greedy_res.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])[0]]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])[0]]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j])[0] for j in range(len(data['gts'][i]))]

    # _, scores = Bleu(4).compute_score(gts, res)
    # scores = np.array(scores[3])
    # res = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    # gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}

    res = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}

    if only_cider:
        _, scores = CiderD_scorer.compute_score(gts, res)
    else:
        scores = evaluate_captions_mix(gts, res)
        _ = scores.mean()
    print('Metric scores:', _)

    metric_scores = scores[:batch_size] - scores[batch_size:]
    # metric_scores = metric_scores * (metric_scores >= 0.)
    # metric_scores = np.exp(scores[batch_size:] - 7.2) * metric_scores

    rewards = np.repeat(metric_scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards


def c2f_get_self_critical_reward(model, fc_feats, att_feats, data,
                                 gen_result, gen_result_fine, gen_result_coarse,
                                 only_cider=True):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])

    # get greedy decoding baseline
    greedy_res, _, greedy_res_fine, _, greedy_res_coarse, _ = model.sample(
        Variable(fc_feats.data, volatile=True), Variable(att_feats.data, volatile=True))

    res = OrderedDict()

    gen_result = gen_result.cpu().numpy()
    greedy_res = greedy_res.cpu().numpy()
    # gen_result_fine = gen_result_fine.cpu().numpy()
    # greedy_res_fine = greedy_res_fine.cpu().numpy()
    # gen_result_coarse = gen_result_coarse.cpu().numpy()
    # greedy_res_coarse = greedy_res_coarse.cpu().numpy()

    res_discount = np.zeros((6 * batch_size,), dtype='float32')

    for i in range(batch_size):
        tmp1, tmp2 = array_to_str(gen_result[i])
        res[i] = [tmp1]
        res_discount[i] = tmp2
        tmp1, tmp2 = array_to_str(greedy_res[i])
        res[batch_size + i] = [tmp1]
        res_discount[batch_size + i] = tmp2

        # tmp1, tmp2 = array_to_str(gen_result_fine[i])
        # res[2 * batch_size + i] = [tmp1]
        # res_discount[2 * batch_size + i] = tmp2
        # tmp1, tmp2 = array_to_str(greedy_res_fine[i])
        # res[3 * batch_size + i] = [tmp1]
        # res_discount[3 * batch_size + i] = tmp2

        # tmp1, tmp2 = array_to_str(gen_result_coarse[i])
        # res[4 * batch_size + i] = [tmp1]
        # res_discount[4 * batch_size + i] = tmp2
        # tmp1, tmp2 = array_to_str(greedy_res_coarse[i])
        # res[5 * batch_size + i] = [tmp1]
        # res_discount[5 * batch_size + i] = tmp2

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j])[0] for j in range(len(data['gts'][i]))]

    res = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    if only_cider:
        _, scores = CiderD_scorer.compute_score(gts, res)
    else:
        scores = evaluate_captions_mix(gts, res)
    # scores *= res_discount
    print('Metric scores:', scores.mean())

    scores_final = scores[:batch_size] - scores[batch_size: 2 * batch_size]
    # # scores_final = scores[:batch_size] - \
    # #     np.maximum(scores[batch_size: 2 * batch_size], scores[2 * batch_size: 3 * batch_size])
    # scores_fine = scores[2 * batch_size: 3 * batch_size] - scores[3 * batch_size: 4 * batch_size]
    # scores_coarse = scores[4 * batch_size: 5 * batch_size] - scores[5 * batch_size: 6 * batch_size]

    rewards = np.repeat(scores_final[:, np.newaxis], gen_result.shape[1], 1)
    # rewards_fine = np.repeat(scores_fine[:, np.newaxis], gen_result_fine.shape[1], 1)
    # rewards_coarse = np.repeat(scores_coarse[:, np.newaxis], gen_result_coarse.shape[1], 1)

    rewards_fine = 0. * rewards
    rewards_fine = 0. * rewards

    return rewards, rewards_fine, rewards_coarse