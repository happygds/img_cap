import cPickle as pickle
import os
import sys
sys.path.append('../coco-caption')
from caption_eval.coco_caption.pycxevalcap.bleu.bleu import Bleu
from caption_eval.coco_caption.pycxevalcap.rouge.rouge import Rouge
from caption_eval.coco_caption.pycxevalcap.meteor.meteor import Meteor
from caption_eval.coco_caption.pycxevalcap.cider.cider import Cider
# from caption_eval.coco_caption.pycxevalcap.spice.spice import Spice
from caption_eval.coco_caption.pycxevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import pdb
import numpy as np
import jieba


def score_all(ref, hypo):
    print('tokenization...')
    tokenizer = PTBTokenizer()
    ref = tokenizer.tokenize(ref)
    hypo = tokenizer.tokenize(hypo)

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Meteor(), "METEOR"),
        (Cider(), "CIDEr")
        # (Spice(), "SPICE")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    return final_scores


def evaluate_for_particular_captions(cand, data_path='./data', split='val', get_scores=False):
    reference_path = os.path.join(data_path, "%s/%s.references.pkl" % (split, split))

    # load caption data
    with open(reference_path, 'rb') as f:
        ref = pickle.load(f)
    # with open(candidate_path, 'rb') as f:
    #     cand = pickle.load(f)

    # make dictionary
    hypo = {}
    refe = {}

    for i, caption in enumerate(cand):
        caption = ''.join(caption.split(' ')).strip().replace('.', '')
        caption = jieba.cut(caption.encode('utf-8'), cut_all=False)
        caption = ' '.join(caption)
        # caption = unicode(caption.encode('utf-8'), 'utf-8')
        hypo[i] = [{'caption': caption}]
        # refe[i] = [{'caption': cap} for cap in ref[i]]
        refe[i] = ref[i]

    # compute bleu score
    final_scores = score_all(refe, hypo)

    # print out scores

    return final_scores


def evaluate_captions_cider(ref, cand):
    hypo = {}
    refe = {}

    for i, caption in enumerate(cand):
        # hypo[i] = [{'caption': caption}]
        # refe[i] = [{'caption': cap} for cap in ref[i]]
        hypo[i] = [caption]
        refe[i] = ref[i]

    # tokenizer = PTBTokenizer()
    # refe = tokenizer.tokenize(refe)
    # hypo = tokenizer.tokenize(hypo)

    scorers = [
        (Cider(), "CIDEr")
    ]
    for scorer, method in scorers:
        _, scores = scorer.compute_score(refe, hypo)
        assert len(scores) == len(cand)
    return scores


def evaluate_captions_mix(ref, cand):
    hypo = {}
    refe = {}
    for i, caption in enumerate(cand):
        # hypo[i] = [{'caption': caption}]
        # refe[i] = [{'caption': cap} for cap in ref[i]]
        caption = ''.join(caption.split(' ')[1:]).strip().replace('.', '').encode('utf-8')
        caption = jieba.cut(caption, cut_all=False)
        caption = ' '.join(caption).encode('utf-8')
        # caption = unicode(caption.encode('utf-8'), 'utf-8')
        hypo[i] = [caption]
        refe[i] = [' '.join(x.split(' ')[1:]).strip().replace('.', '').encode('utf-8') for x in ref[i]]

    # tokenizer = PTBTokenizer()
    # refe = tokenizer.tokenize(refe)
    # hypo = tokenizer.tokenize(hypo)refe
    # print refe[0]
    # pdb.set_trace()

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Cider(), "CIDEr"),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]
    final_scores = {}
    for scorer, method in scorers:
        if type(method) == list:
            score, scores = scorer.compute_score(refe, hypo, verbose=0)
        else:
            score, scores = scorer.compute_score(refe, hypo)
        if type(score) == list:
            for m, s in zip(method, scores):
                final_scores[m] = np.asarray(s)
                assert len(s) == len(cand)
        else:
            final_scores[method] = np.asarray(scores)
            assert len(scores) == len(cand)
    return final_scores['CIDEr'] + 2. * final_scores['ROUGE_L'] + 2. * final_scores['Bleu_4'] + 5. * final_scores['METEOR']


def evaluate(data_path='./data', split='val', get_scores=False):
    reference_path = os.path.join(data_path, "%s/%s.references.pkl" % (split, split))
    candidate_path = os.path.join(data_path, "%s/%s.candidate.captions.pkl" % (split, split))

    # load caption data
    with open(reference_path, 'rb') as f:
        ref = pickle.load(f)
    with open(candidate_path, 'rb') as f:
        cand = pickle.load(f)

    # make dictionary
    hypo = {}
    for i, caption in enumerate(cand):
        caption = ''.join(caption.split(' ')).strip().replace('.', '')
        caption = jieba.cut(caption.encode('utf-8'), cut_all=False)
        caption = ' '.join(caption)
        hypo[i] = [{'caption': caption}]

    # compute bleu score
    final_scores = score_all(ref, hypo)

    # print out scores
    print 'Bleu_1:\t', final_scores['Bleu_1']
    print 'Bleu_2:\t', final_scores['Bleu_2']
    print 'Bleu_3:\t', final_scores['Bleu_3']
    print 'Bleu_4:\t', final_scores['Bleu_4']
    print 'Meteor:\t', final_scores['METEOR']
    print 'ROUGE_L:', final_scores['ROUGE_L']
    print 'CIDEr:\t', final_scores['CIDEr']

    if get_scores:
        return final_scores


if __name__ == "__main__":
    ref = [[u'a tiddy bear', u'a animal'], [u'<START> a number of luggage bags on a cart in a lobby .', u'<START> a cart filled with suitcases and bags .',
                                            u'<START> trolley used for transporting personal luggage to guests rooms .', u'<START> wheeled cart with luggage at lobby of commercial business .', u'<START> a luggage cart topped with lots of luggage .']]
    dec = [u'some one', u' a man is standing next to a car with a suitcase .']
    r = [evaluate_captions_cider([k], [v]) for k, v in zip(ref, dec)]
    print r
