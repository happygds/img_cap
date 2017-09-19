import cPickle as pickle
import os
import sys
sys.path.append('../coco-caption')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.ciderD.ciderD import CiderD
from pycocoevalcap.cider.cider import Cider
import pdb


def score_all(ref, hypo):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (CiderD(), "CIDEr-D"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
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
        hypo[i] = [caption]
        refe[i] = ref[i]
    # compute bleu score
    final_scores = score_all(refe, hypo)

    # print out scores

    return final_scores


def evaluate_captions_ciderD(ref, cand):
    hypo = {}
    refe = {}
    # print type(cand), len(cand), type(ref), len(ref)
    # pdb.set_trace()

    for i, caption in enumerate(cand):
        hypo[i] = [caption]
        refe[i] = ref[i]
    scorers = [
        (CiderD(), "CIDEr-D")
    ]
    for scorer, method in scorers:
        _, scores = scorer.compute_score(refe, hypo)
        assert len(scores) == len(cand)
    return scores


def evaluate_captions_bleu(ref, cand):
    hypo = {}
    refe = {}
    for i, caption in enumerate(cand):
        hypo[i] = [caption]
        refe[i] = ref[i]
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])
    ]
    final_scores = {}
    for scorer, method in scorers:
        _, scores = scorer.compute_score(refe, hypo)
        for m, s in zip(method, scores):
            final_scores[m] = s
            assert len(s) == len(cand)
    return final_scores['Bleu_4']


def evaluate_captions_mixD(ref, cand):
    hypo = {}
    refe = {}
    for i, caption in enumerate(cand):
        hypo[i] = [caption]
        refe[i] = ref[i]
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (CiderD(), "CIDEr-D"),
        (Rouge(), "ROUGE_L")
    ]
    final_scores = {}
    for scorer, method in scorers:
        _, scores = scorer.compute_score(refe, hypo)
        if type(scores) == list:
            for m, s in zip(method, scores):
                final_scores[m] = s
                assert len(s) == len(cand)
        else:
            final_scores[method] = scores
            assert len(scores) == len(cand)
    return final_scores['CIDEr-D'] + 2. * final_scores['ROUGE_L'] + final_scores['Bleu_4'] + 0.5 * final_scores['Bleu_3']


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
        hypo[i] = [caption]

    # compute bleu score
    final_scores = score_all(ref, hypo)

    # print out scores
    print 'Bleu_1:\t', final_scores['Bleu_1']
    print 'Bleu_2:\t', final_scores['Bleu_2']
    print 'Bleu_3:\t', final_scores['Bleu_3']
    print 'Bleu_4:\t', final_scores['Bleu_4']
    print 'CIDEr-D:\t', final_scores['CIDEr-D']
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
