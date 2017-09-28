import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os
import pdb
import json
import cPickle as pickle
import random
from scipy import ndimage
from utils import *
from bleu import evaluate, evaluate_captions_mix, evaluate_for_particular_captions
from multiprocessing.dummy import Pool as ThreadPool
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

pool = ThreadPool(10)
eps = 1e-10


def names2data(features, names):
    data = np.asarray(pool.map(lambda x: features[x][:], names))
    return data


class CaptioningSolver(object):
    def __init__(self, model, data, val_data, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
            - data: Training data; dictionary with the following keys:
                - features: Feature vectors of shape (82783, 196, 512)
                - file_names: Image file names of shape (82783, )
                - captions: Captions of shape (400000, 17)
                - image_idxs: Indices for mapping caption to image of shape (400000, )
                - word_to_idx: Mapping dictionary from word to index
            - val_data: validation data; for print out BLEU scores for each epoch.
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - print_every: Integer; training losses will be printed every print_every iterations.
            - save_every: Integer; model variables will be saved every save_every epoch.
            - pretrained_model: String; pretrained model path
            - model_path: String; model path for saving
            - test_model: String; model path for test
        """

        self.model = model
        self.data = data
        self.val_data = val_data
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_bleu = kwargs.pop('print_bleu', False)
        self.print_every = kwargs.pop('print_every', 100)
        self.save_every = kwargs.pop('save_every', 1)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')
        self.n_batches = kwargs.pop('n_batches', 10000)
        self.max_len = self.model.T + 1

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def train(self):
        # train/val dataset
        n_examples = self.data['captions'].shape[0]
        n_iters_per_epoch = int(np.ceil(float(n_examples) / self.batch_size))
        n_iters_per_epoch = min(n_iters_per_epoch, self.n_batches)
        # features = self.data['features']
        captions = self.data['captions']
        image_idxs = self.data['image_idxs']
        image_names = np.array([os.path.split(path)[1] for path in self.data['file_names']])

        val_features = self.val_data['features']
        val_captions = self.val_data['captions']
        val_idxs = self.val_data['image_idxs']
        val_names = np.array([os.path.split(path)[1] for path in self.val_data['file_names']])
        val_names_idxs = val_names[val_idxs]
        n_iters_val = int(np.ceil(float(len(val_features)) / self.batch_size))

        # build graphs for training model and sampling captions
        _ = self.model.build_model()
        tf.get_variable_scope().reuse_variables()
        alphas, betas, sampled_captions, _ = self.model.build_multinomial_sampler(max_len=self.max_len)

        _, _, greedy_caption = self.model.build_sampler(max_len=self.max_len)

        rewards = tf.placeholder(tf.float32, [None])
        base_line = tf.placeholder(tf.float32, [None])

        grad_mask = tf.placeholder(tf.int32, [None, self.model.T])
        t1 = tf.expand_dims(grad_mask, 1)
        t1_mul = tf.to_float(tf.transpose(t1, [0, 2, 1]))

        # important step
        loss = self.model.build_loss()

        # train op
        with tf.name_scope('optimizer'):

            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                       global_step=global_step,
                                                       decay_steps=n_iters_per_epoch * 3,
                                                       decay_rate=0.8)

            optimizer = self.optimizer(learning_rate=learning_rate)
            norm = tf.reduce_sum(t1_mul)
            r = rewards - base_line
            # r = tf.pow(tf.abs(r), iter_count) * tf.sign(r * iter_count + eps)

            sum_loss = -  tf.reduce_sum(
                tf.transpose(tf.mul(tf.transpose(loss, [2, 1, 0]), r), [2, 1, 0])) / norm

            grads_rl, _ = tf.clip_by_global_norm(tf.gradients(sum_loss, tf.trainable_variables(
            ), aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N), 5.0)

            grads_and_vars = list(zip(grads_rl, tf.trainable_variables()))

            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        # summary op

        print "The number of epoch: %d" % self.n_epochs
        print "Data size: %d" % n_examples
        print "Batch size: %d" % self.batch_size
        print "Iterations per epoch: %d" % n_iters_per_epoch

        config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        config.gpu_options.allocator_type = 'BFC'
        with tf.Session(config=config) as sess:

            saver = tf.train.Saver()
            if self.model.pretrained_model is not None:
                saver.restore(sess, self.model.pretrained_model)
            # sess.run(tf.global_variables_initializer())

            start_t = time.time()

            with open(os.path.join(self.model_path, 'val.RandB.scores.txt'), 'w') as f:
                all_decoded_for_eval = []
                for k in range(n_iters_val):
                    captions_batch = np.array(val_captions[k * self.batch_size:(k + 1) * self.batch_size])
                    names_batch = val_names[k * self.batch_size:(k + 1) * self.batch_size]
                    features_batch = names2data(val_features, names_batch)
                    feed_dict = {self.model.features: features_batch, self.model.captions: captions_batch}

                    greedy_words = sess.run(greedy_caption,
                                            feed_dict)
                    _, greedy_decoded = decode_captions_for_blue(np.array(greedy_words), self.model.idx_to_word)
                    all_decoded_for_eval.extend(greedy_decoded)

                scores = evaluate_for_particular_captions(all_decoded_for_eval, data_path='./data', split='val',
                                                          get_scores=True)

                f.write('\n')
                f.write("before train:")
                f.write('\n')
                f.write("Bleu_1:" + str(scores['Bleu_1']))
                f.write('\n')
                f.write("Bleu_2:" + str(scores['Bleu_2']))
                f.write('\n')
                f.write("Bleu_3:" + str(scores['Bleu_3']))
                f.write('\n')
                f.write("Bleu_4:" + str(scores['Bleu_4']))
                f.write('\n')
                # f.write("SPICE:" + str(scores['SPICE']))
                # f.write('\n')
                f.write("ROUGE_L:" + str(scores['ROUGE_L']))
                f.write('\n')
                f.write("Meteor:" + str(scores['METEOR']))
                f.write('\n')
                f.write("CIDEr:" + str(scores['CIDEr']))
                f.write('\n')
                f.write("metric" + str(2. * scores['Bleu_4'] + scores['CIDEr'] +
                                       2. * scores['ROUGE_L'] + 5. * scores['METEOR']))
                f.write('\n')

            for e in range(self.n_epochs):
                rand_idxs = np.random.permutation(n_examples)
                captions = np.array(captions[rand_idxs])
                image_idxs = np.array(image_idxs[rand_idxs])
                image_names = np.asarray(image_names[image_idxs])
                assert captions.shape[0] == image_idxs.shape[0]

                # read training split features
                n_iters_list = [0]
                image_idxs_list_tmp = []
                captions_list_tmp = []
                for s in range(7):
                    image_idxs_split = image_idxs[np.where(np.logical_and(
                        image_idxs >= s * 30000, image_idxs < (s + 1) * 30000))]
                    captions_split = captions[np.where(np.logical_and(
                        image_idxs >= s * 30000, image_idxs < (s + 1) * 30000))]

                    image_idxs_list_tmp.append(image_idxs_split)
                    captions_list_tmp.append(captions_split)
                # random shuffle
                inds_list = range(7)
                random.shuffle(inds_list)
                image_idxs_list = [image_idxs_list_tmp[x] for x in inds_list]
                captions_list = [captions_list_tmp[x] for x in inds_list]
                print('random split indices : {}'.format(inds_list))

                n_iters_list = [0] + [len(x) // self.batch_size for x in image_idxs_list]
                n_iters_list = np.cumsum(np.asarray(n_iters_list))[:-1]
                assert len(n_iters_list) == 7
                print('iter_list is {}').format(n_iters_list)

                b_for_eval = []
                greedy_decoded_for_eval = []
                for i in range(n_iters_per_epoch):
                    if i in n_iters_list:
                        tmp1 = time.time()
                        split = inds_list[np.where(n_iters_list == i)[0][0]]
                        s_num = np.where(n_iters_list == i)[0][0]
                        features_split = None
                        print('read split {} features'.format(split))
                        with h5py.File('./data/train/train_split{}.features.h5'.format(split), 'r') as f_tmp:
                            features_split = f_tmp['features'][:]
                        captions_split = captions_list[s_num]
                        image_idxs_split = image_idxs_list[s_num]
                        assert np.all(image_idxs_split >= 0)
                        tmp2 = time.time()
                        print('read features data from h5 file: {} s'.format(tmp2 - tmp1))
                    tmp_i = i - n_iters_list[s_num]
                    assert tmp_i >= 0.

                    captions_batch = np.array(captions_split[tmp_i * self.batch_size:(tmp_i + 1) * self.batch_size])
                    image_idxs_batch = np.array(image_idxs_split[tmp_i * self.batch_size:(tmp_i + 1) * self.batch_size])

                    features_batch = features_split[image_idxs_batch - split * 30000]
                    assert features_batch.shape[2] == 1536, 'dimension of visual features should be (11*11, 1536)'

                    tmp1 = time.time()
                    # ground_truths = []
                    # for j in range(len(image_idxs_batch)):
                    #     ground_truth = captions[image_idxs == image_idxs_batch[j]]
                    #     assert len(ground_truth) > 0, 'zero-length captions for image {}'.format(image_names_batch[j])
                    #     ground_truths.append(ground_truth)
                    ground_truths = [captions_split[image_idxs_split == image_idxs_batch[j]] for j in
                                     range(len(image_idxs_batch))]

                    ref_decoded = [decode_captions(ground_truths[j], self.model.idx_to_word)
                                   for j in range(len(ground_truths))]

                    feed_dict = {self.model.features: features_batch, self.model.captions: captions_batch}

                    # fetch captions as cache
                    samples, greedy_words = sess.run([sampled_captions, greedy_caption],
                                                     feed_dict)
                    mask, all_decoded = decode_captions_for_blue(samples, self.model.idx_to_word)
                    _, greedy_decoded = decode_captions_for_blue(greedy_words, self.model.idx_to_word)
                    greedy_decoded_for_eval.extend(greedy_decoded)

                    # r = [evaluate_captions_bleu([k], [v]) for k, v in zip(ref_decoded, all_decoded)]
                    # b = [evaluate_captions_bleu([k], [v]) for k, v in zip(ref_decoded, greedy_decoded)]
                    r = evaluate_captions_mix(ref_decoded, all_decoded)
                    b = evaluate_captions_mix(ref_decoded, greedy_decoded)

                    # pdb.set_trace()
                    b_for_eval.extend(b)

                    feed_dict = {grad_mask: mask, self.model.sample_caption: samples, rewards: r, base_line: b,
                                 self.model.features: features_batch, self.model.captions: captions_batch,
                                 global_step: e * n_iters_per_epoch + i
                                 }  # write summary for tensorboard visualization
                    _ = sess.run([train_op], feed_dict)
                    if i % 100 == 0:
                        tmp2 = time.time()
                        print('record summary: {} s'.format(tmp2 - tmp1))

                    if i % self.print_every == 0 and i > 0:
                        print('Epoch {}: has runned {} iterations !'.format(e + 1, i))
                        print('Reward {} and baseline {}'.format(r[::4], b[::4]))
                        ground_truths = captions_split[image_idxs_split == image_idxs_batch[0]]
                        decoded = decode_captions(ground_truths, self.model.idx_to_word)
                        for j, gt in enumerate(decoded):
                            print "Ground truth %d: %s" % (j + 1, gt)
                        feed_dict = {self.model.features: features_batch, self.model.captions: captions_batch}
                        gen_caps = sess.run(greedy_caption, feed_dict)
                        decoded = decode_captions(gen_caps, self.model.idx_to_word)
                        print "Generated caption: %s\n" % decoded[0]

                # print out BLEU scores and file write
                if self.print_bleu:

                    with open(os.path.join(self.model_path, 'val.RandB.scores.txt'), 'a') as f:
                        all_decoded_for_eval = []
                        for k in range(n_iters_val):
                            names_batch = val_names[k * self.batch_size:(k + 1) * self.batch_size]
                            features_batch = names2data(val_features, names_batch)
                            feed_dict = {self.model.features: features_batch}

                            greedy_words = sess.run(greedy_caption, feed_dict)
                            _, greedy_decoded = decode_captions_for_blue(np.array(greedy_words), self.model.idx_to_word)
                            all_decoded_for_eval.extend(greedy_decoded)

                        scores = evaluate_for_particular_captions(
                            all_decoded_for_eval, data_path='./data', split='val', get_scores=True)

                        s = [i > j for i, j in zip(r, b)]
                        f.write('Epoch %d\n' % (e + 1))
                        f.write("b:" + str(np.mean(np.array(b_for_eval))))
                        f.write('\n')
                        f.write("count true:" + str(s.count(True)))
                        f.write('\n')
                        f.write(str(ref_decoded[0][0]))
                        f.write('\n')
                        f.write(str(all_decoded[0]))
                        f.write('\n')
                        f.write(str(greedy_decoded[0]))
                        f.write('\n')
                        f.write("Bleu_1:" + str(scores['Bleu_1']))
                        f.write('\n')
                        f.write("Bleu_2:" + str(scores['Bleu_2']))
                        f.write('\n')
                        f.write("Bleu_3:" + str(scores['Bleu_3']))
                        f.write('\n')
                        f.write("Bleu_4:" + str(scores['Bleu_4']))
                        f.write('\n')
                        # f.write("SPICE:" + str(scores['SPICE']))
                        # f.write('\n')
                        f.write("ROUGE_L:" + str(scores['ROUGE_L']))
                        f.write('\n')
                        f.write("Meteor:" + str(scores['METEOR']))
                        f.write('\n')
                        f.write("CIDEr:" + str(scores['CIDEr']))
                        f.write('\n')
                        f.write("metric" + str(2. * scores['Bleu_4'] + scores['CIDEr'] +
                                               2. * scores['ROUGE_L'] + 5. * scores['METEOR']))
                        f.write('\n')
                        if (e + 1) % self.save_every == 0:
                            saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e + 1)
                            print "model-%s saved." % (e + 1)

    def test(self, data, split='train', attention_visualization=False, save_sampled_captions=True):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - captions: Captions of shape (24210, 17)
            - image_idxs: Indices for mapping caption to image of shape (24210, )
            - features_to_captions: Mapping feature to captions (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
        '''

        features = data['features']
        names = data['names']

        # n_examples = self.data['captions'].shape[0]
        # n_iters_per_epoch = int(np.ceil(float(n_examples) / self.batch_size))
        # build a graph to sample captions
        alphas, betas, sampled_captions = self.model.build_sampler(max_len=self.max_len)  # (N, max_len, L), (N, max_len)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            features_batch, image_files = sample_coco_minibatch_inference(data, self.batch_size)
            feed_dict = {self.model.features: features_batch}
            alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
            decoded = decode_captions(sam_cap, self.model.idx_to_word)

            # if attention_visualization:
            #     for n in range(10):
            #         print "Sampled Caption: %s" % decoded[n]

            #         # Plot original image
            #         img = ndimage.imread(image_files[n])
            #         plt.clf()
            #         plt.subplot(4, 5, 1)
            #         plt.imshow(img)
            #         plt.axis('off')

            #         # Plot images with attention weights
            #         words = decoded[n].split(" ")
            #         for t in range(len(words)):
            #             if t > 18:
            #                 break
            #             plt.subplot(4, 5, t + 2)
            #             plt.text(0, 1, '%s(%.2f)' % (words[t], bts[n, t]), color='black', backgroundcolor='white',
            #                      fontsize=8)
            #             plt.imshow(img)
            #             alp_curr = alps[n, t, :].reshape(14, 14)
            #             alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
            #             plt.imshow(alp_img, alpha=0.85)
            #             plt.axis('off')
            #         plt.savefig(str(n) + 'test.pdf')

            if save_sampled_captions:
                all_sam_cap = np.ndarray((len(features), self.max_len))
                num_iter = int(np.ceil(float(len(features)) / self.batch_size))
                for i in range(num_iter):
                    names_batch = names[i * self.batch_size:(i + 1) * self.batch_size]
                    features_batch = names2data(features, names_batch)
                    feed_dict = {self.model.features: features_batch}
                    all_sam_cap[i * self.batch_size:(i + 1) * self.batch_size] = sess.run(sampled_captions, feed_dict)
                all_decoded = decode_captions(all_sam_cap, self.model.idx_to_word)
                save_pickle(all_decoded, "./data/%s/%s.candidate.captions.pkl" % (split, split))
                if split == 'val':
                    scores = evaluate(data_path='./data', split='val', get_scores=True)

    def save_result(self, data, split='test', batch_size=1, save_path='./result.json'):

        def decode_captions_save(captions, idx_to_word):
            if captions.ndim == 1:
                T = captions.shape[0]
                N = 1
            else:
                N, T = captions.shape
            decoded = []
            for i in range(N):
                words = []
                for t in range(T):
                    if captions.ndim == 1:
                        word = idx_to_word[captions[t]]
                    else:
                        word = idx_to_word[captions[i, t]]
                    if word == '<END>':
                        break
                    if word != '<NULL>':
                        words.append(word)
                decoded.append(''.join(words))
            return decoded

        if batch_size is None:
            batch_size = self.batch_size
        features = data['features']
        file_names = np.array([os.path.split(path)[1] for path in data['file_names']])
        for _, fname in enumerate(file_names):
            assert fname.lower().endswith('.jpg')

        n_examples = len(file_names)
        # build a graph to sample captions
        alphas, betas, sampled_captions = self.model.build_sampler(max_len=self.max_len)

        config = tf.ConfigProto(allow_soft_placement=True)
        n_iter_test = int(np.ceil(float(n_examples) / batch_size))
        config.gpu_options.allow_growth = True
        output_list = []
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            for k in xrange(n_iter_test):
                filename_batch = file_names[batch_size * k: batch_size * (k + 1)]
                features_batch = names2data(features, filename_batch)
                feed_dict = {self.model.features: features_batch}
                if k % 100 == 0:
                    print('processing {} batches'.format(k))

                alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
                decoded = decode_captions_save(sam_cap, self.model.idx_to_word)
                for filename, caption in zip(filename_batch, decoded):
                    output_dict = {}
                    image_id = filename[:-4]
                    output_dict['image_id'] = image_id
                    output_dict['caption'] = caption.encode('utf-8')
                    output_list.append(output_dict)
            with open(save_path, 'w') as f:
                json.dump(output_list, f, ensure_ascii=False, indent=4, separators=(',', ':'))

    def save_beamsearch_result(self, data, split='test', batch_size=1, save_path='./result.json'):

        def decode_captions_save(captions, idx_to_word):
            if captions.ndim == 1:
                T = captions.shape[0]
                N = 1
            else:
                N, T = captions.shape
            decoded = []
            for i in range(N):
                words = []
                for t in range(T):
                    if captions.ndim == 1:
                        word = idx_to_word[captions[t]]
                    else:
                        word = idx_to_word[captions[i, t]]
                    if word == '<END>':
                        break
                    if word != '<NULL>':
                        words.append(word)
                decoded.append(''.join(words))
            return decoded

        if batch_size is None:
            batch_size = self.batch_size
        features = data['features']
        file_names = np.array([os.path.split(path)[1] for path in self.data['file_names']])
        n_examples = len(file_names)
        # build a graph to sample captions
        config = tf.ConfigProto(allow_soft_placement=True)
        n_iter_test = int(np.ceil(float(n_examples) / batch_size))
        config.gpu_options.allow_growth = True
        output_list = []
        bk = 2
        var_list = self.model.build_beamsearch()
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            for k in xrange(n_iter_test):
                filename_batch = file_names[batch_size * k: batch_size * (k + 1)]
                features_batch = names2data(features, filename_batch).astype(np.float32)

                if k % 100 == 0:
                    print('processing {} batches, bt is {}'.format(k, bk))

                sam_cap = self.model.beamsearch_sampler(
                    var_list, features_batch, sess, max_len=self.max_len, k=bk)  # (N, max_len, L), (N, max_len)
                decoded = decode_captions_save(sam_cap, self.model.idx_to_word)
                # print '%s' % decoded[0]
                for filename, caption in zip(filename_batch, decoded):
                    output_dict = {}
                    image_id = filename[:-4]
                    output_dict['image_id'] = image_id
                    output_dict['caption'] = caption.encode('utf-8')
                    output_list.append(output_dict)
            with open(save_path, 'w') as f:
                json.dump(output_list, f, ensure_ascii=False, indent=4, separators=(',', ':'))
