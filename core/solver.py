import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os
import cPickle as pickle
import random
from scipy import ndimage
from utils import *
from bleu import evaluate
from multiprocessing.dummy import Pool as ThreadPool
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def names2data(features, names):
    pool = ThreadPool(8)
    # sort_names, sort_inds = np.unique(names, return_inverse=True)
    # if len(sort_names) < len(names):
    #     data = np.asarray(pool.map(lambda x: features[x][:], sort_names))
    #     data = np.take(data, sort_inds, axis=0)
    # else:
    #     data = np.asarray(pool.map(lambda x: features[x][:], names))
    data = np.asarray(pool.map(lambda x: features[x][:], names))
    pool.close()
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
        val_names = np.array([os.path.split(path)[1] for path in self.val_data['file_names']])
        n_iters_val = int(np.ceil(float(len(val_features)) / self.batch_size))

        # build graphs for training model and sampling captions
        loss = self.model.build_model()
        tf.get_variable_scope().reuse_variables()
        _, _, generated_captions = self.model.build_sampler(max_len=self.max_len)

        # train op
        with tf.name_scope('optimizer'):
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                       global_step=global_step,
                                                       decay_steps=n_iters_per_epoch * 3,
                                                       decay_rate=0.9)

            optimizer = self.optimizer(learning_rate=learning_rate)
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        # summary op
        tf.summary.scalar('batch_loss', loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        for grad, var in grads_and_vars:
            tf.summary.histogram(var.op.name + '/gradient', grad)

        summary_op = tf.merge_all_summaries()

        print "The number of epoch: %d" % self.n_epochs
        print "Data size: %d" % n_examples
        print "Batch size: %d" % self.batch_size
        print "Iterations per epoch: %d" % n_iters_per_epoch

        config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=40)

            if self.pretrained_model is not None:
                print "Start training with pretrained Model.."
                saver.restore(sess, self.pretrained_model)

            prev_loss = -1
            curr_loss = 0
            start_t = time.time()

            for e in range(self.n_epochs):
                rand_idxs = np.random.permutation(n_examples)
                captions = captions[rand_idxs]
                image_idxs = image_idxs[rand_idxs]
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

                    feed_dict = {self.model.features: features_batch, self.model.captions: captions_batch,
                                 global_step: e * n_iters_per_epoch + i
                                 }
                    _, l = sess.run([train_op, loss], feed_dict)
                    curr_loss += l

                    # write summary for tensorboard visualization
                    if i % 100 == 0 and i > 0:
                        summary = sess.run(summary_op, feed_dict)
                        summary_writer.add_summary(summary, e * n_iters_per_epoch + i)

                    if (i + 1) % self.print_every == 0:
                        print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" % (e + 1, i + 1, l)
                        ground_truths = captions_split[image_idxs_split == image_idxs_batch[0]]
                        decoded = decode_captions(ground_truths, self.model.idx_to_word)
                        for j, gt in enumerate(decoded):
                            print "Ground truth %d: %s" % (j + 1, gt)
                        gen_caps = sess.run(generated_captions, feed_dict)
                        decoded = decode_captions(gen_caps, self.model.idx_to_word)
                        print "Generated caption: %s\n" % decoded[0]

                print "Previous epoch loss: ", prev_loss / n_examples
                print "Current epoch loss: ", curr_loss / n_examples
                print "Elapsed time: ", time.time() - start_t
                prev_loss = curr_loss
                curr_loss = 0

                # print out BLEU scores and file write
                if self.print_bleu:
                    all_gen_cap = np.ndarray((len(val_features), self.max_len))
                    for i in range(n_iters_val):
                        names_batch = val_names[i * self.batch_size:(i + 1) * self.batch_size]
                        features_batch = names2data(val_features, names_batch)

                        feed_dict = {self.model.features: features_batch}
                        gen_cap = sess.run(generated_captions, feed_dict=feed_dict)
                        all_gen_cap[i * self.batch_size:(i + 1) * self.batch_size] = gen_cap

                    all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
                    save_pickle(all_decoded, "./data/val/val.candidate.captions.pkl")
                    scores = evaluate(data_path='./data', split='val', get_scores=True)
                    write_bleu(scores=scores, path=self.model_path, epoch=e)

                # save model's parameters
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
        idxs = data['image_idxs']
        n_examples = self.data['captions'].shape[0]
        n_iters_per_epoch = int(np.ceil(float(n_examples) / self.batch_size))
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

            if self.print_bleu:
                all_gen_cap = np.ndarray((len(features), self.max_len))
                for i in range(n_iters_per_epoch):
                    features_batch = features[i * self.batch_size:(i + 1) * self.batch_size]
                    feed_dict = {self.model.features: features_batch}
                    gen_cap = sess.run(sampled_captions, feed_dict=feed_dict)
                    all_gen_cap[i * self.batch_size:(i + 1) * self.batch_size] = gen_cap

                all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
                save_pickle(all_decoded, "./data/val/val.candidate.captions.pkl")
                scores = evaluate(data_path='./data', split='val', get_scores=True)

            if attention_visualization:
                for n in range(10):
                    print "Sampled Caption: %s" % decoded[n]

                    # Plot original image
                    img = ndimage.imread(image_files[n])
                    plt.clf()
                    plt.subplot(4, 5, 1)
                    plt.imshow(img)
                    plt.axis('off')

                    # Plot images with attention weights
                    words = decoded[n].split(" ")
                    for t in range(len(words)):
                        if t > 18:
                            break
                        plt.subplot(4, 5, t + 2)
                        plt.text(0, 1, '%s(%.2f)' % (words[t], bts[n, t]), color='black', backgroundcolor='white',
                                 fontsize=8)
                        plt.imshow(img)
                        alp_curr = alps[n, t, :].reshape(14, 14)
                        alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
                        plt.imshow(alp_img, alpha=0.85)
                        plt.axis('off')
                    plt.savefig(str(n) + 'test.pdf')

            if save_sampled_captions:
                all_sam_cap = np.ndarray((len(features), self.max_len))
                num_iter = int(np.ceil(float(len(features)) / self.batch_size))
                for i in range(num_iter):
                    idxs_batch = idxs[k * self.batch_size:(k + 1) * self.batch_size]
                    names_batch = names[idxs_batch]
                    features_batch = names2data(features, names_batch)
                    feed_dict = {self.model.features: features_batch}
                    all_sam_cap[i * self.batch_size:(i + 1) * self.batch_size] = sess.run(sampled_captions, feed_dict)
                all_decoded = decode_captions(all_sam_cap, self.model.idx_to_word)
                save_pickle(all_decoded, "./data/%s/%s.candidate.captions.pkl" % (split, split))

    def inference(self, data, split='train', attention_visualization=True, save_sampled_captions=True):
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

        # build a graph to sample captions
        alphas, betas, sampled_captions = self.model.build_sampler(max_len=self.max_len)  # (N, max_len, L), (N, max_len)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, './model/lstm/model-20')
            features_batch, image_files = sample_coco_minibatch_inference(data, self.batch_size)
            feed_dict = {self.model.features: features_batch}
            alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
            decoded = decode_captions(sam_cap, self.model.idx_to_word)
            print "end"
            print decoded
