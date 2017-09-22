# =========================================================================================
# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".
# There are some notations.
# N is batch size.
# L is spacial size of feature vector (196).
# D is dimension of image feature vector (512).
# T is the number of time step which is equal to caption's length-1 (16).
# V is vocabulary size (about 10000).
# M is dimension of word vector which is embedding size (default is 512).
# H is dimension of hidden state (default is 1024).
# =========================================================================================

from __future__ import division

import tensorflow as tf
import numpy as np
import copy


class CaptionGenerator(object):
    def __init__(self, word_to_idx, dim_feature=[196, 512], dim_embed=512, dim_hidden=1024, n_time_step=16,
                 prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, dropout=True):
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_feature: (optional) Dimension of vggnet19 conv5_3 feature vectors.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM.
            prev2out: (optional) previously generated word to hidden state. (see Eq (2) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (2) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.alpha_c = alpha_c
        self.selector = selector
        self.dropout = dropout
        self.V = len(word_to_idx)
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and captions
        self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])

        # sentence concat <start>
        self.sample_caption = tf.placeholder(tf.int32, [None, self.T])

    def _get_initial_lstm(self, features):
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h

    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj

    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)  # (N, L, D)
            # h_att = tf.nn.tanh(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)  # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])  # (N, L)
            alpha = tf.nn.softmax(out_att)
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')  # (N, D)
            return context, alpha

    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')  # (N, 1)
            context = tf.mul(beta, context, name='selected_context')
            return context, beta

    def _decode_lstm(self, x, h, context, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)
            h_logits = tf.matmul(h, w_h) + b_h

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)

            if self.prev2out:
                h_logits += x
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits

    def _batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode == 'train'),
                                            updates_collections=None,
                                            scope=(name + 'batch_norm'))

    def build_model(self):
        features = self.features
        captions = self.captions
        batch_size = tf.shape(features)[0]

        captions_in = captions[:, :self.T]
        captions_out = captions[:, 1:]
        mask = tf.to_float(tf.not_equal(captions_out, self._null))

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='train', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        x = self._word_embedding(inputs=captions_in)
        features_proj = self._project_features(features=features)

        loss = 0.0
        alpha_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(self.T):
            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t != 0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t != 0))

            with tf.variable_scope('lstm', reuse=(t != 0)):
                _, (c, h) = lstm_cell(inputs=tf.concat(1, [x[:, t, :], context]), state=[c, h])

            logits = self._decode_lstm(x[:, t, :], h, context, dropout=self.dropout, reuse=(t != 0))
            loss += tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits, captions_out[:, t]) * mask[:, t])

            # # focal loss
            # focal_loss = - (1. - logits) ** 2 * captions_out[:, t] * tf.log(
            #     tf.clip_by_value(logits, 1e-20, 1.0)) - logits ** 2 * (
            #     1. - captions_out[:, t]) * tf.log(tf.clip_by_value(1. - logits, 1e-20, 1.0))
            # loss += tf.reduce_sum(focal_loss * mask[:, t])

        if self.alpha_c > 0:
            alphas = tf.transpose(tf.pack(alpha_list), (1, 0, 2))  # (N, T, L)
            alphas_all = tf.reduce_sum(alphas, 1)  # (N, L)
            alpha_reg = self.alpha_c * tf.reduce_sum((float(self.T) / float(self.L) - alphas_all) ** 2)
            loss += alpha_reg

        return loss / tf.to_float(batch_size)

    def build_sampler(self, max_len=20):
        features = self.features

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        features_proj = self._project_features(features=features)

        sampled_word_list = []
        alpha_list = []
        beta_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        sampled_word = None

        for t in range(max_len):
            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=True)

            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t != 0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t != 0))
                beta_list.append(beta)

            with tf.variable_scope('lstm', reuse=(t != 0)):
                _, (c, h) = lstm_cell(inputs=tf.concat(1, [x, context]), state=[c, h])

            logits = self._decode_lstm(x, h, context, reuse=(t != 0))
            sampled_word = tf.argmax(logits, 1)
            sampled_word_list.append(sampled_word)

        alphas = tf.transpose(tf.pack(alpha_list), (1, 0, 2))  # (N, T, L)
        betas = tf.transpose(tf.squeeze(beta_list), (1, 0))  # (N, T)
        sampled_captions = tf.transpose(tf.pack(sampled_word_list), (1, 0))  # (N, max_len)
        return alphas, betas, sampled_captions

    def build_loss(self):
        features = self.features
        captions = self.sample_caption

        mask = tf.to_float(tf.not_equal(captions, self._null))

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        x = self._word_embedding(inputs=captions)
        features_proj = self._project_features(features=features)

        loss = []
        alpha_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(self.T):
            if t == 0:
                word = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
            else:
                word = x[:, t - 1, :]
            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t != 0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t != 0))

            with tf.variable_scope('lstm', reuse=(t != 0)):
                _, (c, h) = lstm_cell(inputs=tf.concat(1, [word, context]), state=[c, h])

            logits = self._decode_lstm(word, h, context, reuse=(t != 0))
            softmax = tf.nn.softmax(logits, dim=-1, name=None)

            loss.append(tf.transpose(tf.mul(tf.transpose(tf.log(
                tf.clip_by_value(softmax, 1e-20, 1.0)) * tf.one_hot(
                captions[:, t], self.V), [1, 0]), mask[:, t]), [1, 0]))

        loss_out = tf.transpose(tf.pack(loss), (1, 0, 2))  # (N, T, max_len)

        return loss_out

    def build_multinomial_sampler(self, max_len=16):
        features = self.features

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        features_proj = self._project_features(features=features)

        sampled_word_list = []
        alpha_list = []
        beta_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        loss = []
        sampled_word = None
        for t in range(self.T):
            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=True)

            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t != 0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t != 0))
                beta_list.append(beta)

            with tf.variable_scope('lstm', reuse=(t != 0)):
                _, (c, h) = lstm_cell(inputs=tf.concat(1, [x, context]), state=[c, h])

            logits = self._decode_lstm(x, h, context, reuse=(t != 0))
            softmax = tf.nn.softmax(logits, dim=-1, name=None)
            sampled_word = tf.multinomial(tf.log(tf.clip_by_value(softmax, 1e-20, 1.0)), 1)

            sampled_word = tf.reshape(sampled_word, [-1])
            loss.append(tf.log(tf.clip_by_value(softmax, 1e-20, 1.0)) * tf.one_hot(tf.identity(sampled_word), self.V))

            sampled_word_list.append(sampled_word)

        alphas = tf.transpose(tf.pack(alpha_list), (1, 0, 2))  # (N, T, L)
        betas = tf.transpose(tf.squeeze(beta_list), (1, 0))  # (N, T)
        loss_out = tf.transpose(tf.pack(loss), (1, 0, 2))  # (N, T, max_len)
        sampled_captions = tf.transpose(tf.pack(sampled_word_list), (1, 0))  # (N, max_len)
        return alphas, betas, sampled_captions, loss_out

    def build_beamsearch_sampler(self, max_len=20, k=5):
        features = self.features

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        features_proj = self._project_features(features=features)

        alpha_list = []
        beta_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        sampled_word = None
        hyp_scores = tf.zeros(1)
        hyp_samples = []

        for t in range(max_len):
            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=True)

            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t != 0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t != 0))
                beta_list.append(beta)

            with tf.variable_scope('lstm', reuse=(t != 0)):
                _, (c, h) = lstm_cell(inputs=tf.concat(1, [x, context]), state=[c, h])

            logits = self._decode_lstm(x, h, context, reuse=(t != 0))
            softmax = tf.nn.softmax(logits, dim=-1, name=None)
            cand_scores = hyp_scores[:, None] - tf.log(softmax)
            cand_flat = tf.reshape(cand_scores, (tf.shape(cand_scores)[0] * tf.shape(cand_scores)[1], ))
            voc_size = tf.shape(softmax)[1]
            costs, ranks_flat = tf.nn.top_k(cand_flat, k)
            trans_indices = tf.div(ranks_flat, voc_size)
            word_indices = tf.mod(ranks_flat, voc_size)
            new_hyp_samples = []
            new_hyp_scores = []
            new_hyp_c = []
            new_hyp_h = []
            print 't------', t
            for idx in range(k):
                ti = trans_indices[idx]
                wi = word_indices[idx]
                if t == 0:
                    new_hyp_samples.append(wi)
                elif t == 1:
                    new_hyp_samples.append(tf.concat([hyp_samples[ti], tf.expand_dims(wi, axis=-1)], 0))
                else:
                    new_hyp_samples.append(tf.concat([hyp_samples[ti], tf.expand_dims(wi, axis=-1)], 0))
                new_hyp_scores.append(costs[ti])
                new_hyp_c.append(c[ti])
                new_hyp_h.append(h[ti])

            c = tf.stack(new_hyp_c)
            h = tf.stack(new_hyp_h)
            hyp_scores = tf.stack(new_hyp_scores)
            hyp_samples = tf.stack(new_hyp_samples)
            if t == 0:
                sampled_word = hyp_samples[:]
            else:
                sampled_word = hyp_samples[:, -1]

        alphas = tf.transpose(tf.pack(alpha_list), (1, 0, 2))  # (N, T, L)
        betas = tf.expand_dims(tf.squeeze(beta_list), axis=0)  # (N, T)
        sampled_captions = tf.expand_dims(hyp_samples[tf.argmax(hyp_scores)], axis=0)  # (N, max_len)
        return alphas, betas, sampled_captions

    def build_beamsearch(self):
        features_ori = self.features
        # caluculate initial c, h np_value
        init_c_s, init_h_s = self._get_initial_lstm(features=features_ori)
        features = self._batch_norm(features_ori, mode='test', name='conv_features')
        init_c = tf.placeholder(tf.float32, (None, self.H))
        init_h = tf.placeholder(tf.float32, (None, self.H))
        # build one-step LSTM forward
        features_proj = self._project_features(features=features)
        sampled_word = tf.placeholder(tf.int32, (None,))
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        x = self._word_embedding(inputs=sampled_word)
        context, alpha = self._attention_layer(features, features_proj, init_h)
        if self.selector:
            context, beta = self._selector(context, init_h)

        with tf.variable_scope('lstm'):
            _, (c, h) = lstm_cell(inputs=tf.concat(1, [x, context]), state=[init_c, init_h])

        logits = self._decode_lstm(x, h, context)
        softmax = tf.nn.softmax(logits, dim=-1, name=None)
        return [features_ori, init_c_s, init_h_s, init_c, init_h, sampled_word, c, h, softmax]

    def beamsearch_sampler(self, var_list, init_feat, sess, max_len=20, k=5):

        # batch normalize feature vectors
        features, init_c_s, init_h_s, init_c, init_h, sampled_word, c, h, softmax = var_list

        next_c, next_h = sess.run([init_c_s, init_h_s], feed_dict={features: init_feat})
        dead_k = 0
        live_k = 1
        hyp_samples = [[]] * live_k
        hyp_scores = np.zeros(live_k).astype('float32')
        hyp_c = []
        hyp_h = []
        sample = []
        sample_score = []
        next_feat = init_feat
        for t in range(max_len):
            if t == 0:
                sampled_word_np = np.array([1])
                sampled_word_np.fill(self._start)
            next_p, next_c, next_h = sess.run([softmax, c, h],
                                              feed_dict={init_c: next_c, init_h: next_h,
                                              features: next_feat, sampled_word: sampled_word_np})
            cand_scores = hyp_scores[:, None] - np.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[: (k - dead_k)]
            voc_size = next_p.shape[1]
            trans_indices = np.divide(ranks_flat, voc_size)  # index of row
            word_indices = np.mod(ranks_flat, voc_size)  # index of col
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = np.zeros(k - dead_k).astype('float32')
            new_hyp_c = []
            new_hyp_h = []
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti] + [wi])
                new_hyp_scores[idx] = copy.copy(costs[ti])
                new_hyp_c.append(copy.copy(next_c[ti]))
                new_hyp_h.append(copy.copy(next_h[ti]))
                # new_hyp_memory_cells.append(copy.copy(next_memory_cell[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_c = []
            hyp_h = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(np.asarray(new_hyp_samples[idx], dtype=np.int32))
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_c.append(new_hyp_c[idx])
                    hyp_h.append(new_hyp_h[idx])
                    # hyp_memory_cells.append(new_hyp_memory_cells[idx])
            hyp_scores = np.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            sampled_word_np = np.array([w[-1] for w in hyp_samples])
            next_c = np.asarray(hyp_c)
            next_h = np.array(hyp_h)
            # next_memory_cell =np.asarray(hyp_memory_cells)
            next_feat = np.tile(init_feat, (live_k, 1, 1))
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(np.asarray(hyp_samples[idx], dtype=np.int32))
                sample_score.append(hyp_scores[idx])
        index = np.argmax(sample_score)
        # print 'index', index
        # print 'index', type(index)
        # print len(sample)
        # print sample[0].shape
        max_sample = sample[index][np.newaxis, :]
        return max_sample