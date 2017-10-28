# This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils

from .CaptionModel import CaptionModel

class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.ss_prob = 0.0 # Schedule sampling probability

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()),
                Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()))

    def forward(self, fc_feats, att_feats, seq):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        outputs = []

        if type(self) == C2FTopDownModel:
            outputs_final = []
            outputs_coarse = []

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        _att_feats = self.att_embed(att_feats.view(-1, self.att_feat_size))
        att_feats = _att_feats.view(*(att_feats.size()[:-1] + (self.rnn_size,)))

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats.view(-1, self.rnn_size))
        p_att_feats = p_att_feats.view(*(att_feats.size()[:-1] + (self.att_hid_size,)))
        if type(self) == C2FTopDownModel:
            p_att_feats_final = self.ctx2att_final(att_feats.view(-1, self.rnn_size))
            p_att_feats_final = p_att_feats_final.view(*(att_feats.size()[:-1] + (self.att_hid_size,)))

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                # if type(self) == C2FTopDownModel:
                #     sample_prob_fine = fc_feats.data.new(batch_size).uniform_(0, 1)
                #     sample_mask_fine = sample_prob_fine < self.ss_prob

                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                    it_fine = seq[:, i].clone()
                    it_coarse = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    if type(self) == C2FTopDownModel:
                        # sample_ind_fine = sample_mask_fine.nonzero().view(-1)
                        it_fine = seq[:, i].data.clone()
                        it_coarse = seq[:, i].data.clone()

                    prob_prev = torch.exp(outputs_final[-1].data)  # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(
                        prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
                    if type(self) == C2FTopDownModel:
                        prob_prev_fine = torch.exp(outputs[-1].data)
                        it_fine.index_copy_(0, sample_ind, torch.multinomial(
                            prob_prev_fine, 1).view(-1).index_select(0, sample_ind))
                        it_fine = Variable(it_fine, requires_grad=False)
                        prob_prev_coarse = torch.exp(outputs_coarse[-1].data)
                        it_coarse.index_copy_(0, sample_ind, torch.multinomial(
                            prob_prev_coarse, 1).view(-1).index_select(0, sample_ind))
                        it_coarse = Variable(it_coarse, requires_grad=False)
            else:
                it = seq[:, i].clone()
                it_fine = seq[:, i].clone()
                it_coarse = seq[:, i].clone()
            # break if all the sequences end
            if i >= 1 and seq[:, i].data.sum() == 0:
                break

            xt = self.embed(it)
            if type(self) == C2FTopDownModel:
                xt_fine = self.embed(it_fine)
                xt_coarse = self.embed(it_coarse)
                output, state = self.core(xt, xt_fine, xt_coarse, fc_feats, att_feats, p_att_feats, p_att_feats_final, state)
                outputcoarse = F.log_softmax(self.logit_coarse(output[0]))
                outputfine = F.log_softmax(self.logit(output[1]))
                outputfinal = F.log_softmax(self.logit_final(output[2]))
                outputs_coarse.append(outputcoarse)
                outputs.append(outputfine)
                outputs_final.append(outputfinal)
            else:
                output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state)
                output = F.log_softmax(self.logit(output))
                outputs.append(output)
        if type(self) == C2FTopDownModel:
            return [torch.cat([_.unsqueeze(1) for _ in outputs_coarse], 1), torch.cat([_.unsqueeze(1) for _ in outputs], 1), torch.cat([_.unsqueeze(1) for _ in outputs_final], 1)]
        else:
            return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def get_logprobs_state(self, it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state):
        # 'it' is Variable contraining a word index

        if type(self) == C2FTopDownModel:
            xt = self.embed(it[0])
            xt_fine = self.embed(it[1])
            output, state = self.core(xt, xt_fine, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats[0], tmp_p_att_feats[1], state)
            logprobs = F.log_softmax(self.logit(output[0]))
            logprobsfinal = F.log_softmax(self.logit(output[1]))
            return [logprobs, logprobsfinal], state
        else:
            xt = self.embed(it)
            output, state = self.core(xt, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state)
            logprobs = F.log_softmax(self.logit(output))
            return logprobs, state

    def sample_beam(self, fc_feats, att_feats, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        _att_feats = self.att_embed(att_feats.view(-1, self.att_feat_size))
        att_feats = _att_feats.view(*(att_feats.size()[:-1] + (self.rnn_size,)))

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats.view(-1, self.rnn_size))
        p_att_feats = p_att_feats.view(*(att_feats.size()[:-1] + (self.att_hid_size,)))
        if type(self) == C2FTopDownModel:
            p_att_feats_final = self.ctx2att_final(att_feats.view(-1, self.rnn_size))
            p_att_feats_final = p_att_feats_final.view(*(att_feats.size()[:-1] + (self.att_hid_size,)))

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = fc_feats[k:k+1].expand(beam_size, fc_feats.size(1))
            tmp_att_feats = att_feats[k:k+1].expand(*((beam_size,)+att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = p_att_feats[k:k+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous()
            if type(self) == C2FTopDownModel:
                tmp_p_att_feats_final = p_att_feats_final[k:k+1].expand(*((beam_size,)+p_att_feats_final.size()[1:])).contiguous()

            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(Variable(it, requires_grad=False))
                    if type(self) == C2FTopDownModel:
                        it_fine = fc_feats.data.new(beam_size).long().zero_()
                        xt_fine = self.embed(Variable(it_fine, requires_grad=False))

                if type(self) == C2FTopDownModel:
                    output, state = self.core(xt, xt_fine, xt_coarse, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_p_att_feats_final, state)
                    logprobs = F.log_softmax(self.logit(output[0]))
                    logprobsfinal = F.log_softmax(self.logit(output[1]))
                else:
                    output, state = self.core(xt, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state)
                    logprobs = F.log_softmax(self.logit(output))
            if type(self) == C2FTopDownModel:
                self.done_beams[k] = self.beam_search(state, [logprobs, logprobsfinal], tmp_fc_feats, tmp_att_feats, [tmp_p_att_feats, tmp_p_att_feats_final], opt=opt)
            else:
                self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        if type(self) == C2FTopDownModel:
            return seq.transpose(0, 1), seqLogprobs.transpose(0, 1), seq.transpose(0, 1), seqLogprobs.transpose(0, 1)
        else:
            return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def sample(self, fc_feats, att_feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        _att_feats = self.att_embed(att_feats.view(-1, self.att_feat_size))
        att_feats = _att_feats.view(*(att_feats.size()[:-1] + (self.rnn_size,)))

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats.view(-1, self.rnn_size))
        p_att_feats = p_att_feats.view(*(att_feats.size()[:-1] + (self.att_hid_size,)))
        if type(self) == C2FTopDownModel:
            p_att_feats_final = self.ctx2att_final(att_feats.view(-1, self.rnn_size))
            p_att_feats_final = p_att_feats_final.view(*(att_feats.size()[:-1] + (self.att_hid_size,)))

        seq = []
        seqLogprobs = []
        if type(self) == C2FTopDownModel:
            seq_fine = []
            seqLogprobs_fine = []
            seq_coarse = []
            seqLogprobs_coarse = []
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.data.new(batch_size).long().zero_()
                if type(self) == C2FTopDownModel:
                    it_fine = fc_feats.data.new(batch_size).long().zero_()
                    it_coarse = fc_feats.data.new(batch_size).long().zero_()
            elif sample_max:
                if type(self) == C2FTopDownModel:
                    sampleLogprobs, it = torch.max(logprobsfinal.data, 1)
                    it = it.view(-1).long()
                    sampleLogprobs_fine, it_fine = torch.max(logprobs.data, 1)
                    it_fine = it_fine.view(-1).long()
                    sampleLogprobs_coarse, it_coarse = torch.max(logprobscoarse.data, 1)
                    it_coarse = it_coarse.view(-1).long()
                else:
                    sampleLogprobs, it = torch.max(logprobs.data, 1)
                    it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    if type(self) == C2FTopDownModel:
                        prob_prev = torch.exp(logprobsfinal.data).cpu()
                        prob_prev_fine = torch.exp(logprobs.data).cpu()
                        prob_prev_coarse = torch.exp(logprobscoarse.data).cpu()
                    else:
                        prob_prev = torch.exp(logprobs.data).cpu()  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    if type(self) == C2FTopDownModel:
                        prob_prev = torch.exp(torch.div(logprobsfinal.data, temperature)).cpu()
                        prob_prev_fine = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                        prob_prev_coarse = torch.exp(torch.div(logprobscoarse.data, temperature)).cpu()
                    else:
                        prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()

                if type(self) == C2FTopDownModel:
                    # it_fine = torch.multinomial(prob_prev_fine, 1).cuda()
                    it_fine = torch.max(prob_prev_fine, 1)[1].view(-1, 1).cuda()
                    it_coarse = torch.max(prob_prev_coarse, 1)[1].view(-1, 1).cuda()
                    # sampleLogprobs_fine, it_fine = torch.max(logprobs.data, 1)
                    sampleLogprobs = logprobsfinal.gather(1, Variable(it, requires_grad=False))
                    sampleLogprobs_fine = logprobs.gather(1, Variable(it_fine, requires_grad=False))
                    sampleLogprobs_coarse = logprobscoarse.gather(1, Variable(it_coarse, requires_grad=False))

                    it_fine = it_fine.view(-1).long()  # and flatten indices for downstream processing
                    it_coarse = it_coarse.view(-1).long()  # and flatten indices for downstream processing

                else:
                    sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False))
                    # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing
            xt = self.embed(Variable(it, requires_grad=False))
            if type(self) == C2FTopDownModel:
                xt_fine = self.embed(Variable(it_fine, requires_grad=False))
                xt_coarse = self.embed(Variable(it_coarse, requires_grad=False))

            if type(self) == C2FTopDownModel:
                if t >= 1:
                    # stop when all finished
                    if t == 1:
                        unfinished = it > 0
                        unfinished_fine = it_fine > 0
                        unfinished_coarse = it_coarse > 0
                    else:
                        unfinished = unfinished * (it > 0)
                        unfinished_fine = unfinished_fine * (it_fine > 0)
                        unfinished_coarse = unfinished_coarse * (it_coarse > 0)
                    if unfinished.sum() == 0 and unfinished_fine.sum() == 0 and unfinished_coarse.sum() == 0:
                        break
                    it = it * unfinished.type_as(it)
                    it_fine = it_fine * unfinished_fine.type_as(it_fine)
                    it_coarse = it_coarse * unfinished_coarse.type_as(it_coarse)
                    seq.append(it)  # seq[t] the input of t+2 time step
                    seqLogprobs.append(sampleLogprobs.view(-1))
                    seq_fine.append(it_fine)
                    seqLogprobs_fine.append(sampleLogprobs_fine.view(-1))
                    seq_coarse.append(it_coarse)
                    seqLogprobs_coarse.append(sampleLogprobs_coarse.view(-1))

                output, state = self.core(xt, xt_fine, xt_coarse, fc_feats, att_feats, p_att_feats, p_att_feats_final, state)
                logprobscoarse = F.log_softmax(self.logit_coarse(output[0]))
                logprobs = F.log_softmax(self.logit(output[1]))
                logprobsfinal = F.log_softmax(self.logit_final(output[2]))
            else:
                if t >= 1:
                    # stop when all finished
                    if t == 1:
                        unfinished = it > 0
                    else:
                        unfinished = unfinished * (it > 0)
                    if unfinished.sum() == 0 == 0:
                        break
                    it = it * unfinished.type_as(it)
                    seq.append(it)  # seq[t] the input of t+2 time step
                    seqLogprobs.append(sampleLogprobs.view(-1))

                output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state)
                logprobs = F.log_softmax(self.logit(output))
        if type(self) == C2FTopDownModel:
            return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1), torch.cat([_.unsqueeze(1) for _ in seq_fine], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs_fine], 1), torch.cat([_.unsqueeze(1) for _ in seq_coarse], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs_coarse], 1)
        else:
            return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)

class AdaAtt_lstm(nn.Module):
    def __init__(self, opt, use_maxout=True):
        super(AdaAtt_lstm, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        # self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.use_maxout = use_maxout

        # Build a LSTM
        self.w2h = nn.Linear(self.input_encoding_size, (4+(use_maxout==True)) * self.rnn_size)
        self.v2h = nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size)

        self.i2h = nn.ModuleList([nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size) for _ in range(self.num_layers - 1)])
        self.h2h = nn.ModuleList([nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size) for _ in range(self.num_layers)])

        # Layers for getting the fake region
        if self.num_layers == 1:
            self.r_w2h = nn.Linear(self.input_encoding_size, self.rnn_size)
            self.r_v2h = nn.Linear(self.rnn_size, self.rnn_size)
        else:
            self.r_i2h = nn.Linear(self.rnn_size, self.rnn_size)
        self.r_h2h = nn.Linear(self.rnn_size, self.rnn_size)


    def forward(self, xt, img_fc, state):

        hs = []
        cs = []
        for L in range(self.num_layers):
            # c,h from previous timesteps
            prev_h = state[0][L]
            prev_c = state[1][L]
            # the input to this layer
            if L == 0:
                x = xt
                i2h = self.w2h(x) + self.v2h(img_fc)
            else:
                x = hs[-1]
                x = F.dropout(x, self.drop_prob_lm, self.training)
                i2h = self.i2h[L-1](x)

            all_input_sums = i2h+self.h2h[L](prev_h)

            sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
            sigmoid_chunk = F.sigmoid(sigmoid_chunk)
            # decode the gates
            in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
            forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
            out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
            # decode the write inputs
            if not self.use_maxout:
                in_transform = F.tanh(all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size))
            else:
                in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size)
                in_transform = torch.max(\
                    in_transform.narrow(1, 0, self.rnn_size),
                    in_transform.narrow(1, self.rnn_size, self.rnn_size))
            # perform the LSTM update
            next_c = forget_gate * prev_c + in_gate * in_transform
            # gated cells form the output
            tanh_nex_c = F.tanh(next_c)
            next_h = out_gate * tanh_nex_c
            if L == self.num_layers-1:
                if L == 0:
                    i2h = self.r_w2h(x) + self.r_v2h(img_fc)
                else:
                    i2h = self.r_i2h(x)
                n5 = i2h+self.r_h2h(prev_h)
                fake_region = F.sigmoid(n5) * tanh_nex_c

            cs.append(next_c)
            hs.append(next_h)

        # set up the decoder
        top_h = hs[-1]
        top_h = F.dropout(top_h, self.drop_prob_lm, self.training)
        fake_region = F.dropout(fake_region, self.drop_prob_lm, self.training)

        state = (torch.cat([_.unsqueeze(0) for _ in hs], 0),
                torch.cat([_.unsqueeze(0) for _ in cs], 0))
        return top_h, fake_region, state

class AdaAtt_attention(nn.Module):
    def __init__(self, opt):
        super(AdaAtt_attention, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_hid_size = opt.att_hid_size

        # fake region embed
        self.fr_linear = nn.Sequential(
            nn.Linear(self.rnn_size, self.input_encoding_size),
            nn.ReLU(),
            nn.Dropout(self.drop_prob_lm))
        self.fr_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)

        # h out embed
        self.ho_linear = nn.Sequential(
            nn.Linear(self.rnn_size, self.input_encoding_size),
            nn.Tanh(),
            nn.Dropout(self.drop_prob_lm))
        self.ho_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)

        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.att2h = nn.Linear(self.rnn_size, self.rnn_size)

    def forward(self, h_out, fake_region, conv_feat, conv_feat_embed):

        # View into three dimensions
        att_size = conv_feat.numel() // conv_feat.size(0) // self.rnn_size
        conv_feat = conv_feat.view(-1, att_size, self.rnn_size)
        conv_feat_embed = conv_feat_embed.view(-1, att_size, self.att_hid_size)

        # view neighbor from bach_size * neighbor_num x rnn_size to bach_size x rnn_size * neighbor_num
        fake_region = self.fr_linear(fake_region)
        fake_region_embed = self.fr_embed(fake_region)

        h_out_linear = self.ho_linear(h_out)
        h_out_embed = self.ho_embed(h_out_linear)

        txt_replicate = h_out_embed.unsqueeze(1).expand(h_out_embed.size(0), att_size + 1, h_out_embed.size(1))

        img_all = torch.cat([fake_region.view(-1,1,self.input_encoding_size), conv_feat], 1)
        img_all_embed = torch.cat([fake_region_embed.view(-1,1,self.input_encoding_size), conv_feat_embed], 1)

        hA = F.tanh(img_all_embed + txt_replicate)
        hA = F.dropout(hA,self.drop_prob_lm, self.training)

        hAflat = self.alpha_net(hA.view(-1, self.att_hid_size))
        PI = F.softmax(hAflat.view(-1, att_size + 1))

        visAtt = torch.bmm(PI.unsqueeze(1), img_all)
        visAttdim = visAtt.squeeze(1)

        atten_out = visAttdim + h_out_linear

        h = F.tanh(self.att2h(atten_out))
        h = F.dropout(h, self.drop_prob_lm, self.training)
        return h

class AdaAttCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(AdaAttCore, self).__init__()
        self.lstm = AdaAtt_lstm(opt, use_maxout)
        self.attention = AdaAtt_attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state):
        h_out, p_out, state = self.lstm(xt, fc_feats, state)
        atten_out = self.attention(h_out, p_out, att_feats, p_att_feats)
        return atten_out, state

class TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats)

        lang_lstm_input = torch.cat([att, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state

class C2FTopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(C2FTopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        self.finelang_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # h^1_t, \hat v
        self.finallang_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)
        self.proj_ctx_coarse = nn.Linear(opt.input_encoding_size, opt.rnn_size)
        self.proj_ctx = nn.Linear(opt.input_encoding_size, opt.rnn_size)
        self.proj_ctx_final = nn.Linear(opt.input_encoding_size, opt.rnn_size)
        self.attention_fine = Attention(opt)
        self.attention_final = Attention(opt)

    def forward(self, xt, xt_fine, xt_coarse, fc_feats, att_feats, p_att_feats, p_att_feats_final, state):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        output_att = h_att + fc_feats
        output_att += self.proj_ctx_coarse(xt_coarse)
        output_att = F.dropout(output_att, self.drop_prob_lm, self.training)

        att_fine = self.attention_fine(h_att, att_feats, p_att_feats)

        finelang_lstm_input = torch.cat([att_fine, h_att, xt], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang_fine, c_lang_fine = self.finelang_lstm(finelang_lstm_input, (state[0][1], state[1][1]))
        output_fine = h_lang_fine + att_fine
        output_fine += self.proj_ctx(xt_fine)
        output_fine = F.dropout(output_fine, self.drop_prob_lm, self.training)

        att_final = self.attention_final(h_lang_fine + att_fine, att_feats, p_att_feats_final)
        finallang_lstm_input = torch.cat([att_final, h_lang_fine, xt], 1)

        h_lang_final, c_lang_final = self.finallang_lstm(finallang_lstm_input, (state[0][2], state[1][2]))
        output_final = h_lang_final + att_final
        output_final += self.proj_ctx_final(xt)
        output_final = F.dropout(output_final, self.drop_prob_lm, self.training)

        state = (torch.stack([h_att, h_lang_fine, h_lang_final]), torch.stack([c_att, c_lang_fine, c_lang_final]))
        output = (output_att, output_fine, output_final)
        return output, state

class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = F.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size

        weight = F.softmax(dot)                             # batch * att_size
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res


class Att2in2Core(nn.Module):
    def __init__(self, opt):
        super(Att2in2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        #self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        # Build a LSTM
        self.a2c = nn.Linear(self.rnn_size, 2 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state):
        att_res = self.attention(state[0][-1], att_feats, p_att_feats)

        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size) + \
            self.a2c(att_res)
        in_transform = torch.max(\
            in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state

class AdaAttModel(AttModel):
    def __init__(self, opt):
        super(AdaAttModel, self).__init__(opt)
        self.core = AdaAttCore(opt)

# AdaAtt with maxout lstm
class AdaAttMOModel(AttModel):
    def __init__(self, opt):
        super(AdaAttMOModel, self).__init__(opt)
        self.core = AdaAttCore(opt, True)

class Att2in2Model(AttModel):
    def __init__(self, opt):
        super(Att2in2Model, self).__init__(opt)
        self.core = Att2in2Core(opt)
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x : x

class TopDownModel(AttModel):
    def __init__(self, opt):
        super(TopDownModel, self).__init__(opt)
        self.num_layers = 2
        self.core = TopDownCore(opt)

class C2FTopDownModel(AttModel):
    def __init__(self, opt):
        super(C2FTopDownModel, self).__init__(opt)
        self.ctx2att_final = nn.Linear(self.rnn_size, self.att_hid_size)
        self.logit_final = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.logit_coarse = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.num_layers = 3
        self.core = C2FTopDownCore(opt)

