from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc']:
        return False
    return True

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.


def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq > 0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = input * Variable(mask)
        # print(torch.pow((1. - torch.exp(output)), 1).float())
        # output = output * torch.pow((1. - torch.exp(output)), 1)
        output = output * reward
        output = - torch.sum(output) / torch.sum(mask)

        return output


class LanguageModelCriterion(nn.Module):
    def __init__(self, gamma=0):
        self.gamma = gamma
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = input.gather(1, target) * mask
        # print(torch.pow((1. - torch.exp(output)), 1).float())
        # output = output * torch.pow((1. - torch.exp(output)), self.gamma)
        output = - torch.sum(output) / torch.sum(mask)

        return output


class c2fLanguageModelCriterion(nn.Module):
    def __init__(self, gamma=0):
        self.gamma = gamma
        super(c2fLanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        input_coarse = input[0]
        input_fine = input[1]
        input_final = input[2]

        target_coarse = target[:, :input_coarse.size(1)]
        mask_coarse = mask[:, :input_coarse.size(1)]
        input_coarse = to_contiguous(input_coarse).view(-1, input_coarse.size(2))
        target_coarse = to_contiguous(target_coarse).view(-1, 1)
        mask_coarse = to_contiguous(mask_coarse).view(-1, 1)
        output_coarse = input_coarse.gather(1, target_coarse) * mask_coarse
        # tmp_coarse = torch.pow(torch.clamp(1. - torch.exp(output_coarse), 1e-16, 1.), self.gamma)
        # # print(tmp.float())
        # output_coarse = output_coarse * tmp_coarse * mask_coarse
        # (torch.sum(tmp_coarse * mask_coarse) / torch.sum(mask_coarse))
        output_coarse = - torch.sum(output_coarse) / torch.sum(mask_coarse)

        # truncate to the same size
        target_fine = target[:, :input_fine.size(1)]
        mask_fine = mask[:, :input_fine.size(1)]
        input_fine = to_contiguous(input_fine).view(-1, input_fine.size(2))
        target_fine = to_contiguous(target_fine).view(-1, 1)
        mask_fine = to_contiguous(mask_fine).view(-1, 1)
        output_fine = input_fine.gather(1, target_fine) * mask_fine
        # tmp_fine = torch.pow(torch.clamp(1. - torch.exp(output_fine), 1e-16, 1.), self.gamma)
        # # print(tmp.float())
        # output_fine = output_fine * tmp_fine * mask_fine
        output_fine = - torch.sum(output_fine) / torch.sum(mask_fine)

        target_final = target[:, :input_final.size(1)]
        mask_final = mask[:, :input_final.size(1)]
        input_final = to_contiguous(input_final).view(-1, input_final.size(2))
        target_final = to_contiguous(target_final).view(-1, 1)
        mask_final = to_contiguous(mask_final).view(-1, 1)
        output_final = input_final.gather(1, target_final) * mask_final
        # tmp_final = torch.pow(torch.clamp(1. - torch.exp(output_final), 1e-16, 1.), self.gamma)
        # # print(tmp_final.float())
        # output_final = output_final * tmp_final * mask_final
        output_final = - torch.sum(output_final) / torch.sum(mask_final)

        return output_fine + output_final + output_coarse


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)