# Actor-Critic Sequence Training for Image Captioning

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *


class CriticModel(nn.Module):
    def __init__(self, opt):
        super(CriticModel, self).__init__()
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_hid = nn.Sequential(nn.Linear(self.rnn_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        self.fc_out = nn.Sequential(nn.Linear(self.rnn_size, 1),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))

    def forward(self, hid_states):
        h1 = self.fc_hid(hid_states)
        h2 = self.fc_out(h1)
        # print('hid_state size,', hid_states.size())
        return h2.view(-1, hid_states.size()[1])

    