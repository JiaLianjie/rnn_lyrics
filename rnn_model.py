#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import funcs
import torch.nn as nn
import pdb


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state): # inputs: (batch, seq_len)
        # 获取one-hot向量表示
        # pdb.set_trace()
        X = funcs.to_onehot(inputs, self.vocab_size)
        pdb.set_trace()
        Y, self.state = self.rnn(torch.stack(X), state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size,
        # num_hiddens)，它的输出形状为(num_steps * batch_size,
        # vocab_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state
    