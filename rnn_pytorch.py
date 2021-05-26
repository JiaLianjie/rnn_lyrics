#!/usr/bin/python
# -*- coding: UTF-8 -*-

import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import sys
import funcs
from rnn_model import RNNModel
import pdb


def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char, char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]]  # output会记录prefix加上输出
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor(output[-1], device=device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple): # LSTM, state:(h, c)
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)

        Y, state = model(X, state) # 前向计算不需要传入模型参数
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))

    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                  corpus_indices, idx_to_char, char_to_idx,
                                  num_epoches, num_steps, lr, clipping_theta,
                                  batch_size, pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    for epoch in range(num_epoches):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = funcs.data_iter_consecutive(corpus_indices, batch_size, num_steps, device) # 相邻采样
        for X, Y in data_iter:
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                if isinstance(state, tuple): # LSTM, state:(h, c)
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()

            output, state = model(X, state) # output: 形状为(num_steps * batch_size, vocab_size)
            # pdb.set_trace()

            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())

            optimizer.zero_grad()
            l.backward()
            #梯度裁剪
            funcs.grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf')

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, perplexity, time.time() - start))
            for prefix in prefixes:
                print('-', predict_rnn_pytorch(
                    prefix, pred_len, model, vocab_size, device, idx_to_char, char_to_idx
                ))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
corpus_indices, char_to_idx, idx_to_char, vocab_size = funcs.load_data_jay_lyrics()

# 定义模型
num_hiddens = 256
num_steps = 35
batch_size = 2

rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
# rnn_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens)
# rnn_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
#
# X = torch.rand(num_steps, batch_size, vocab_size)
# Y, state_new = rnn_layer(X, state)
# print(Y.shape, len(state_new), state_new[0].shape)

model = RNNModel(rnn_layer, vocab_size).to(device)
# predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx)
num_epoches, batch_size, lr, clipping_theta = 500, 32, 1e-3, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
# pred_period, pred_len, prefixes = 50, 50, ['天青色等烟雨', '可爱的女人']
train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                              corpus_indices, idx_to_char, char_to_idx,
                              num_epoches, num_steps, lr, clipping_theta,
                              batch_size, pred_period, pred_len, prefixes)
