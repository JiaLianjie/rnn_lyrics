#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 语言模型数据集（周杰伦专辑歌词）

import torch
import torch.nn as nn
import random
import zipfile
import pdb
import funcs
import time
import math


# 定义模型训练函数
def train_and_predict_rnn(rnn, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epoches, num_steps,
                          lr, clipping_theta, batch_size, pred_period, pred_len, prefixes):
    pdb.set_trace()
    if is_random_iter:
        data_iter_fn = funcs.data_iter_random
    else:
        data_iter_fn = funcs.data_iter_consecutive
    params = funcs.get_params(num_inputs, num_hiddens, num_outputs, device)
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epoches):
        if not is_random_iter:   # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter: # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, device)
            else: # 否则需要使用detach函数从计算图分离隐藏状态
                for s in state:
                    s.detach_()

            pdb.set_trace()
            inputs = funcs.to_onehot(X, vocab_size)
            # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
            outputs, state = rnn(inputs, state, params)
            # 拼接之后形状为(num_steps * batch_size, vocab_size)
            outputs = torch.cat(outputs, dim=0)
            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            # 使用交叉熵损失计算平均分类误差
            l = loss(outputs, y.long())

            # 梯度清0
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            funcs.grad_clipping(params, clipping_theta, device)
            funcs.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print('-', funcs.predict_rnn(prefix, pred_len, rnn, params, init_rnn_state,
                                       num_hiddens, vocab_size, device, idx_to_char, char_to_idx))



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(device)
num_epoches, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e-3, 1e2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
# 读取数据集
corpus_indices, char_to_idx, idx_to_char, vocab_size = funcs.load_data_jay_lyrics()

# 初始化模型参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
# get_params()

rnn = funcs.rnn
init_rnn_state = funcs.init_rnn_state
train_and_predict_rnn(rnn, init_rnn_state, num_hiddens,
                      vocab_size, device, corpus_indices, idx_to_char,
                      char_to_idx, True, num_epoches, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len, prefixes)




# 时序数据的采用
# 随机采样