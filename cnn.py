# coding=utf-8

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class CNN(nn.Module):
  def __init__(self, args):
    super(CNN, self).__init__()
    self.dp = args.dp # Position embedding dimension
    self.vac_len_pos = args.vac_len_pos
    self.dw = args.dw
    self.vac_len_word = args.vac_len_word
    self.vac_len_rel = args.vac_len_rel
    self.dc = args.dc
    self.seq_len = args.seq_len
    self.dropout_rate = args.dropout_rate

    self.word_embedding = nn.Embedding(self.vac_len_word, self.dw)
    self.word_embedding.weight.data.copy_(torch.from_numpy(args.word_embedding))
    self.pos_embedding_pos1 = nn.Embedding(self.vac_len_pos, self.dp)
    self.pos_embedding_pos2 = nn.Embedding(self.vac_len_pos, self.dp)

    self.dropout = nn.Dropout(args.dropout_rate)

    # self.convs = nn.ModuleList([nn.Conv1d(self.dw + 2 * self.dp, self.dc,\
    #   kernel_size=kernel_size, padding=int((kernel_size - 1) / 2)) for kernel_size in args.kernel_sizes])
    self.convs = nn.ModuleList([nn.Sequential(
        nn.Conv1d(self.dw + 2 * self.dp, self.dc, kernel_size=kernel_size,padding=int((kernel_size - 0) / 2)),
        nn.Tanh(),
        nn.MaxPool1d(self.seq_len)
    ) for kernel_size in args.kernel_sizes])
    
    self.fc = nn.Linear(self.dc * len(args.kernel_sizes) + self.dw * 2, self.vac_len_rel)
  
  def _k_maxpooling(self, input, dim, k=1):
    index = input.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return input.gather(dim, index)
  
  def forward(self, W, W_pos1, W_pos2, e1, e2):
    e1 = self.word_embedding(e1)
    e2 = self.word_embedding(e2)
    W = self.word_embedding(W)
    W_pos1 = self.pos_embedding_pos1(W_pos1)
    W_pos2 = self.pos_embedding_pos2(W_pos2)

    Wa = torch.cat([W, W_pos1, W_pos2], dim=2)

    conv = [conv(Wa.permute(0, 2, 1)) for conv in self.convs]
    conv = torch.cat(conv, dim=1)

    conv = self.dropout(conv)

    e_concat = torch.cat([e1, e2], dim=1)

    all_concat = torch.cat([e_concat.view(e_concat.size(0), -1), conv.view(conv.size(0), -1)], dim=1)

    out = self.fc(all_concat)

    out = F.softmax(out)

    return out

