# coding=utf-8

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
import argparse
from cnn import CNN

from utils import *
import evaluate

def accuracy(preds, labels):
  total = preds.size(0)
  preds = torch.max(preds, dim=1)[1]
  correct = (preds == labels).sum()
  acc = correct.cpu().data[0] / total
  return acc

def main():
  parser = argparse.ArgumentParser("CNN")
  parser.add_argument("--dp", type=int, default=5)
  parser.add_argument("--dc", type=int, default=32)
  parser.add_argument("--lr", type=float, default=0.001)
  parser.add_argument("--seq_len", type=int, default=120)
  parser.add_argument("--vac_len_rel", type=int, default=19)
  parser.add_argument("--nepoch", type=int ,default=100)
  parser.add_argument("--num_workers", type=int, default=4)
  parser.add_argument("--eval_every", type=int, default=10)
  parser.add_argument("--dropout_rate", type=float, default=0.4)
  parser.add_argument("--bz", type=int, default=128)
  parser.add_argument("--kernel_sizes", type=str, default="3,4,5")
  parser.add_argument('--gpu', type=int, default=0)
  parser.add_argument('--train_filename', default='./data/train.txt')
  parser.add_argument('--test_filename', default='./data/test.txt')
  parser.add_argument('--model_file', default='./cnn.pt')
  parser.add_argument('--embedding_filename', default='./data/embeddings.txt')
  parser.add_argument('--embedding_wordlist_filename', default='./data/words.lst')

  args = parser.parse_args()
  args.kernel_sizes = list(map(int, args.kernel_sizes.split(',')))
  # Initilization
  torch.cuda.set_device(args.gpu)

  # Load data
  dataset = SemEvalDataset(args.train_filename, max_len=args.seq_len)
  dataloader = DataLoader(dataset, args.bz, True, num_workers=args.num_workers)
  dataset_val = SemEvalDataset(args.test_filename, max_len=args.seq_len, d=(dataset.d, dataset.rel_d))
  dataloader_val = DataLoader(dataset_val, args.bz, True, num_workers=args.num_workers)
  args.word_embedding = load_embedding(args.embedding_filename, args.embedding_wordlist_filename,\
    dataset.d)
  args.vac_len_pos = 122
  args.vac_len_word = len(dataset.d.word2id)
  args.vac_len_rel = len(dataset.rel_d.word2id)
  args.dw = args.word_embedding.shape[1]


  print(args)

  model = CNN(args).cuda()
  loss_func = nn.CrossEntropyLoss()
  # optimizer = optim.SGD(model.parameters(), lr=0.2)
  optimizer = optim.Adam(model.parameters(), lr=0.01)
  best_eval_acc = 0.

  for i in range(args.nepoch):
    # Training
    total_loss = 0.
    total_acc = 0.
    ntrain_batch = 0
    model.train()
    for (seq, e1, e2, dist1, dist2, r) in dataloader:
      ntrain_batch += 1
      seq = Variable(seq).cuda()
      e1 = Variable(e1).cuda()
      e2 = Variable(e2).cuda()
      dist1 = Variable(dist1).cuda()
      dist2 = Variable(dist2).cuda()
      r = Variable(r).cuda()
      r = r.view(r.size(0))

      pred = model(seq, dist1, dist2, e1, e2)
      l = loss_func(pred, r)
      acc = accuracy(pred, r)
      total_acc += acc
      total_loss += l.data[0]

      optimizer.zero_grad()
      l.backward()
      optimizer.step()
    print("Epoch: {}, Training loss : {:.4}, acc: {:.4}".\
      format(i, total_loss/ntrain_batch, total_acc / ntrain_batch))

    # Evaluation
    if i % args.eval_every == args.eval_every - 1:
      val_total_acc = 0.
      nval_batch = 0
      model.eval()
      for (seq, e1, e2, dist1, dist2, r) in dataloader_val:
        nval_batch += 1
        seq = Variable(seq).cuda()
        e1 = Variable(e1).cuda()
        e2 = Variable(e2).cuda()
        dist1 = Variable(dist1).cuda()
        dist2 = Variable(dist2).cuda()
        r = Variable(r).cuda()
        r = r.view(r.size(0))

        pred = model(seq, dist1, dist2, e1, e2)
        acc = accuracy(pred, r)
        val_total_acc += acc
      best_eval_acc = max(best_eval_acc, val_total_acc/nval_batch)
      print("Epoch: {}, Val acc: {:.4f}".\
        format(i, val_total_acc/nval_batch))
  print(best_eval_acc)
  torch.save(model.state_dict(), args.model_file)

if __name__ == '__main__':
  main()