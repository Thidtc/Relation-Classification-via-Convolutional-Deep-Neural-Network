# coding=utf-8
import numpy as np
import torch
from collections import Counter
from torch.utils.data import Dataset

class SemEvalDataset(Dataset):
  def __init__(self, filename, max_len, d=None):
    seqs, e1_pos, e2_pos, rs = load_data(filename)
    self.max_len = max_len
    if d is None:
      self.d = build_dict(seqs)
      self.rel_d = build_dict([[r] for r in rs], add_extra=False)
    else:
      self.d = d[0]
      self.rel_d = d[1]
    self.seqs, self.e1s, self.e2s, self.dist1s, self.dist2s =\
      self.vectorize_seq(seqs, e1_pos, e2_pos)
    self.rs = np.array([[self.rel_d.word2id[r]] for r in rs])
  
  def vectorize_seq(self, seqs, e1_pos, e2_pos):
    new_seqs = np.zeros((len(seqs), self.max_len))
    dist1s = np.zeros((len(seqs), self.max_len))
    dist2s = np.zeros((len(seqs), self.max_len))
    e1s = np.zeros((len(seqs), 1))
    e2s = np.zeros((len(seqs), 1))
    for r, (seq, e1_p, e2_p) in enumerate(zip(seqs, e1_pos, e2_pos)):
      seq = list(map(lambda x: self.d.word2id[x] if x in self.d.word2id else 0, seq))
      dist1 = list(map(map_pos, [idx - e1_p[1] for idx, _ in enumerate(seq)])) # Last word
      dist2 = list(map(map_pos, [idx - e2_p[1] for idx, _ in enumerate(seq)]))
      e1s[r] = seq[e1_p[1]]
      e2s[r] = seq[e2_p[1]]
      for i in range(min(self.max_len, len(seq))):
        new_seqs[r, i] = seq[i]
        dist1s[r, i] = dist1[i]
        dist2s[r, i] = dist2[i]
    return new_seqs, e1s, e2s, dist1s, dist2s

  def __len__(self):
    return len(self.seqs)
  
  def __getitem__(self, index):
    seq = torch.from_numpy(self.seqs[index]).long()
    e1 = torch.from_numpy(self.e1s[index]).long()
    e2 = torch.from_numpy(self.e2s[index]).long()
    dist1 = torch.from_numpy(self.dist1s[index]).long()
    dist2 = torch.from_numpy(self.dist2s[index]).long()
    r = torch.from_numpy(self.rs[index]).long()
    return seq, e1, e2, dist1, dist2, r

def load_data(filename):
  seqs = []
  e1_pos = []
  e2_pos = []
  rs = []
  with open(filename, 'r') as f:
    for line in f:
      data = line.strip().split('\t')
      data[0] = data[0].lower().split(' ')
      seqs.append(data[0])
      e1_pos.append((int(data[1]), int(data[2])))
      e2_pos.append((int(data[3]), int(data[4])))
      rs.append(data[5])
  return seqs, e1_pos, e2_pos, rs

class Dictionary(object):
  def __init__(self):
    self.word2id = {}
    self.id2word = []
  def add_word(self, word):
    if word not in self.word2id:
      self.word2id[word] = len(self.id2word)
      self.id2word.append(word)

def build_dict(seqs, add_extra=True, dict_size=100000):
  d = Dictionary()
  cnt = Counter()
  for seq in seqs:
    cnt.update(seq)
  d = Dictionary()
  if add_extra:
    d.add_word(None) # 0 for not in the dictionary
  for word, cnt in cnt.most_common()[:dict_size]:
    d.add_word(word)
  return d

def map_pos(p):
  if p < -60:
    return 0
  elif -60 <= p < 60:
    return p + 61
  else:
    return 121

def load_embedding(embedding_file, word_list_file, d):
  word_list = {}
  with open(word_list_file) as f:
    for i, line in enumerate(f):
      word_list[line.strip()] = i

  with open(embedding_file, 'r') as f:
    lines = f.readlines()
  def process_line(line):
    return list(map(float, line.split(' ')))
  lines = list(map(process_line, lines))

  val_len = len(d.id2word)
  dw = len(lines[0])
  embedding = np.random.uniform(-0.01, 0.01, size=[val_len, dw])
  num_pretrained = 0
  for i in range(1, val_len):
    if d.id2word[i] in word_list:
      embedding[i,:] = lines[word_list[d.id2word[i]]]
      num_pretrained += 1
  
  print('#pretrained: {}, #vocabulary: {}'.format(num_pretrained, val_len))
  return embedding