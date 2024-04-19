import torch
import json
from tqdm import tqdm
from torch import nn
import math
from torch.utils.data import Dataset, DataLoader, random_split
from accelerate import Accelerator

edge_index = torch.load('../data/edge_index.pt')
edge_type = torch.load("../data/edge_type.pt")
label = torch.load('../data/label.pt')
f = torch.load('../data/tf_ft_lp.pt')

ml = 16

samples = torch.zeros(259705, ml+1, 768)
mask = torch.zeros(259705, ml+1)
y= torch.zeros(259705, ml+1)
train = torch.zeros(259705, ml+1).bool()
valid = torch.zeros(259705, ml+1).bool()

train_mask = torch.load('../data/train_index.pt') + 406896
valid_mask = torch.load('../data/val_index.pt') + 406896
train_mask, _ = torch.sort(train_mask)
valid_mask, _ = torch.sort(valid_mask)
t = 0
v = 0

for i in tqdm(range(259705)):
    samples[i, 0] = f[i]
    mask[i, 0] = 1

for i in tqdm(range(len(edge_type))):
    if edge_type[i] == 1:
        uid = edge_index[0, i]
        rid = edge_index[1, i]
        for j in range(ml+1):
            if mask[uid, j] == 0:
                samples[uid, j] = f[rid]
                y[uid, j] = label[rid]
                mask[uid, j] = 1
                try:
                    while train_mask[t] < rid:
                        t += 1
                    if train_mask[t] == rid:
                        train[uid, j] = True
                except:
                    ...
                try:
                    while valid_mask[v] < rid:
                        v += 1
                    if valid_mask[v] == rid:
                        valid[uid, j] = True
                except:
                    ...
                break


torch.save(samples, './pretrain/samples-16-ft-lp.pt')
torch.save(mask, './pretrain/mask-16.pt')
torch.save(y, './pretrain/label-16.pt')
torch.save(train, './pretrain/train-16.pt')
torch.save(valid, './pretrain/valid-16.pt')
