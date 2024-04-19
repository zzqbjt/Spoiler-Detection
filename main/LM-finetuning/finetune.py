import pandas as pd
from transformers import RobertaModel, RobertaTokenizer, MobileBertTokenizer, MobileBertModel
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import json
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
import math
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
setup_seed(42)
tokenizer = RobertaTokenizer.from_pretrained('../../roberta-base')

class Review(Dataset):
    def __init__(self, rinput_ids, rattention_mask, label):
        self.rii = rinput_ids
        self.ram = rattention_mask
        self.y = label
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.rii[index], self.ram[index], self.y[index]

class Model(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('../../roberta-base')
        self.out = nn.Linear(768, 2)
     
    def forward(self, ids, attn):
        reviews = self.roberta(ids, attn).pooler_output
        return self.out(reviews)

edge_index = torch.load('./data/edge_index.pt')
edge_type = torch.load('./data/edge_type.pt')
input_ids = torch.load("./data/input_ids.pt").to(torch.long)
attention_mask = torch.load("./data/attention_mask.pt").to(torch.long)
rii = input_ids[1572:]
ram = attention_mask[1572:]
label = torch.load("./data/label.pt")[263407+1572:].to(torch.long)

train_mask = torch.load('./data/train_idx.pt')
valid_mask = torch.load('./data/val_idx.pt')
test_mask = torch.load('./data/test_idx.pt')

train_set = Review(rii[train_mask], ram[train_mask], label[train_mask])
valid_set = Review(rii[valid_mask], ram[valid_mask], label[valid_mask])
test_set = Review(rii[test_mask], ram[test_mask], label[test_mask])

batch_size = 32
train_loader = DataLoader(train_set, batch_size, True)
valid_loader = DataLoader(valid_set, batch_size, False)
test_loader = DataLoader(test_set, batch_size, False)

epochs = 1
lr = 1e-5
warmup = 0.1
model = Model()
# model.load_state_dict(torch.load('encoders/roberta_dot_product.pth'))
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
loss_func = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=lr, step_size_up=warmup*len(train_loader)*epochs, step_size_down=(1-warmup)*len(train_loader)*epochs, cycle_momentum=False)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
model, optimizer, scheduler, train_loader, valid_loader, test_loader = accelerator.prepare(model, optimizer, scheduler, train_loader, valid_loader, test_loader)

for epoch in range(epochs):
    model.train()
    step = 0
    total_train_loss = 0
    total_train_acc = 0
    total = 0
    for rii, ram, y in tqdm(train_loader, disable=not accelerator.is_local_main_process):
        optimizer.zero_grad()
        out = model(rii, ram)
        loss = loss_func(out, y)

        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

        pred = out.argmax(1)
        acc = (pred == y).sum() / len(y)

        total_train_loss += loss.item()
        total_train_acc += acc

        step += 1
        if step % 1000 == 0:
            accelerator.print(f"Train Loss: {total_train_loss/step:.5f}, Train Acc: {total_train_acc/step:.5f}")

    total_train_loss /= len(train_loader)
    total_train_acc /= len(train_loader)

    with torch.no_grad():
        total_valid_loss = 0
        total_valid_acc = 0
        for rii, ram, y in tqdm(valid_loader, disable=not accelerator.is_local_main_process):
            out = model(rii, ram)
            loss = loss_func(out, y)

            pred = out.argmax(1)
            acc = (pred == y).sum() / len(y)

            total_valid_loss += loss.item()
            total_valid_acc += acc
    
    total_valid_loss /= len(valid_loader)
    total_valid_acc /= len(valid_loader)

    accelerator.print(f"""Epoch: {epoch + 1}: Train Loss: {total_train_loss:.4f}, Train Acc: {total_train_acc:.4f}
    Valid Loss: {total_valid_loss:.4f}, Valid Acc: {total_valid_acc:.4f}""")

if accelerator.is_local_main_process:
    model = accelerator.unwrap_model(model)
    torch.save(model.state_dict(), './models/roberta.pth')

LABEL = torch.tensor([])
PRED = torch.tensor([])
model.eval()

with torch.no_grad():
    for rii, ram, y in tqdm(test_loader, disable=not accelerator.is_local_main_process):
        out = model(rii, ram)

        pred = out.argmax(1).to(torch.long)
        pred, y = accelerator.gather_for_metrics((pred, y))

        PRED = torch.cat((PRED, pred.cpu()))
        LABEL = torch.cat((LABEL, y.cpu()))

acc = metrics.accuracy_score(LABEL, PRED)
f1 = metrics.f1_score(LABEL, PRED)
auc = metrics.roc_auc_score(LABEL, PRED)
torch.save(PRED, 'RoBERTa_pred.pt')
accelerator.print(f'Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}')
