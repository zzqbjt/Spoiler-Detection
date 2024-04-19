import pandas as pd
from transformers import RobertaModel, RobertaTokenizer 
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

tokenizer = RobertaTokenizer.from_pretrained('../../roberta-base')

class Review(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.ii = input_ids
        self.am = attention_mask

    def __len__(self):
        return self.ii.shape[0]
    def __getitem__(self, index):
        return self.ii[index], self.am[index]

class Model(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('../../roberta-base')
        self.out = nn.Linear(768, 2)
     
    def forward(self, ids, attn):
        reviews = self.roberta(ids, attn).pooler_output
        return reviews

input_ids = torch.load("./data/input_ids.pt").to(torch.long)
attention_mask = torch.load("./data/attention_mask.pt").to(torch.long)

Set = Review(input_ids, attention_mask)
batch_size = 32
loader = DataLoader(Set, batch_size, False)

model = Model()
model.load_state_dict(torch.load('./models/roberta.pth', map_location='cpu'))

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
model, loader = accelerator.prepare(model, loader)

TF = torch.tensor([], device=accelerator.device)
model.eval()
with torch.no_grad():
    # try: 
    for ii, am in tqdm(loader, disable=not accelerator.is_local_main_process):
        out = model(ii, am)
        out = accelerator.gather_for_metrics(out)
        TF = torch.cat([TF, out], dim=0)
    # except:
    #     ...

print(TF.shape)
torch.save(TF.cpu(), './data/tf_ft.pt')
