import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from models import GAT3View_GatedMoE, RGCN3View_GatedMoE, GAT3View_MoE, GAT3View_TransformerMoE, GAT3View_MLP, GAT3View, GAT3View_GatedMoE_Cat, GATModel, GAT3View_GatedMoE_Mean, GAT3View_GatedMoE_Max, GAT2View_GatedMoE_GT, GAT2View_GatedMoE_GM, GAT2View_GatedMoE_MT
import json
from scipy import stats
import numpy as np
from sklearn import metrics
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm
import math
from torch import nn
import random
import os
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch_geometric.transforms as T
import matplotlib.pyplot as plt

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class Trainer:
    def __init__(
        self,
        epochs: int=60,
        device: str="cuda:7",
        k_hop=2,
        model_dir="./records/",
        path = "../data/",
        lr=1e-4,
        weight_decay=1e-4,
        w=1e-2,
        batch_size=512,
        optimizer=torch.optim.AdamW,
        lr_scheduler=torch.optim.lr_scheduler.ExponentialLR
    ):
        self.path = path
        self.k = k_hop
        self.batch_size = batch_size
        self.device = torch.device(device)        
        self.load_data()          

        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay 
        self.w = w
        self.model = GAT3View_MoE(in_dim=768, meta_dim=6, hidden_dim=512, out_dim=256, dropout=0.2, num_experts=2, k=1)
        self.model.load_state_dict(torch.load('./final_models/ft+up+GAT3View+MoE2+Transformer5.pth'))

        self.opt = optimizer(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        if lr_scheduler:
            self.lr_scheduler = lr_scheduler(self.opt, gamma=0.95)
        else:
            self.lr_scheduler = None
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        self.model, self.opt, self.train_loader, self.val_loader, self.test_loader = self.accelerator.prepare(self.model, self.opt, self.train_loader, self.val_loader, self.test_loader)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def load_data(self):
        path = self.path
        device = self.device

        edge_index = torch.load(path + "edge_index.pt").to(torch.long)
        edge_type = torch.load(path + "edge_type.pt").to(torch.long).squeeze()

        feature = torch.load(path + "tf_ft_lp.pt")
        user = torch.load('../preliminary/users/user_ft_lp.pt')  
        feature = torch.cat([user, feature[259705:]], dim=0)
                       
        meta = torch.load(path + "mf6.pt")
        feature = torch.cat([feature, meta], dim=1)
        label = torch.load(path + "label.pt").to(torch.long)
        
        train_mask = torch.load(path + 'train_index.pt') + 406896
        valid_mask = torch.load(path + 'val_index.pt') + 406896
        test_mask = torch.load(path + 'test_index.pt') + 406896

        node_type = torch.tensor([0]*259705+[1]*147191+[2]*1860715).unsqueeze(1)

        data = Data(x=feature, edge_index=edge_index, edge_type=edge_type, y=label, pos=node_type)
        self.train_loader = NeighborLoader(data, num_neighbors=[200]*2, input_nodes=train_mask, directed=True, batch_size=self.batch_size, shuffle=True)
        self.val_loader = NeighborLoader(data, num_neighbors=[200]*2, input_nodes=valid_mask, directed=True, batch_size=self.batch_size)
        self.test_loader = NeighborLoader(data, num_neighbors=[200]*2, input_nodes=test_mask, directed=True, batch_size=self.batch_size) 

    def train(self):
        for epoch in range(self.epochs):
            total_train_acc = 0
            total_train_loss = 0
            total_train = 0
            self.model.train()
            for batch in tqdm(self.train_loader, disable=not self.accelerator.is_local_main_process):
                out, loss = self.model(batch)
                y = batch.y[:batch.batch_size]
                train_loss = self.loss_func(out, y) + loss * self.w
                
                self.opt.zero_grad()
                self.accelerator.backward(train_loss)
                self.opt.step()

                _, pred = out.max(1)
                correct = (pred == y).sum().item()

                total_train_acc += correct
                total_train_loss += train_loss.item() * batch.batch_size
                total_train += batch.batch_size
            
            total_train_acc /= total_train
            total_train_loss /= total_train

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            self.accelerator.print(f"Epoch: {epoch + 1}, Train Loss: {total_train_loss:.4f}, Train Accuracy: {total_train_acc:.4f}")
            self.test(self.val_loader)
    
    @torch.no_grad()
    def test(self, loader):
        LABEL = torch.tensor([], device=self.accelerator.device)
        PRED = torch.tensor([], device=self.accelerator.device)
        OUT = torch.tensor([], device=self.accelerator.device)

        self.model.eval()
        total_loss = 0
        for batch in tqdm(loader, disable=not self.accelerator.is_local_main_process):
            out, loss = self.model(batch)
            y = batch.y[:batch.batch_size]
            total_loss += loss
            
            _, pred = out.max(1)
            pred, y = self.accelerator.gather_for_metrics((pred, y))

            PRED = torch.cat((PRED, pred))
            LABEL = torch.cat((LABEL, y))

        LABEL = LABEL.cpu()
        PRED = PRED.cpu()
        
        acc = metrics.accuracy_score(LABEL, PRED)
        f1 = metrics.f1_score(LABEL, PRED)
        auc = metrics.roc_auc_score(LABEL, PRED)
        avg_loss = total_loss / len(loader)
        self.accelerator.print(f'Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, BL loss: {avg_loss:.4f}')
        

def main():
    trainer=Trainer()
    # trainer.train()
    trainer.test(trainer.test_loader)
    if trainer.accelerator.is_local_main_process:
        model = trainer.accelerator.unwrap_model(trainer.model)
        torch.save(model.state_dict(), './final_models/ft+up+GAT3View+MoE2+Transformer5.pth')
    
if __name__ == "__main__":
    setup_seed(42)
    main()
