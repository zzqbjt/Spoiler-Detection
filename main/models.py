import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import RGCNConv, GATConv, HGTConv, GCNConv, TransformerConv, RGATConv
from torch_geometric.sampler import NeighborSampler
from torch_geometric.utils import softmax, k_hop_subgraph
import math     

class GCNModel(nn.Module):
     def __init__(self, in_dim=768, hidden_dim=256, out_dim=128, dropout=0.3):
          super().__init__()

          self.dropout = dropout
          self.linear_relu1 = nn.Sequential(
               nn.Linear(in_dim,hidden_dim),
               nn.ReLU()
          )
          self.text1 = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
          self.text2 = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
          self.linear_relu2 = nn.Sequential(
               nn.Linear(hidden_dim,out_dim),
               nn.ReLU()
          )

          self.classifier = nn.Linear(out_dim, 2)

     def forward(self, data):
          origin, edge_index = data.x, data.edge_index

          textual = self.linear_relu1(origin)

          textual = self.text1(textual, edge_index)
          textual = F.dropout(textual, p=self.dropout, training=self.training)   

          textual = self.text2(textual, edge_index)
          textual = self.linear_relu2(textual)
          
          x = self.classifier(textual)
          return  x[:data.batch_size]

     def embed(self, data):
          textual, edge_index = data.x, data.edge_index

          textual = self.linear_relu1(textual)
          textual = self.text1(textual, edge_index)
          textual = F.dropout(textual, p=self.dropout, training=self.training)   

          textual = self.text2(textual, edge_index)
          textual = self.linear_relu2(textual)
          return  textual

class RGCNModel(nn.Module):   
     def __init__(self, in_dim=768, hidden_dim=256, out_dim=128, num_relations=3, dropout=0.3):
          super().__init__()
          self.dropout = nn.Dropout(dropout)
          self.linear1 = nn.Sequential(
               nn.Linear(in_dim,hidden_dim),
               nn.ReLU()
          )
          self.conv1 = RGCNConv(in_channels=hidden_dim, out_channels=hidden_dim, num_relations=num_relations)
          self.conv2 = RGCNConv(in_channels=hidden_dim, out_channels=hidden_dim, num_relations=num_relations)
          self.linear2 = nn.Sequential(
               nn.Linear(hidden_dim,out_dim),
               nn.ReLU()
          )
          self.relu = nn.ReLU()
          self.classifier = nn.Linear(out_dim, 2)

     def forward(self, data):
          origin, edge_index, edge_type = data.x, data.edge_index, data.edge_type

          textual = self.linear1(origin)

          textual = self.conv1(textual, edge_index, edge_type)
          textual = self.dropout(textual)   

          textual = self.conv2(textual,edge_index, edge_type)
          textual = self.linear2(textual)
          
          x = self.classifier(textual)
          return  x[:data.batch_size]

     def embed(self, data):
          origin, edge_index, edge_type = data.x, data.edge_index, data.edge_type
          textual = self.linear1(origin)

          textual = self.conv1(textual, edge_index, edge_type)
          textual = self.dropout(textual)   
          textual = self.relu(textual)

          textual = self.conv2(textual, edge_index, edge_type)
          textual = self.linear2(textual)
          return  textual

class GATModel(nn.Module):
     def __init__(self, in_dim=768, hidden_dim=256, out_dim=64, heads=8, dropout=0.3):
          super().__init__()
          self.dropout = nn.Dropout(dropout)
          self.linear1 = nn.Sequential(
               nn.Linear(in_dim, hidden_dim),
               nn.ReLU()
          )
          self.conv1 = GATConv(in_channels=hidden_dim, out_channels=hidden_dim // heads, heads=heads)
          self.conv2 = GATConv(in_channels=hidden_dim, out_channels=hidden_dim // heads, heads=heads)
          self.linear2 = nn.Sequential(
               nn.Linear(hidden_dim, out_dim),
               nn.ReLU()
          )
          self.relu = nn.ReLU()
          self.classifier = nn.Linear(out_dim, 2)

     def forward(self, data):
          x = self.embed(data)
          x = self.classifier(x)
          return  x[:data.batch_size], 0

     def embed(self, data):
          x, edge_index = data.x, data.edge_index

          x = self.linear1(x)
          x = self.conv1(x, edge_index)
          
          x = self.dropout(x) 
          x = self.relu(x)
            
          x = self.conv2(x, edge_index)
          x = self.linear2(x)
          return x

class MoE(torch.nn.Module):
     def __init__(self, dim=256, hidden_dim=None, out_dim=None, num_experts=4, k=2, dropout=0.2):
          super().__init__()
          if hidden_dim is None:
               hidden_dim = 4 * dim
          if out_dim is None:
               out_dim = dim
          self.softplus = nn.Softplus()
          self.gate = nn.Linear(dim, num_experts, bias=False)
          self.noise = nn.Linear(dim, 1, bias=False)
          self.experts = nn.ModuleList([nn.Sequential(nn.Linear(dim, hidden_dim), nn.Dropout(dropout), nn.ReLU(), nn.Linear(hidden_dim, out_dim)) for _ in range(num_experts)])
          
          self.num_experts = num_experts
          self.k = k
          self.dim = dim  
     
     def loss(self, logits):
          tmp = logits.sum(0)
          loss = (tmp.std() / tmp.mean()) ** 2
          return loss             
     
     def forward(self, x, query=None):
          if query is None:
               query = x
          g = self.gate(query)
          n = self.softplus(self.noise(query))
          h = g + torch.randn(g.shape).to(x.device) * n
          _, indices = torch.topk(h, self.k, dim=1, largest=False)
          h[torch.arange(x.shape[0]).unsqueeze(1), indices] = -torch.inf
          L = F.softmax(h, dim=1)
          logits, indices = L.topk(self.k, dim=1)
          
          loss = self.loss(L)
          
          input, map_list = self.split(x, indices)
          output = []
          for i in range(self.num_experts):
               output.append(self.experts[i](input[i]))
          out = self.gather(output, map_list, logits)
          return out, loss
     
     def split(self, x, indices):
          index = torch.arange(x.shape[0]).repeat(self.k).to(x.device)
          x = x.repeat(self.k, 1)
          indices = indices.view(-1)
          out = []
          map_list = []
          for i in range(self.num_experts):
               mask = indices == i
               out.append(x[mask])
               map_list.append(index[mask])
          return out, map_list
     
     def gather(self, x_list, map_list, logits):
          out = torch.zeros(logits.shape[0], self.dim).to(logits.device)
          logits = logits.view(-1, 1)
          for i in range(self.num_experts):
               out[map_list[i]] += logits[map_list[i]] * x_list[i]
          return out                         

class Gate(nn.Module):
     def __init__(self, dim):
          super().__init__()
          self.rate = nn.Sequential(
               nn.Linear(dim*2, 1),
               nn.Sigmoid()
          )
          self.tanh = nn.Tanh()
     def forward(self, x, y):
          seq = torch.cat([x, y], dim=1)
          r = self.rate(seq)
          out = self.tanh(y) * r + x * (1 - r)
          return out

class GAT3View(nn.Module):
     def __init__(self, in_dim=768, meta_dim=6, hidden_dim=256, out_dim=128, dropout=0.3, num_experts=4, k=2):
          super().__init__()
          self.graphencoder = GATModel(in_dim+meta_dim, hidden_dim, out_dim, 1, dropout)
          self.in_dim = in_dim
          self.textencoder = nn.Sequential(
               nn.Linear(in_dim, out_dim),
               nn.ReLU()
          )
          self.metaencoder = nn.Sequential(
               nn.Linear(meta_dim, hidden_dim),
               nn.Dropout(dropout), 
               nn.ReLU(),
               nn.Linear(hidden_dim, out_dim),
               nn.ReLU()
          )
          self.fusion = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=out_dim, nhead=4, dim_feedforward=4*out_dim, dropout=dropout, batch_first=True), num_layers=4)      
          self.classifier = nn.Sequential(
               nn.Flatten(),
               nn.Linear(out_dim*3, 2) 
          )

     def forward(self, data):
          x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

          meta = x[:data.batch_size, self.in_dim:]
          text = x[:data.batch_size, :self.in_dim]
          
          graph = self.graphencoder.embed(data)
          x_g = graph[:data.batch_size]
          x_t = self.textencoder(text)
          x_m = self.metaencoder(meta)
          
          input = torch.cat([x_g.unsqueeze(1), x_t.unsqueeze(1), x_m.unsqueeze(1)], dim=1)
          output = self.fusion(input)
          return self.classifier(output), 0

class GAT3View_MoE(nn.Module):
     def __init__(self, in_dim=768, meta_dim=6, hidden_dim=256, out_dim=128, dropout=0.3, num_experts=4, k=2):
          super().__init__()
          self.graphencoder = GATModel(in_dim+meta_dim, hidden_dim, out_dim, 1, dropout)
          self.in_dim = in_dim
          self.textencoder = nn.Sequential(
               nn.Linear(in_dim, out_dim),
               nn.ReLU()
          )
          self.metaencoder = nn.Sequential(
               nn.Linear(meta_dim, hidden_dim),
               nn.Dropout(dropout),
               nn.ReLU(),
               nn.Linear(hidden_dim, out_dim),
               nn.ReLU()
          )
          self.fusion = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=out_dim, nhead=4, dim_feedforward=4*out_dim, dropout=dropout, batch_first=True), num_layers=4)      
          self.classifier = nn.Sequential(
               nn.Flatten(),
               nn.Linear(out_dim*3, 2) 
          )
          self.moe = nn.ModuleList([MoE(dim=out_dim, num_experts=num_experts, k=k) for _ in range(3)])

     def forward(self, data):
          x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

          meta = x[:data.batch_size, self.in_dim:]
          text = x[:data.batch_size, :self.in_dim]
          
          graph = self.graphencoder.embed(data)
          x_g = graph[:data.batch_size]
          x_t = self.textencoder(text)
          x_m = self.metaencoder(meta)

          x_g, loss1 = self.moe[0](x_g)
          x_t, loss2 = self.moe[1](x_t)
          x_m, loss3 = self.moe[2](x_m)
          
          loss = loss1 + loss2 + loss3
          input = torch.cat([x_g.unsqueeze(1), x_t.unsqueeze(1), x_m.unsqueeze(1)], dim=1)
          output = self.fusion(input)
          return self.classifier(output), loss

class GAT3View_MLP(nn.Module):
     def __init__(self, in_dim=768, meta_dim=6, hidden_dim=256, out_dim=128, dropout=0.3, num_experts=4, k=2):
          super().__init__()
          self.graphencoder = GATModel(in_dim+meta_dim, hidden_dim, out_dim, 1, dropout)
          self.in_dim = in_dim
          self.textencoder = nn.Sequential(
               nn.Linear(in_dim, out_dim),
               nn.ReLU()
          )
          self.metaencoder = nn.Sequential(
               nn.Linear(meta_dim, hidden_dim),
               nn.Dropout(dropout),
               nn.ReLU(),
               nn.Linear(hidden_dim, out_dim),
               nn.ReLU()
          )
          self.fusion = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=out_dim, nhead=4, dim_feedforward=4*out_dim, dropout=dropout, batch_first=True), num_layers=4)      
          self.classifier = nn.Sequential(
               nn.Flatten(),
               nn.Linear(out_dim*3, 2) 
          )
          self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(out_dim, out_dim*4), nn.Dropout(dropout), nn.ReLU(), nn.Linear(out_dim*4, out_dim)) for _ in range(3)])

     def forward(self, data):
          x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

          meta = x[:data.batch_size, self.in_dim:]
          text = x[:data.batch_size, :self.in_dim]
          graph = self.graphencoder.embed(data)
          
          x_g = graph[:data.batch_size]
          x_t = self.textencoder(text)
          x_m = self.metaencoder(meta)

          x_g = self.mlp[0](x_g)
          x_t = self.mlp[1](x_t)
          x_m = self.mlp[2](x_m)
          
          input = torch.cat([x_g.unsqueeze(1), x_t.unsqueeze(1), x_m.unsqueeze(1)], dim=1)
          output = self.fusion(input)
          return self.classifier(output), 0


class RGCN3View_MoE(nn.Module):
     def __init__(self, in_dim=768, meta_dim=6, hidden_dim=256, out_dim=128, dropout=0.3, num_experts=4, k=2):
          super().__init__()
          self.graphencoder = RGCNModel(in_dim+meta_dim, hidden_dim, out_dim, 3, dropout)
          self.in_dim = in_dim
          self.textencoder = nn.Sequential(
               nn.Linear(in_dim, out_dim),
               nn.ReLU()
          )
          self.metaencoder = nn.Sequential(
               nn.Linear(meta_dim, hidden_dim),
               nn.Dropout(dropout),
               nn.ReLU(),
               nn.Linear(hidden_dim, out_dim),
               nn.ReLU()
          )
          self.fusion = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=out_dim, nhead=4, dim_feedforward=4*out_dim, dropout=dropout, batch_first=True), num_layers=4)      
          self.classifier = nn.Sequential(
               nn.Flatten(),
               nn.Linear(out_dim*3, 2) 
          )
          self.moe = nn.ModuleList([MoE(dim=out_dim, num_experts=num_experts, k=k) for _ in range(3)])

     def forward(self, data):
          x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

          meta = x[:data.batch_size, self.in_dim:]
          text = x[:data.batch_size, :self.in_dim]
          
          graph = self.graphencoder.embed(data)
          x_g = graph[:data.batch_size]
          x_t = self.textencoder(text)
          x_m = self.metaencoder(meta)

          x_g, loss1 = self.moe[0](x_g)
          x_t, loss2 = self.moe[1](x_t)
          x_m, loss3 = self.moe[2](x_m)
          
          loss = loss1 + loss2 + loss3
          input = torch.cat([x_g.unsqueeze(1), x_t.unsqueeze(1), x_m.unsqueeze(1)], dim=1)
          output = self.fusion(input)
          return self.classifier(output), loss

class GAT3View_GatedMoE(nn.Module):
     def __init__(self, in_dim=768, meta_dim=6, hidden_dim=256, out_dim=128, dropout=0.3, num_experts=4, k=2):
          super().__init__()
          self.graphencoder = GATModel(in_dim+meta_dim, hidden_dim, out_dim, 1, dropout)
          self.in_dim = in_dim
          self.textencoder = nn.Sequential(
               nn.Linear(in_dim, out_dim),
               nn.ReLU()
          )
          self.metaencoder = nn.Sequential(
               nn.Linear(meta_dim, hidden_dim),
               nn.Dropout(dropout),
               nn.ReLU(),
               nn.Linear(hidden_dim, out_dim),
               nn.ReLU()
          )
          self.fusion = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=out_dim, nhead=4, dim_feedforward=4*out_dim, dropout=dropout, batch_first=True), num_layers=4)      
          self.classifier = nn.Sequential(
               nn.Flatten(),
               nn.Linear(out_dim*3, 2) 
          )
          self.moe = nn.ModuleList([MoE(dim=out_dim, num_experts=num_experts, k=k, dropout=dropout) for _ in range(3)])
          self.gate = nn.ModuleList([Gate(out_dim) for _ in range(3)])

     def forward(self, data):
          x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

          meta = x[:data.batch_size, self.in_dim:]
          text = x[:data.batch_size, :self.in_dim]
          
          graph = self.graphencoder.embed(data)
          x_g = graph[:data.batch_size]
          x_t = self.textencoder(text)
          x_m = self.metaencoder(meta)

          x1, loss1 = self.moe[0](x_g)
          x_g = self.gate[0](x_g, x1)
          
          x2, loss2 = self.moe[1](x_t)
          x_t = self.gate[1](x_t, x2)
          
          x3, loss3 = self.moe[2](x_m)
          x_m = self.gate[2](x_m, x3)
          
          loss = loss1 + loss2 + loss3
          input = torch.cat([x_g.unsqueeze(1), x_t.unsqueeze(1), x_m.unsqueeze(1)], dim=1)
          output = self.fusion(input)
          return self.classifier(output), loss

class GAT3View_GatedMoE_Cat(nn.Module):
     def __init__(self, in_dim=768, meta_dim=6, hidden_dim=256, out_dim=128, dropout=0.3, num_experts=4, k=2):
          super().__init__()
          self.graphencoder = GATModel(in_dim+meta_dim, hidden_dim, out_dim, 1, dropout)
          self.in_dim = in_dim
          self.textencoder = nn.Sequential(
               nn.Linear(in_dim, out_dim),
               nn.ReLU()
          )
          self.metaencoder = nn.Sequential(
               nn.Linear(meta_dim, hidden_dim),
               nn.Dropout(dropout),
               nn.ReLU(),
               nn.Linear(hidden_dim, out_dim),
               nn.ReLU()
          )      
          self.classifier = nn.Linear(out_dim*3, 2) 
          self.moe = nn.ModuleList([MoE(dim=out_dim, num_experts=num_experts, k=k, dropout=dropout) for _ in range(3)])
          self.gate = nn.ModuleList([Gate(out_dim) for _ in range(3)])

     def forward(self, data):
          x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

          meta = x[:data.batch_size, self.in_dim:]
          text = x[:data.batch_size, :self.in_dim]
          
          graph = self.graphencoder.embed(data)
          x_g = graph[:data.batch_size]
          x_t = self.textencoder(text)
          x_m = self.metaencoder(meta)

          x1, loss1 = self.moe[0](x_g)
          x_g = self.gate[0](x_g, x1)
          
          x2, loss2 = self.moe[1](x_t)
          x_t = self.gate[1](x_t, x2)
          
          x3, loss3 = self.moe[2](x_m)
          x_m = self.gate[2](x_m, x3)
          
          loss = loss1 + loss2 + loss3
          input = torch.cat([x_g, x_t, x_m], dim=1)
          return self.classifier(input), loss

class GAT3View_GatedMoE_Mean(nn.Module):
     def __init__(self, in_dim=768, meta_dim=6, hidden_dim=256, out_dim=128, dropout=0.3, num_experts=4, k=2):
          super().__init__()
          self.graphencoder = GATModel(in_dim+meta_dim, hidden_dim, out_dim, 1, dropout)
          self.in_dim = in_dim
          self.textencoder = nn.Sequential(
               nn.Linear(in_dim, out_dim),
               nn.ReLU()
          )
          self.metaencoder = nn.Sequential(
               nn.Linear(meta_dim, hidden_dim),
               nn.Dropout(dropout),
               nn.ReLU(),
               nn.Linear(hidden_dim, out_dim),
               nn.ReLU()
          )      
          self.classifier = nn.Linear(out_dim, 2) 
          self.moe = nn.ModuleList([MoE(dim=out_dim, num_experts=num_experts, k=k, dropout=dropout) for _ in range(3)])
          self.gate = nn.ModuleList([Gate(out_dim) for _ in range(3)])

     def forward(self, data):
          x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

          meta = x[:data.batch_size, self.in_dim:]
          text = x[:data.batch_size, :self.in_dim]
          
          graph = self.graphencoder.embed(data)
          x_g = graph[:data.batch_size]
          x_t = self.textencoder(text)
          x_m = self.metaencoder(meta)

          x1, loss1 = self.moe[0](x_g)
          x_g = self.gate[0](x_g, x1)
          
          x2, loss2 = self.moe[1](x_t)
          x_t = self.gate[1](x_t, x2)
          
          x3, loss3 = self.moe[2](x_m)
          x_m = self.gate[2](x_m, x3)
          
          loss = loss1 + loss2 + loss3
          input = (x_g + x_t + x_m) / 3
          return self.classifier(input), loss

class GAT3View_GatedMoE_Max(nn.Module):
     def __init__(self, in_dim=768, meta_dim=6, hidden_dim=256, out_dim=128, dropout=0.3, num_experts=4, k=2):
          super().__init__()
          self.graphencoder = GATModel(in_dim+meta_dim, hidden_dim, out_dim, 1, dropout)
          self.in_dim = in_dim
          self.textencoder = nn.Sequential(
               nn.Linear(in_dim, out_dim),
               nn.ReLU()
          )
          self.metaencoder = nn.Sequential(
               nn.Linear(meta_dim, hidden_dim),
               nn.Dropout(dropout),
               nn.ReLU(),
               nn.Linear(hidden_dim, out_dim),
               nn.ReLU()
          )
          self.fusion = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=out_dim, nhead=4, dim_feedforward=4*out_dim, dropout=dropout, batch_first=True), num_layers=4)      
          self.classifier = nn.Linear(out_dim, 2) 
          self.moe = nn.ModuleList([MoE(dim=out_dim, num_experts=num_experts, k=k, dropout=dropout) for _ in range(3)])
          self.gate = nn.ModuleList([Gate(out_dim) for _ in range(3)])

     def forward(self, data):
          x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

          meta = x[:data.batch_size, self.in_dim:]
          text = x[:data.batch_size, :self.in_dim]
          
          graph = self.graphencoder.embed(data)
          x_g = graph[:data.batch_size]
          x_t = self.textencoder(text)
          x_m = self.metaencoder(meta)

          x1, loss1 = self.moe[0](x_g)
          x_g = self.gate[0](x_g, x1)
          
          x2, loss2 = self.moe[1](x_t)
          x_t = self.gate[1](x_t, x2)
          
          x3, loss3 = self.moe[2](x_m)
          x_m = self.gate[2](x_m, x3)
          
          loss = loss1 + loss2 + loss3
          input = torch.cat([x_g.unsqueeze(1), x_t.unsqueeze(1), x_m.unsqueeze(1)], dim=1)
          output, _ = input.max(1)
          return self.classifier(output), loss

class GAT2View_GatedMoE_GT(nn.Module):
     def __init__(self, in_dim=768, meta_dim=6, hidden_dim=256, out_dim=128, dropout=0.3, num_experts=4, k=2):
          super().__init__()
          self.graphencoder = GATModel(in_dim+meta_dim, hidden_dim, out_dim, 1, dropout)
          self.in_dim = in_dim
          self.textencoder = nn.Sequential(
               nn.Linear(in_dim, out_dim),
               nn.ReLU()
          )
          self.fusion = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=out_dim, nhead=4, dim_feedforward=4*out_dim, dropout=dropout, batch_first=True), num_layers=4)      
          self.classifier = nn.Sequential(
               nn.Flatten(),
               nn.Linear(out_dim*2, 2) 
          )
          self.moe = nn.ModuleList([MoE(dim=out_dim, num_experts=num_experts, k=k, dropout=dropout) for _ in range(2)])
          self.gate = nn.ModuleList([Gate(out_dim) for _ in range(2)])

     def forward(self, data):
          x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

          text = x[:data.batch_size, :self.in_dim] 
          graph = self.graphencoder.embed(data)
          
          x_g = graph[:data.batch_size]
          x_t = self.textencoder(text)
          
          x1, loss1 = self.moe[0](x_g)
          x_g = self.gate[0](x_g, x1)
          
          x2, loss2 = self.moe[1](x_t)
          x_t = self.gate[1](x_t, x2)
          
          loss = loss1 + loss2
          input = torch.cat([x_g.unsqueeze(1), x_t.unsqueeze(1)], dim=1)
          output = self.fusion(input)
          return self.classifier(output), loss
