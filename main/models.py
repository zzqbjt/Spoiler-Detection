import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import RGCNConv, GATConv, HGTConv, GCNConv, TransformerConv, RGATConv
from torch_geometric.sampler import NeighborSampler
from torch_geometric.utils import softmax, k_hop_subgraph
import math     

metadata = (['user', 'movie', 'review'], [('movie', 'commented by', 'review'), ('user', 'post', 'review'), ('review', 'posted by', 'user')])

def rand_zero(tensor, prob):
    random = torch.rand(tensor.shape)
    tensor[random < prob] = 0
    return tensor

def to_edge_index_list(edge_index, edge_type, num_relations):
    edge_index_list = []
    for i in range(num_relations):
        edge_index_list.append(edge_index[:, edge_type == i])
    return edge_index_list

def transpose(X, r):
    X = X.reshape(X.shape[0], -1, r)
    X = X.permute(0, 2, 1)
    return X

def Transpose(X, h, r):
    X = X.reshape(X.shape[0], r, h, -1)
    X = X.permute(1, 2, 0, 3)
    return X

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

class RGATModel(nn.Module):
     def __init__(self, in_dim=768, hidden_dim=256, out_dim=64, heads=1, num_relations=3, dropout=0.3):
          super().__init__()
          self.dropout = dropout
          self.linear_relu1 = nn.Sequential(
               nn.Linear(in_dim,hidden_dim),
               nn.ReLU()
          )
          self.text1 = RGATConv(in_channels=hidden_dim, out_channels=hidden_dim // heads, num_relations=num_relations, heads=heads)
          self.text2 = RGATConv(in_channels=hidden_dim, out_channels=hidden_dim // heads, num_relations=num_relations, heads=heads)
          self.linear_relu = nn.Sequential(
               nn.Linear(hidden_dim,out_dim),
               nn.ReLU()
          )
          self.classifier = nn.Linear(out_dim, 2)

     def forward(self, data):
          textual = self.embed(data)
          x = self.classifier(textual)

          return  x

     def embed(self, data):
          textual, edge_index, edge_type = data.x, data.edge_index, data.edge_type

          textual = self.linear_relu1(textual)
          textual = self.text1(textual, edge_index, edge_type)

          textual = self.text2(textual, edge_index, edge_type)        
          textual = self.linear_relu(textual)

          return textual

class GraphTransformer(nn.Module):
     def __init__(self, in_dim=768, hidden_dim=256, out_dim=64, heads=4, dropout=0.3):
          super().__init__()
          self.dim = in_dim
          self.Wk = nn.Linear(out_dim, out_dim)
          self.Wq = nn.Linear(out_dim, out_dim)
          # self.Wv = nn.Linear(out_dim, out_dim)

          self.dropout = nn.Dropout(dropout)
          self.linear_relu1 = nn.Sequential(
               nn.Linear(in_dim,hidden_dim),
               nn.ReLU()
          )
          self.linear_relu2 = nn.Sequential(
               nn.Linear(hidden_dim,out_dim),
               nn.ReLU()
          )

          self.conv1 = TransformerConv(in_channels=hidden_dim, out_channels=hidden_dim // heads, heads=heads, dropout=dropout)
          self.conv2 = TransformerConv(in_channels=hidden_dim, out_channels=hidden_dim // heads, heads=heads, dropout=dropout)
          
          self.classifier = nn.Linear(out_dim, 2)  
          # self.gate = nn.Linear(hidden_dim, 1)
     
     def embed(self, data):
          x, edge_index = data.x, data.edge_index 
          x = self.linear_relu1(x)
          x = self.conv1(x, edge_index)

          # h = self.gate(y)
          # x = h * y + (1 - h) * x
          x = self.dropout(x)
               
          x = self.conv2(x, edge_index)
          # h = self.gate(y)
          # x = h * y + (1 - h) * x

          x = self.linear_relu2(x)
          return x

     def loss(self, data):
          x, edge_index = data.x, data.edge_index
          emb = self.embed(data)
          adj = torch.zeros(emb.shape[0], emb.shape[0]).to(x.device)
          adj[edge_index[0], edge_index[1]] = 1.

          mask = adj.sum(1) != 0
          adj = adj[mask]
          adj = adj / adj.sum(1).unsqueeze(1)

          Q = self.Wq(emb)
          K = self.Wk(emb)
          pred = Q @ K.T / math.sqrt(emb.shape[1])
          pred = F.softmax(pred, dim=1)
          pred = pred[mask]

          loss = nn.MSELoss()(pred, adj) * pred.shape[0]
          
          return loss

     def forward(self, data):
          x = self.embed(data)
          return self.classifier(x)[:data.batch_size]
          # return self.loss(data)

class Semantic(torch.nn.Module): #语义注意力
     def __init__(self, in_dim, out_dim, num_heads, num_relations):
          super(Semantic, self).__init__()
          self.out_dim = out_dim
          self.num_heads = num_heads
          self.num_relations = num_relations

          self.Wd = torch.nn.ModuleList()
          for _ in range(num_heads):
               self.Wd.append(torch.nn.Sequential(
                    torch.nn.Linear(in_dim, out_dim),
                    torch.nn.Tanh(),
                    torch.nn.Linear(out_dim, 1, bias=False),
               ))
          self.softmax = torch.nn.Softmax(dim=0)

     def forward(self, H): #H: 5301*2*128          
          output = torch.zeros(H.shape[0], H.shape[-1]).to(H.device)
          for d in range(self.num_heads):
               W = self.Wd[d](H).mean(0).squeeze()
               beta = self.softmax(W)
               for r in range(self.num_relations):
                    output += (beta[r].item()) * H[:, r, :]
          output /= self.num_heads
          return output

class RGTLayer(torch.nn.Module): #RGT层，调用TransformerConv实现
     def __init__(self, in_dim, out_dim, num_heads, num_relations, dropout):
          super(RGTLayer, self).__init__()
          self.out_dim = out_dim
          self.num_heads = num_heads
          self.num_relations = num_relations

          self.layers = torch.nn.ModuleList()
          for _ in range(num_relations):
               self.layers.append(TransformerConv(in_channels=in_dim, out_channels=out_dim, 
                                                      heads=num_heads, dropout=dropout, concat=False))
          self.gate = torch.nn.Sequential(
               torch.nn.Linear(in_dim + out_dim, in_dim),
               torch.nn.Sigmoid()
          )         
          self.tanh = torch.nn.Tanh()

          self.semantic = Semantic(in_dim, in_dim, num_heads, num_relations)

     def forward(self, x, edge_index_list):
          H = torch.zeros(x.shape[0], self.num_relations, self.out_dim).to(x.device) #H 5301*128
          for r in range(self.num_relations):
               U = self.layers[r](x, edge_index_list[r])
               Z = self.gate(torch.cat([U, x], dim=1))
               H[:, r, :] = self.tanh(U) * Z + x * (1 - Z)
          
          return self.semantic(transpose(H, self.num_relations))

class RGTModel(torch.nn.Module): #RGT模型
     def __init__(self, in_dim=768, hidden_dim=256, out_dim=2, num_heads=8, num_relations=3, dropout=0.3):
          super(RGTModel, self).__init__()
          self.dropout = dropout 
          self.num_relations = num_relations
          self.linear_relu1 = nn.Sequential(
               nn.Linear(in_dim, hidden_dim),
               nn.ReLU()
          )

          self.conv1 = RGTLayer(in_dim=hidden_dim, out_dim=hidden_dim, num_heads = num_heads, num_relations=num_relations, dropout=dropout)
          self.conv2 = RGTLayer(in_dim=hidden_dim, out_dim=hidden_dim, num_heads = num_heads, num_relations=num_relations, dropout=dropout)
          self.linear_relu = nn.Sequential(
               nn.Linear(hidden_dim, out_dim),
               nn.ReLU()
          )
          self.classifier = nn.Linear(out_dim, 2)

     def forward(self, data):
          x, edge_index, edge_type = data.x, data.edge_index, data.edge_type 
          edge_index_list = to_edge_index_list(edge_index, edge_type, self.num_relations)
          x = self.linear_relu1(x)
          x = self.conv1(x, edge_index_list)
          x = F.dropout(x, p=self.dropout, training=self.training)

          x = self.conv2(x, edge_index_list)
          x = self.linear_relu(x)
          return self.classifier(x)
     
     def embed(self, data): 
          x, edge_index, edge_type = data.x, data.edge_index, data.edge_type 
          edge_index_list = to_edge_index_list(edge_index, edge_type, self.num_relations)
          x = self.linear_relu1(x)
          x = self.conv1(x, edge_index_list)
          x = F.dropout(x, p=self.dropout, training=self.training)

          x = self.conv2(x, edge_index_list)
          x = self.linear_relu(x)
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

class HGN3View_MoE(nn.Module):
     def __init__(self, in_dim=768, meta_dim=6, hidden_dim=256, out_dim=128, dropout=0.3, num_experts=4, k=2):
          super().__init__()
          self.graphencoder = HGNModel(in_dim+meta_dim, hidden_dim, out_dim, dropout, 0.05)
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

class GAT3View_MoE_Cat(nn.Module):
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
          input = torch.cat([x_g, x_t, x_m], dim=1)
          return self.classifier(input), loss

class GAT3View_MoE_Mean(nn.Module):
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
          input = (x_g + x_t + x_m) / 3
          return self.classifier(input), loss

class GAT3View_MoE_Max(nn.Module):
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
          output, _ = input.max(1)
          return self.classifier(output), loss

class MyModel2View(nn.Module):
     def __init__(self, in_dim=768, hidden_dim=256, out_dim=128, dropout=0.3):
          super().__init__()
          self.graphencoder = GraphTransformer(in_dim+5, hidden_dim, out_dim, 8, dropout)
          self.in_dim = in_dim
          self.textencoder = nn.Sequential(
               nn.Linear(in_dim, hidden_dim),
               # nn.Dropout(dropout),
               nn.ReLU(),
               nn.Linear(hidden_dim, out_dim)
          )
          self.metaencoder = nn.Sequential(
               nn.Linear(5, hidden_dim),
               # nn.Dropout(dropout),
               nn.ReLU(),
               nn.Linear(hidden_dim, out_dim)
          )
          self.fusion = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=out_dim, nhead=4, dim_feedforward=4*out_dim, dropout=dropout, batch_first=True), num_layers=4)      
          self.classifier = nn.Sequential(
               nn.Flatten(),
               nn.Linear(out_dim*2, 2) 
          )

     def forward(self, data):
          x = data.x
          meta = x[:, self.in_dim:]
          x = x[:, :self.in_dim]

          # x_g = self.graphencoder.embed(data)[:data.batch_size].unsqueeze(1)
          x_t = self.textencoder(x[:data.batch_size]).unsqueeze(1)
          x_m = self.metaencoder(meta[:data.batch_size]).unsqueeze(1)
          
          input = torch.cat([x_t, x_m], dim = 1)
          output = self.fusion(input)
          return  self.classifier(output)

class HGTModel(nn.Module):
     def __init__(self, in_dim=768, hidden_dim=256, out_dim=128, heads=4, dropout=0.3):
          super().__init__()

          self.dropout = dropout
          self.linear_relu1 = nn.Sequential(
               nn.Linear(in_dim,hidden_dim),
               nn.ReLU()
          )
          self.text1 = HGTConv(in_channels=hidden_dim, out_channels=hidden_dim, metadata=metadata, heads=heads)
          self.text2 = HGTConv(in_channels=hidden_dim, out_channels=hidden_dim, metadata=metadata, heads=heads)
          self.linear_relu2 = nn.Sequential(
               nn.Linear(hidden_dim,out_dim),
               nn.ReLU()
          )

          self.classifier = nn.Linear(out_dim, 2)

     def forward(self, data):
          origin = {key: data[key].x for key in data.metadata()[0]}
          edge_index = {key: data[key].edge_index for key in data.metadata()[1]}

          textual = {key: self.linear_relu1(origin[key]) for key in origin.keys()}
          movie = textual['movie']

          textual = self.text1(textual, edge_index)
          textual['movie'] = movie
          textual = {key: F.dropout(textual[key], p=self.dropout, training=self.training) for key in textual.keys()}
          
          textual = self.text2(textual, edge_index)
          textual = self.linear_relu2(textual['review'][:data['review'].batch_size])

          return self.classifier(textual)

     def embed(self, data):
          origin = {key: data[key].x for key in data.metadata()[0]}
          edge_index = {key: data[key].edge_index for key in data.metadata()[1]}

          textual = {key: self.linear_relu1(origin[key]) for key in origin.keys()}
          movie = textual['movie']

          textual = self.text1(textual, edge_index)
          textual['movie'] = movie
          textual = {key: F.dropout(textual[key], p=self.dropout, training=self.training) for key in textual.keys()}
          
          textual = self.text2(textual, edge_index)
          textual = self.linear_relu2(textual['review'])

          return  textual

class MyHeteroModel(nn.Module):
     def __init__(self, in_dim=768, hidden_dim=512, out_dim=256, dropout=0.3):
          super().__init__()
          self.graphencoder = HGTModel(in_dim+5, hidden_dim, out_dim, 8, dropout)
          self.in_dim = in_dim
          self.textencoder = nn.Sequential(
               nn.Linear(in_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, out_dim)
          )
          self.metaencoder = nn.Sequential(
               nn.Linear(5, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, out_dim)
          )
          self.fusion = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=out_dim, nhead=4, dim_feedforward=4*out_dim, dropout=dropout, batch_first=True), num_layers=4)      
          self.classifier = nn.Sequential(
               nn.Flatten(),
               nn.Linear(out_dim*3, 2) 
          )
          self.moe = MoE()

     def forward(self, data):
          x = data['review'].x
          meta = x[:data['review'].batch_size, self.in_dim:]
          text = x[:data['review'].batch_size, :self.in_dim]
          
          x_g = self.graphencoder.embed(data)[:data['review'].batch_size].unsqueeze(1)
          x_t = self.textencoder(text).unsqueeze(1)
          x_m = self.metaencoder(meta).unsqueeze(1)
          
          input = torch.cat([x_g, x_t, x_m], dim = 1)
          loss = torch.tensor([0., 0., 0.], device=input.device)
          for i in range(3):
               input[:, i, :], loss[i] = self.moe(input[:, i, :], i)
          output = self.fusion(input)
          # output, loss = self.moe(input)
          return  self.classifier(output), loss.sum()

class SimpleHGN(MessagePassing):
     def __init__(self, in_channels, out_channels, num_edge_type, rel_dim=200, beta=None, final_layer=False):
          super(SimpleHGN, self).__init__(aggr = "add", node_dim=0)
          self.W = torch.nn.Linear(in_channels, out_channels, bias=False)
          self.W_r = torch.nn.Linear(rel_dim, out_channels, bias=False)
          self.a = torch.nn.Linear(3*out_channels, 1, bias=False)
          self.W_res = torch.nn.Linear(in_channels, out_channels, bias=False)
          self.rel_emb = torch.nn.Embedding(num_edge_type, rel_dim)
          self.beta = beta
          self.leaky_relu = torch.nn.LeakyReLU(0.2)
          self.ELU = torch.nn.ELU()
          self.final = final_layer
          
     def init_weight(self):
          for m in self.modules():
               if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight.data)
                         
     def forward(self, x, edge_index, edge_type, pre_alpha=None):
          
          node_emb = self.propagate(x=x, edge_index=edge_index, edge_type=edge_type, pre_alpha=pre_alpha)
          output = node_emb + self.W_res(x)
          output = self.ELU(output)
          if self.final:
               output = F.normalize(output, dim=1)
               
          return output, self.alpha.detach()
          
     def message(self, x_i, x_j, edge_type, pre_alpha, index, ptr, size_i):
          out = self.W(x_j)
          rel_emb = self.rel_emb(edge_type)
          alpha = self.leaky_relu(self.a(torch.cat((self.W(x_i), self.W(x_j), self.W_r(rel_emb)), dim=1)))
          alpha = softmax(alpha, index, ptr, size_i)
          if pre_alpha is not None and self.beta is not None:
               self.alpha = alpha*(1-self.beta) + pre_alpha*(self.beta)
          else:
               self.alpha = alpha
          out = out * alpha.view(-1,1)
          return out

     def update(self, aggr_out):
          return aggr_out

class HGNModel(torch.nn.Module):
     def __init__(self,in_dim=768, hidden_dim=256, out_dim=64, dropout=0.3, beta=0.05):
          super().__init__()

          self.dropout = nn.Dropout(dropout)
          self.linear1 = nn.Sequential(
               nn.Linear(in_dim,hidden_dim),
               nn.ReLU()
          )
          self.conv1 = SimpleHGN(num_edge_type=3, in_channels=hidden_dim, out_channels=hidden_dim, beta=beta)
          self.conv2 = SimpleHGN(num_edge_type=3, in_channels=hidden_dim, out_channels=hidden_dim, beta=beta, final_layer=True)
          self.linear2 = nn.Sequential(
               nn.Linear(hidden_dim,out_dim),
               nn.ReLU()
          )
          self.classifier = nn.Linear(out_dim, 2)

     def forward(self, textual, edge_index, edge_type):
          textual = self.linear_relu1(textual)
          textual, _ = self.text1(textual,edge_index, edge_type)
          textual = F.dropout(textual, p=self.dropout, training=self.training)   
          textual, _ = self.text2(textual,edge_index, edge_type)
          textual = self.linear_relu(textual)
          x = self.classifier(textual)
          return  x
     
     def embed(self, data):
          textual, edge_index, edge_type = data.x, data.edge_index, data.edge_type
          textual = self.linear1(textual)
          textual, _ = self.conv1(textual,edge_index, edge_type)
          textual = self.dropout(textual)   
          textual, _ = self.conv2(textual,edge_index, edge_type)
          textual = self.linear2(textual)
          return textual

class GAT2View_MoE_GT(nn.Module):
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

     def forward(self, data):
          x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

          text = x[:data.batch_size, :self.in_dim] 
          graph = self.graphencoder.embed(data)
          
          x_g = graph[:data.batch_size]
          x_t = self.textencoder(text)
          
          x_g, loss1 = self.moe[0](x_g)         
          x_t, loss2 = self.moe[1](x_t)
          
          loss = loss1 + loss2
          input = torch.cat([x_g.unsqueeze(1), x_t.unsqueeze(1)], dim=1)
          output = self.fusion(input)
          return self.classifier(output), loss

class GAT2View_MoE_GM(nn.Module):
     def __init__(self, in_dim=768, meta_dim=6, hidden_dim=256, out_dim=128, dropout=0.3, num_experts=4, k=2):
          super().__init__()
          self.graphencoder = GATModel(in_dim+meta_dim, hidden_dim, out_dim, 1, dropout)
          self.in_dim = in_dim
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
               nn.Linear(out_dim*2, 2) 
          )
          self.moe = nn.ModuleList([MoE(dim=out_dim, num_experts=num_experts, k=k, dropout=dropout) for _ in range(2)])

     def forward(self, data):
          x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

          meta = x[:data.batch_size, self.in_dim:]
          graph = self.graphencoder.embed(data)
          
          x_g = graph[:data.batch_size]
          x_m = self.metaencoder(meta)
          
          x_g, loss1 = self.moe[0](x_g)    
          x_m, loss2 = self.moe[1](x_m)
          
          loss = loss1 + loss2
          input = torch.cat([x_g.unsqueeze(1), x_m.unsqueeze(1)], dim=1)
          output = self.fusion(input)
          return self.classifier(output), loss

class GAT2View_MoE_MT(nn.Module):
     def __init__(self, in_dim=768, meta_dim=6, hidden_dim=256, out_dim=128, dropout=0.3, num_experts=4, k=2):
          super().__init__()
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
               nn.Linear(out_dim*2, 2) 
          )
          self.moe = nn.ModuleList([MoE(dim=out_dim, num_experts=num_experts, k=k, dropout=dropout) for _ in range(2)])

     def forward(self, data):
          x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

          meta = x[:data.batch_size, self.in_dim:]
          text = x[:data.batch_size, :self.in_dim] 
          
          x_t = self.textencoder(text)
          x_m = self.metaencoder(meta)
          
          x_t, loss1 = self.moe[0](x_t)
          x_m, loss2 = self.moe[1](x_m)
          
          loss = loss1 + loss2
          input = torch.cat([x_t.unsqueeze(1), x_m.unsqueeze(1)], dim=1)
          output = self.fusion(input)
          return self.classifier(output), loss
