

#####################################################################
## This is the module to build the GNN model
#####################################################################

import math
import os.path as osp
import time
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist

from sklearn.metrics import roc_auc_score
from torch.nn import BCEWithLogitsLoss, Conv1d, MaxPool1d, ModuleList
import torch.nn as nn
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, GCNConv, GATConv, SAGEConv, SortAggregation, PMLP, EdgeConv, GIN
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix, degree
from sklearn.utils import class_weight
from .torchcpd import DeformableRegistration
from torch_scatter import scatter_std,scatter_mean
import networkx as nx
import random
import numpy as np
from typing import List
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
import networkx as nx
import torch
import os
from .parameters import method, train_val
import dgl

from dgl.nn import EGATConv #https://www.dgl.ai/dgl_docs/generated/dgl.nn.pytorch.conv.EGATConv.html#dgl.nn.pytorch.conv.EGATConv



def deformable_registration(X,Y, tolerance, beta, device):
    reg_deformable = DeformableRegistration(X=X, Y=Y, max_iterations=500, tolerance = tolerance, beta = beta, device = device)
    TY, _ = reg_deformable.register()
    return TY
    

class DGCNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, num_features, GNN, conv1d_channels, use_edge_attr, k):
        super().__init__()

        self.convs = ModuleList()
        
        self.convs.append(GNN(num_features, hidden_channels))
        for i in range(0, num_layers - 1):

            self.convs.append(GNN(hidden_channels, hidden_channels))

        self.convs.append(GNN(hidden_channels, 1))


        total_latent_dim = hidden_channels * num_layers + 1
       
        conv1d_kws = [total_latent_dim, 1] 

        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0] )
        
        self.use_edge_attr = use_edge_attr
       
        self.k = k
        
    def forward(self, x, edge_index, edge_attr):
        xs = [x]
        
        
        for conv in self.convs:
            # xs += [conv(xs[-1], edge_index).tanh()]
            if self.use_edge_attr :
                xs += [conv(xs[-1], edge_index, edge_attr)]
            elif not self.use_edge_attr :
                xs += [conv(xs[-1], edge_index)]
           

        x = torch.cat(xs[self.k:], dim=-1)

        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        # print("x before conv1", x.shape)
        x = self.conv1(x)
        # print("x after conv1", x.shape)
        return x[:,:,0]









class DGCNNedge(nn.Module):
    def __init__(self, hidden_channels, num_layers, num_features, edge_feat_dim, conv1d_channels, k, num_heads=5):
        super().__init__()
        
        self.convs = ModuleList()
        self.num_heads = num_heads
        self.k = k
        if num_heads ==1:
            total_latent_dim = hidden_channels * num_layers + 1
        else:
            total_latent_dim = int(hidden_channels *  ((num_heads**(num_layers+1)-num_heads)/(num_heads-1) + 1) + 2)
       

        # First layer
        self.convs.append(
            
            
            EGATConv(
            in_node_feats=num_features,
            in_edge_feats=edge_feat_dim,
            out_node_feats=edge_feat_dim,
            out_edge_feats=edge_feat_dim,  # usually set to same as input edge feat dim or customized
            num_heads=num_heads
            )
            
        )

        

        for _ in range(num_layers - 1):

            hidden_channels = hidden_channels * num_heads

            self.convs.append(             
                EGATConv(
                in_node_feats=int(4*num_heads),
                # in_edge_feats=int(4*num_heads),
                in_edge_feats=int(4),
                out_node_feats=edge_feat_dim,
                out_edge_feats=edge_feat_dim,  # usually set to same as input edge feat dim or customized
                num_heads=num_heads
                )
            )

        hidden_channels = hidden_channels * num_heads

        self.convs.append(
            
            EGATConv(
            in_node_feats=int(4*num_heads),
            # in_edge_feats=int(4*num_heads),
            in_edge_feats=int(4),
            out_node_feats=edge_feat_dim,
            out_edge_feats=edge_feat_dim,  # usually set to same as input edge feat dim or customized
            num_heads=num_heads
            )
            
        )

        
        if self.k ==0:
            # total_latent_dim = int(4*num_heads)*6+10
            total_latent_dim = 31 + 20*(num_heads-1)
        if self.k ==1:
            total_latent_dim = int(4*num_heads)*6

        total_latent_dim = 203
        self.conv1 = Conv1d(1, conv1d_channels[0], total_latent_dim, total_latent_dim)
        # print("self.conv1 = Conv1d(1, conv1d_channels[0], total_latent_dim, total_latent_dim)", 1, conv1d_channels[0], total_latent_dim, total_latent_dim)


    def add_self_loops_with_edge_attr(self, g, edge_attr, edge_feat_dim):
        g = dgl.add_self_loop(g)
        num_new_edges = g.num_edges() - edge_attr.shape[0]
        new_attrs = torch.zeros(num_new_edges, edge_feat_dim, device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, new_attrs], dim=0)
        return g, edge_attr

    
    def forward(self, x, edge_index, edge_attr):

        u, v = edge_index
        g = dgl.graph((v, u), num_nodes=x.shape[0])
        num_nodes = x.shape[0]
       

        xs = [x]
        x_nodes, x = self.convs[0](g, x, edge_attr)  # get node output, shape [N, H, D]
        x = scatter_mean(x.flatten(1), u, dim=0, dim_size=num_nodes) + x_nodes.flatten(1)
        x = x.flatten(1)              # -> [N, H * D]
        xs.append(x)

        

        for conv in self.convs[1:]:


            x_nodes, x_new = conv(g, x, edge_attr)


            x_new = scatter_mean(x_new.flatten(1), u, dim=0, dim_size=num_nodes) + x_nodes.flatten(1)

            x = x + x_new  # True residual update

            xs.append(x)

        x = torch.cat(xs[self.k:], dim=-1)  # [N, total_latent_dim]

        x = x.unsqueeze(1)                  # [N, 1, latent_dim]
        # print("x before conv",x.shape)
        x = self.conv1(x)                   # Conv1D layer
        # print("x after conv",x.shape)
        return x[:, :, 0]                   # [N, out_dim]







def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)
        
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
        

class SimilarityScore(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimilarityScore, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)  # Adjusted input_dim to 32
        self.fc2 = nn.Linear(input_dim, output_dim)
        self.flatten = nn.Flatten(start_dim=1)  # Flatten from the second dimension onward
    
    def forward(self, in1, in2):
        embed1 = self.fc1(in1)  # Shape: [570, 32] -> [570, output_dim]
        embed2 = self.fc2(in2)  # Shape: [570, 32] -> [570, output_dim]
        flat1 = self.flatten(embed1)   # Shape: [570, output_dim]
        flat2 = self.flatten(embed2)   # Shape: [570, output_dim]
        

        # dot_product = torch.sum(flat1 * flat2, dim=-1)  # Shape: [570], batch-wise dot product
        # return dot_product     
        
        product_AB = torch.sum(flat1 * flat2, dim=1)  # Shape: [615]
        book_A_length = torch.linalg.norm(flat1, dim=1)  # Shape: [615]
        book_B_length = torch.linalg.norm(flat2, dim=1)  # Shape: [615]
        cosine_similarity = product_AB / (book_A_length * book_B_length)  # Shape: [615]
        return cosine_similarity
    

class MessagePassing(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(MessagePassing, self).__init__()
        self.update_fn = nn.Linear(input_dim, output_dim)
        self.aggregate_fn = nn.Linear(input_dim, output_dim)
        self.device = device


    def forward(self, x, edge_index):
        adj_matrix = torch.zeros(len(x), len(x), dtype=torch.float32).to(self.device)
        adj_matrix[edge_index[0], edge_index[1]] = 1
        adj_matrix[edge_index[1], edge_index[0]] = 1

        messages = torch.matmul(adj_matrix, x)  # Aggregating messages from neighbors
        aggregated = self.aggregate_fn(messages)  # Applying the aggregate function
        updated = self.update_fn(x) + aggregated  # Updating the node embeddings
        return updated
        



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.m = nn.Conv1d(64, 128, 1, stride=1)
        self.m2 = nn.Conv1d(128, 16, 1, stride=1)
        self.m3 = nn.Conv1d(16, 2, 1, stride=1)

    def forward(self, sim):
        sim = self.m(sim)
        sim = self.m2(sim)
        sim = self.m3(sim)
        return sim



class MLPx(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLPx, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, 64)
        # self.bn1 = nn.BatchNorm1d(64) 
        self.fc2 = torch.nn.Linear(64, out_channels)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.fc2(x)
        return x


class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(64, 128)  
        # self.bn1 = nn.BatchNorm1d(128) 
        self.fc2 = nn.Linear(128, 16)  
        # self.bn2 = nn.BatchNorm1d(16) 
        self.fc3 = nn.Linear(16, 2)    

    def forward(self, sim):
        # If the input has extra dimensions, flatten them before passing through Linear layers
        sim = sim.squeeze(-1)  
        
        sim = self.fc1(sim)    
        # sim = self.bn1(sim)
        sim = self.fc2(sim)
        # sim = self.bn2(sim)
        sim = self.fc3(sim)    

        return sim
    



class NodeLevelGNN(torch.nn.Module):
    def __init__(self,nearby_search_num, norm_scale, with_AM, device):
        super(NodeLevelGNN, self).__init__()
        # self.encoder = DGCNN(hidden_channels=64, num_layers=3, num_features=10, GNN=GATConv, conv1d_channels = [16,32],  use_edge_attr = True, k=0)
        # self.encoder2 = DGCNN(hidden_channels=8, num_layers=5, num_features=10, GNN=GCNConv, conv1d_channels = [16,64],  use_edge_attr = True, k=0)

       
        out_dim1 = 11
        out_dim2 = 11
        self.mp1 = MessagePassing(11, out_dim1, device)
        self.mp2 = MessagePassing(11, out_dim2, device)
        embedding_size = 11
        # self.deepwalk = DeepWalk_deterministic(window_size=5, embedding_size=embedding_size, walk_length=10, walks_per_node=20) ##walks_per_node can't be 1
        # self.deepwalk = DeepWalk(window_size=5, embedding_size=embedding_size, walk_length=10, walks_per_node=20) ##walks_per_node can't be 1
        self.mp3 = MessagePassing(embedding_size, out_dim2, device)


        self.gin = GIN(in_channels=11,
                       hidden_channels=64,
                       num_layers=3,  # Number of layers
                       out_channels=16,
                       dropout=0,  # Dropout for regularization
                       act='relu',  # Activation function
                       norm=None).to(device)  # No normalization in this example
        

        
        self.encoder = DGCNN(hidden_channels=64, num_layers=5, num_features=out_dim1, GNN=GATConv, conv1d_channels = [16,32],  use_edge_attr = False, k=0)
        # self.encoder2 = DGCNN(hidden_channels=8, num_layers=3, num_features=out_dim2, GNN=GATConv, conv1d_channels = [16,64],  use_edge_attr = True, k=0)
        self.encoder2 =  DGCNNedge(hidden_channels=8, num_layers=5, num_features=out_dim1, edge_feat_dim=4, conv1d_channels = [16,64], k=0, num_heads=8)
        # self.encoder3 = DGCNN(hidden_channels=2, num_layers=1, num_features=16, GNN=GATConv, conv1d_channels = [16,64],  use_edge_attr = True, k=0)
        # self.encoder3 =  DGCNNedge(hidden_channels=2, num_layers=1, num_features=16, edge_feat_dim=4, conv1d_channels = [16,64], k=0, num_heads=1)


        self.sageconv1 = SAGEConv(11,16)
        self.sageconv2 = SAGEConv(16,16)

        self.mlpx = MLPx(in_channels= 22, out_channels=16)
        self.edge_conv = EdgeConv(self.mlpx, aggr='max')       
     
        
        # self.bilinear_layer = nn.Bilinear(64, 64, 64) ## for bilinear layer not stable, loss explodes
        self.mlp = MLP([1, 64, 1], dropout=1e-10, norm=None)
        # self.mlp2 = MLP([1, 80, 1], dropout=1e-10, norm=None)


        # self.m = ConvNet()
        self.m = LinearNet()

        self.with_AM = with_AM
        self.device = device
        self.nearby_search_num = nearby_search_num
        self.norm_scale = norm_scale
    

    def each_graph(self, x, edge_index, edge_attr):
        
        
        emb = self.encoder(x, edge_index, edge_attr)

        emb2 = self.encoder2(x, edge_index, edge_attr)


        x1 = self.mp1(x, edge_index) 
        emb3 = self.sageconv1(x1, edge_index)

        emb4 = self.edge_conv(x, edge_index)


        return torch.cat(( emb4, emb3, emb2, emb),dim = 1)

  
    
    
    def get_ground_truth(self,data1,data2):
        eff_label = (data1.y.unsqueeze(1) + data2.y.unsqueeze(0)) > 0  ## to exclude label=0
        AM = ((data1.y.unsqueeze(1) - data2.y.unsqueeze(0)) == 0)*(eff_label)*1
        return AM


    
    def search_nearby_indices(self,X,Y,nearby_search_num, tolerance, beta, norm_scale, device):
        norm_scale = torch.from_numpy(norm_scale).to(device)
        TY_normalized = deformable_registration(X/norm_scale ,Y/norm_scale, tolerance, beta, device)
        TY = (TY_normalized * norm_scale).to(torch.float32)

        distance_matrix = torch.cdist(X, TY)

        # distance_matrix = torch.cdist(X * torch.tensor([5,1,1]), TY* torch.tensor([5,1,1])) ### running test

        nearest_indices = torch.argsort(distance_matrix, dim=1)[:, :nearby_search_num]
        return nearest_indices, distance_matrix



    def get_AM_mask(self,data1,data2,with_AM, method):
 
        # Method 1: from nearby search
        if method == 1:
            coord = data1.x[:,0:3]
            coords2 = data2.x[:,0:3]
            distance_matrix = torch.cdist(coord,coords2)
            nearest_indices = torch.argsort(distance_matrix, dim=1)[:, :self.nearby_search_num]



        ## Method 2: from deformation registration
        if method == 2:
            [tolerance, beta] = [1e-5, 0.5]

            
            coord = data1.x[:,0:3][:,np.array([2,0,1])]
            coords2 = data2.x[:,0:3][:,np.array([2,0,1])]
            nearest_indices, distance_matrix = self.search_nearby_indices(coord,coords2,self.nearby_search_num, tolerance, beta, self.norm_scale, self.device)


        
            # coord = data1.x[:,0:3][:,np.array([2,0,1])][:,1:3]
            # coords2 = data2.x[:,0:3][:,np.array([2,0,1])][:,1:3]
            # nearest_indices, distance_matrix = self.search_nearby_indices(coord,coords2,self.nearby_search_num, tolerance, beta, self.norm_scale[1:], self.device)

        
        ## Method 2: deformation registration on the head area and apply the transformation to the whole
        # if method ==3:
        #     # print("method 3 is used")
        #     [tolerance, beta] = [1e-5, 1]

        #     X = data1.x[:,0:3][:,np.array([2,0,1])].to(self.device) 
        #     Y = data2.x[:,0:3][:,np.array([2,0,1])].to(self.device)
        #     ind1 = torch.from_numpy(self.mask1[tuple(X[:, 1:3].T.int().cpu().numpy())]).to(torch.bool)
        #     ind2 = torch.from_numpy(self.mask2[tuple(Y[:, 1:3].T.int().cpu().numpy())]).to(torch.bool)
    
        #     norm_scale = np.array([ 23 * 5, 512, 512])  - 1
        #     norm_scale = torch.from_numpy(norm_scale).to(self.device)
        #     reg = DeformableRegistration(X=X[ind1] / norm_scale, Y=Y[ind2] / norm_scale, beta=beta, tolerance=tolerance, device=self.device)
        #     maskY_normalized, a = reg.register()
            
        #     Y_norm = (Y/norm_scale).type(torch.float)  
        #     TY_normalized = reg.transform_point_cloud(Y_norm)
        #     TY = (TY_normalized * norm_scale).to(torch.float32)    
        #     distance_matrix = torch.cdist(X, TY)
        #     nearest_indices = torch.argsort(distance_matrix, dim=1)[:, :self.nearby_search_num]

        

        
        mask = torch.zeros_like(distance_matrix, dtype=bool).to(self.device)
        for row_idx, col_indices in enumerate(nearest_indices):
            mask[row_idx, col_indices] = 1

        if with_AM:
            AM = self.get_ground_truth(data1,data2)
            AM_mask = ((mask) + 1*(AM>0))>0
        else:
            AM_mask = mask>0

        
        return AM_mask
    

    def norm_nodes_features(self, x):
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
        x = torch.clamp(x, -1, 1)  # Optional
        return x

    def cat_degree(self, data):
        deg_out = degree(data.edge_index[0], num_nodes=data.num_nodes)
        deg_in = degree(data.edge_index[1], num_nodes=data.num_nodes)
        deg_total = deg_out + deg_in
        deg_total = deg_total.view(-1, 1)  # shape [num_nodes, 1]
        x = torch.cat([data.x[:,0:10], deg_total.to(data.x.device)], dim=1)
        return self.norm_nodes_features(x)
        
    
    def normalize_features(self, edge_attr):
        edge_mean = edge_attr.mean(dim=0)
        edge_std = edge_attr.std(dim=0)
        normalized_edge_feat = (edge_attr - edge_mean) / (edge_std + 1e-6)
        return normalized_edge_feat

    def forward(self,data1,data2):

        pred1 = self.each_graph(self.cat_degree(data1), data1.edge_index, data1.edge_attr[:,0:4]) 
        pred2 = self.each_graph(self.cat_degree(data2), data2.edge_index, data2.edge_attr[:,0:4]) 

        
        
        # AM_mask1 = self.get_AM_mask(data1,data2,self.with_AM,1)
        # # AM_mask2 = self.get_AM_mask(data2,data1,self.with_AM,1)
        # # AM_mask0 = AM_mask1 * AM_mask2.T

        
        AM_mask1 = self.get_AM_mask(data1,data2,self.with_AM,method)
        AM_mask2 = self.get_AM_mask(data2,data1,self.with_AM,method)
        AM_mask = AM_mask1 * AM_mask2.T

        
        # # AM_mask = (AM_mask0 + AM_mask00) >0
        # AM_mask = (AM_mask1 + AM_mask2) > 0

        
        # if train_val != 'train': 
        # AM_mask = self.get_AM_mask(data1,data2,self.with_AM,2)
        AM_mask[data1.y==0] = 0   ## when in evaluation, use this
    
        # AM_mask[data1.y==0][:, data2.y==0] = 0  ## when in trainining, use this data of change: 11-03-24
        
        

        
 
        potential_edge = torch.transpose(torch.stack(torch.where(AM_mask>0)),1,0)

        
        embedding1 = pred1[potential_edge[:,0]]
        embedding2 = pred2[potential_edge[:,1]]
    
    
        # diff = torch.abs(embedding1 - embedding2)
        diff = embedding1 - embedding2
        sim = self.mlp(diff[:,:,None])

        sim = self.m(sim) 
        return sim

    
    def normalize_features(self, edge_attr):
        edge_mean = edge_attr.mean(dim=0)
        edge_std = edge_attr.std(dim=0)
        normalized_edge_feat = (edge_attr - edge_mean) / (edge_std + 1e-6)
        return normalized_edge_feat

    
    def forward_score_matrix(self,data1,data2, AM_mask, threshold):

        pred1 = self.each_graph(self.cat_degree(data1), data1.edge_index, data1.edge_attr[:,0:4]) 
        pred2 = self.each_graph(self.cat_degree(data2), data2.edge_index, data2.edge_attr[:,0:4]) 

 
        potential_edge = torch.transpose(torch.stack(torch.where(AM_mask>0)),1,0)

        
        embedding1 = pred1[potential_edge[:,0]]
        embedding2 = pred2[potential_edge[:,1]]
    
        diff = embedding1 - embedding2

        sim = self.mlp(diff[:,:,None])
        sim = self.m(sim) 
        score = F.softmax(sim,dim = -1)[:,1]

        score_matrix = torch.zeros_like(AM_mask, dtype=torch.float32).to(self.device)
        score_matrix[potential_edge[:, 0], potential_edge[:, 1]] = score
        score_matrix = score_matrix  > threshold  
        
        return score_matrix

    def unique_col_rows(self, score_matrix):
        rows_to_zero = score_matrix.sum(dim=1) > 1
        cols_to_zero = score_matrix.sum(dim=0) > 1
        score_matrix[rows_to_zero] = 0
        score_matrix[:, cols_to_zero] = 0  
        return score_matrix


    def get_mutual_AM_mask(self,data1,data2,with_AM, method):
        AM_mask1 = self.get_AM_mask(data1,data2,with_AM,method)
        AM_mask2 = self.get_AM_mask(data2,data1,with_AM,method)
        AM_mask = AM_mask1 * AM_mask2.T
        return AM_mask


    

    def forward_link_back(self,data1,data2, threshold , use_nms):
        AM_mask12 = self.get_AM_mask(data1,data2,self.with_AM,method)
        if use_nms:
            score_matrix12 = self.forward_score_matrix_nms(data1,data2, AM_mask12, threshold)
        else:
            score_matrix12 = self.forward_score_matrix(data1,data2, AM_mask12, threshold)
        score_matrix12 = self.unique_col_rows(score_matrix12)

        
        AM_mask21 = self.get_AM_mask(data2,data1,self.with_AM,method)
        if use_nms:
            score_matrix21 = self.forward_score_matrix_nms(data2,data1, AM_mask21, threshold)
        else:
            score_matrix21 = self.forward_score_matrix(data2,data1, AM_mask21, threshold)
        score_matrix21 = self.unique_col_rows(score_matrix21)
        
        score_matrix = score_matrix12 * score_matrix21.T
        score_matrix = self.unique_col_rows(score_matrix)

        AM_mask = self.get_mutual_AM_mask(data1,data2,self.with_AM,method)
        AM_mask[data1.y==0] = 0
        score_matrix = score_matrix * AM_mask
        



        

        potential_edge = torch.transpose(torch.stack(torch.where(AM_mask>0)),1,0)
        score = score_matrix[potential_edge[:, 0], potential_edge[:, 1]]
        return score, score_matrix
        

    

    def forward_high_precision(self,data1,data2, score_matrix, threshold):
        '''
        score_matrix: the score matrix from the forward_score_matrix function under the model with the best recall performance
        '''
        AM_mask12 = score_matrix > 0
        score_matrix12 = self.forward_score_matrix(data1,data2, AM_mask12, threshold)
        score_matrix12 = self.unique_col_rows(score_matrix12)
        AM_mask21 = score_matrix.T > 0
        score_matrix21 = self.forward_score_matrix(data2,data1, AM_mask21, threshold)
        score_matrix21 = self.unique_col_rows(score_matrix21)
        score_matrix = score_matrix12 * score_matrix21.T
        score_matrix = self.unique_col_rows(score_matrix)


        AM_mask = self.get_mutual_AM_mask(data1,data2,self.with_AM,2)
        AM_mask[data1.y==0] = 0

        
        
        score_matrix = score_matrix * AM_mask

        potential_edge = torch.transpose(torch.stack(torch.where(AM_mask>0)),1,0)
        score = score_matrix[potential_edge[:, 0], potential_edge[:, 1]]
        return score, score_matrix

    



    def row_wise_non_maximum_suppression(self, pred_matrix, threshold=0.5):
        max_vals, max_indices = torch.max(pred_matrix, dim=1)
        result_matrix = torch.zeros_like(pred_matrix, dtype=torch.int)
        mask = max_vals >= threshold
        result_matrix[torch.arange(pred_matrix.shape[0]), max_indices] = mask.int()
        return result_matrix



    def forward_score_matrix_nms(self,data1,data2, AM_mask, threshold):

        pred1 = self.each_graph(self.cat_degree(data1), data1.edge_index, data1.edge_attr[:,0:4])
        pred2 = self.each_graph(self.cat_degree(data2), data2.edge_index, data2.edge_attr[:,0:4])


        potential_edge = torch.transpose(torch.stack(torch.where(AM_mask>0)),1,0)

        
        embedding1 = pred1[potential_edge[:,0]]
        embedding2 = pred2[potential_edge[:,1]]
    
    

        diff = embedding1 - embedding2

        sim = self.mlp(diff[:,:,None])
        sim = self.m(sim) 



        score = F.softmax(sim,dim = -1)
        

        score_matrix = torch.zeros_like(AM_mask, dtype=torch.float32).to(self.device)
        score_matrix[potential_edge[:, 0], potential_edge[:, 1]] = score[:,1]
        result_matrix = self.row_wise_non_maximum_suppression(score_matrix, threshold=0.5)
        nms_index = result_matrix[potential_edge[:, 0], potential_edge[:, 1]]

        # sim = sim  *  nms_index[:,None]
        
        score_matrix = score_matrix * result_matrix

        return score_matrix > threshold
       





                  
        
                    
def count_toal_num_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("total_params: ",total_params,"trainable_params:",trainable_params)       
                    
    
    
    
    
    
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):

        # probs = F.softmax(inputs, dim=-1)  # Shape: [batch_size, 2]
        probs = F.log_softmax(inputs, dim = -1)
        class_probs = probs.gather(1, targets.view(-1, 1))  # Shape: [batch_size, 1]

        focal_weight = ( 1 - class_probs) ** self.gamma

        log_probs = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        

        loss = focal_weight * log_probs


        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss




    
    
