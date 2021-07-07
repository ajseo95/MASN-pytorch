# --------------------------------------------------------
# This code is modified from Jumpin2's repository.
# https://github.com/Jumpin2/HGA
# --------------------------------------------------------

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.layer_norm = nn.LayerNorm(out_features, elementwise_affine=False)

    def forward(self, input, adj):
        # self.weight of shape (hidden_size, hidden_size)
        support = self.weight(input)
        output = torch.bmm(adj, support)
        output = self.layer_norm(output)
        return output


class GCN(nn.Module):

    def __init__(
            self, input_size, hidden_size, num_classes, num_layers=1,
            dropout=0.1):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolution(input_size, hidden_size))
        for i in range(num_layers - 1):
            self.layers.append(GraphConvolution(hidden_size, hidden_size))
        self.layers.append(GraphConvolution(hidden_size, num_classes))

        self.layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        x_gcn = [x]
        for i, layer in enumerate(self.layers):
            x_gcn.append(self.dropout(F.relu(layer(x_gcn[i], adj))))

        x = self.layernorm(x + x_gcn[-1])
        return x


class VideoAdjLearner(Module):

    def __init__(self, in_feature_dim, hidden_size, dropout=0.1, scale=100):
        super().__init__()
        self.scale = scale

        self.edge_layer_1 = nn.Linear(in_feature_dim, hidden_size, bias=False)
        self.edge_layer_2 = nn.Linear(hidden_size, hidden_size, bias=False)

        # Regularization
        self.dropout = nn.Dropout(p=dropout)
        self.edge_layer_1 = nn.utils.weight_norm(self.edge_layer_1)
        self.edge_layer_2 = nn.utils.weight_norm(self.edge_layer_2)

    def forward(self, v, v_mask=None):
        '''
        :param v: (b, v_len, d)
        :param v_mask: (b, v_len)
        :return: adj: (b, v_len, v_len)
        '''
        # layer 1
        h = self.edge_layer_1(v)  # b, v_l, d
        h = F.relu(h)

        # layer 2
        h = self.edge_layer_2(h)  # b, v_l, d
        h = F.relu(h)

        # outer product
        adj = torch.bmm(h, h.transpose(1, 2))  # b, v_l, v_l

        if v_mask is not None:
            adj_mask = adj.data.new(*adj.size()).fill_(1)
            v_mask_ = torch.matmul(v_mask.unsqueeze(2), v_mask.unsqueeze(2).transpose(1, 2))
            adj_mask = adj_mask * v_mask_

            adj_mask = Variable(adj_mask)
            adj = adj - 1e10*(1 - adj_mask)

            adj = F.softmax(adj * self.scale, dim=-1)
            adj = adj * adj_mask
            adj = adj.masked_fill(adj != adj, 0.)
        else:
            adj = F.softmax(adj * self.scale, dim=-1)

        return adj

