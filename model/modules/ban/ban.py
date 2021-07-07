# --------------------------------------------------------
# This code is modified from jnhwkim's repository.
# https://github.com/jnhwkim/ban-vqa
# --------------------------------------------------------

from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from model.modules.ban.fc import FCNet


class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network
    """
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2,.5], k=3):
        super(BCNet, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim; self.q_dim = q_dim
        self.h_dim = h_dim; self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1]) # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if None == h_out:
            pass
        elif h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, v, q):
        if None == self.h_out:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            logits = torch.einsum('bvk,bqk->bvqk', (v_, q_))
            return logits

        # low-rank bilinear pooling using einsum
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v))
            q_ = self.q_net(q)
            logits = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
            return logits # b x h_out x v x q

        # batch outer product, linear projection
        # memory efficient but slow computation
        else:
            v_ = self.dropout(self.v_net(v)).transpose(1,2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1,2).unsqueeze(2)
            d_ = torch.matmul(v_, q_) # b x h_dim x v x q
            logits = self.h_net(d_.transpose(1,2).transpose(2,3)) # b x v x q x h_out
            return logits.transpose(2,3).transpose(1,2) # b x h_out x v x q

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v) # b x v x d
        q_ = self.q_net(q) # b x q x d
        logits = torch.einsum('bvk,bvq,bqk->bk', (v_, w, q_))
        if 1 < self.k:
            logits = logits.unsqueeze(1) # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k # sum-pooling
        return logits


class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2,.5]):
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3), \
                                  name='h_mat', dim=None)

    def forward(self, v, q, v_mask, q_mask):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        p, logits = self.forward_all(v, q, v_mask, q_mask)
        return p, logits

    def forward_all(self, v, q, v_mask, q_mask, logit=False, mask_with=-float('inf')):
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v,q) # b x g x v x q

        if v_mask is not None:
            v_mask = v_mask.unsqueeze(1).unsqueeze(3).expand_as(logits)
            logits = logits - 1e10*(1 - v_mask)
        if q_mask is not None:
            q_mask = q_mask.unsqueeze(1).unsqueeze(2).expand_as(logits)
            logits = logits - 1e10*(1 - q_mask)

        if not logit:
            p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
            p = p.view(-1, self.glimpse, v_num, q_num)
            if v_mask is not None:
                p = p * v_mask
            if q_mask is not None:
                p = p * q_mask
            p = p.masked_fill(p != p, 0.)

            return p, logits

        return logits


class BAN(nn.Module):
    def __init__(self, num_hid, glimpse):
        super(BAN, self).__init__()
        self.glimpse = glimpse
        self.v_att = BiAttention(num_hid, num_hid, num_hid, glimpse)

        b_net = []
        q_prj = []
        for i in range(glimpse):
            b_net.append(BCNet(num_hid, num_hid, num_hid, None, k=1))
            q_prj.append(FCNet([num_hid, num_hid], '', .2))

        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()

    def forward(self, v, q_emb, v_mask, q_mask):
        """Forward
        v: [batch, num_objs, dim]
        q: [batch_size, q_len, dim]
        v_mask: b, v_len
        q_mask: b, q_len
        return: logits, not probs
        """
        # v_emb = v
        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v, q_emb, v_mask, q_mask) # b x g x v x q

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, att[:,g,:,:]) # b x l x h
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb

        return q_emb, att
