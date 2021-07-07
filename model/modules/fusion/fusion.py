# --------------------------------------------------------
# This code is modified from cuiyuhao1996's repository.
# Licensed under The MIT License [see LICENSE for details]
# https://github.com/cuiyuhao1996/mcan-vqa
# --------------------------------------------------------

from model.modules.fusion.net_utils import FC, MLP, LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math


class AttFlat(nn.Module):
    def __init__(self, hidden_size, flat_mlp_size, flat_glimpses, flat_out_size, dropout_r=0.1):
        super(AttFlat, self).__init__()

        self.flat_glimpses = flat_glimpses

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=flat_mlp_size,
            out_size=flat_glimpses,
            dropout_r=dropout_r,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            hidden_size * flat_glimpses,
            flat_out_size
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)  # b, L, glimpse
        if x_mask is not None:
            x_mask = x_mask.unsqueeze(2)
            att = att - 1e10*(1 - x_mask)
        att = F.softmax(att, dim=1)
        if x_mask is not None:
            att = att * x_mask
        att = att.masked_fill(att != att, 0.)  # b, L, glimpse

        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)  # b, d
        x_atted = self.linear_merge(x_atted)

        return x_atted


class FFN(nn.Module):
    def __init__(self, hidden_size, ff_size, dropout_r=0.1):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=ff_size,
            out_size=hidden_size,
            dropout_r=dropout_r,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


class AttAdj(nn.Module):
    def __init__(self, hidden_size, dropout_r=0.1):
        super(AttAdj, self).__init__()
        self.hidden_size = hidden_size
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_q = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_r)

    def forward(self, k, q, mask=None):
        '''
        :param k: b, kv_l, d
        :param q: b, q_l, d
        :param mask: b, q_l, kv_l
        '''

        k = self.linear_k(k)
        q = self.linear_q(q)

        adj_scores = self.att(k, q, mask)

        return adj_scores

    def att(self, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores - 1e10*(1 - mask)

        att_map = F.softmax(scores, dim=-1)
        if mask is not None:
            att_map = att_map * mask
        att_map = att_map.masked_fill(att_map != att_map, 0.)
        att_map = self.dropout(att_map)

        return att_map


class MotionApprFusion(nn.Module):
    def __init__(self, hidden_size, ff_size, n_layer=1, dropout_r=0.1):
        super(MotionApprFusion, self).__init__()
        self.n_layer = n_layer
        self.hidden_size = hidden_size

        self.appr_att_score = AttAdj(hidden_size, dropout_r)
        self.motion_att_score = AttAdj(hidden_size, dropout_r)
        self.all_att_score = AttAdj(hidden_size, dropout_r)

        self.appr_linear = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layer)])
        self.motion_linear = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layer)])
        self.all_linear = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layer)])

        self.appr_norm = LayerNorm(hidden_size)
        self.motion_norm = LayerNorm(hidden_size)
        self.all_norm = LayerNorm(hidden_size)

        self.ques_guide_att = AttAdj(hidden_size, dropout_r)

        self.ffn = FFN(hidden_size, ff_size, dropout_r)
        self.dropout = nn.Dropout(dropout_r)
        self.norm = LayerNorm(hidden_size)

    def forward(self, U, q_hid, U_mask=None):
        '''
        :param U: b, 2L, d
        :param q_hid: b, d
        :param U_mask: b, 2L
        :return:
        '''
        residual = U
        bsz, seq_len = U.size(0), U.size(1)
        qword_len = int(seq_len/2)
        device = U.device

        appr_mask = torch.zeros(bsz, seq_len, seq_len).to(torch.float).to(device)
        appr_mask[:,:,:qword_len] = 1
        motion_mask = torch.zeros(bsz, seq_len, seq_len).to(torch.float).to(device)
        motion_mask[:,:,qword_len:] = 1

        if U_mask is not None:
            appr_mask = appr_mask * U_mask
            motion_mask = motion_mask * U_mask

        appr_att_adj = self.appr_att_score(U, U, appr_mask)  # b, 2L, 2L
        motion_att_adj = self.motion_att_score(U, U, motion_mask)
        if U_mask is not None:
            all_att_adj = self.all_att_score(U, U, U_mask)
        else:
            all_att_adj = self.all_att_score(U, U, None)

        appr_inp, motion_inp, all_inp = [U, ], [U, ], [U, ]
        for i in range(self.n_layer):
            appr_x = self.appr_linear[i](appr_inp[i])  # b, 2L, d
            appr_x = torch.matmul(appr_att_adj, appr_x)
            appr_inp.append(appr_x)

            motion_x = self.motion_linear[i](motion_inp[i])
            motion_x = torch.matmul(motion_att_adj, motion_x)
            motion_inp.append(motion_x)

            all_x = self.all_linear[i](all_inp[i])
            all_x = torch.matmul(all_att_adj, all_x)
            all_inp.append(all_x)

        appr_x = self.appr_norm(residual + appr_x)
        motion_x = self.motion_norm(residual + motion_x)
        all_x = self.all_norm(residual + all_x)

        graph_out = torch.cat([appr_x.unsqueeze(1), motion_x.unsqueeze(1), all_x.unsqueeze(1)], dim=1)  # b, 3, 2L, d
        fusion_k = torch.sum(graph_out, dim=2)  # b, 3, d

        fusion_q = q_hid.unsqueeze(1)  # b, 1, d
        fusion_att_score = self.ques_guide_att(fusion_k, fusion_q).squeeze()  # b, 3

        fusion_att = graph_out * fusion_att_score.unsqueeze(2).unsqueeze(3)  # b, 3, 2L, d
        fusion_att = torch.sum(fusion_att, dim=1)  # b, 2L, d

        fusion_out = self.norm(fusion_att + self.ffn(self.dropout(fusion_att)))

        return fusion_out

