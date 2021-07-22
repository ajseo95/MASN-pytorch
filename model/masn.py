# --------------------------------------------------------
# This code is modified from Jumpin2's repository.
# https://github.com/Jumpin2/HGA
# --------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from model.modules import linear_weightdrop as dropnn

from torch.autograd import Variable

from model.modules.rnn_encoder import SentenceEncoderRNN
from model.modules.gcn import VideoAdjLearner, GCN
from model.modules.position_embedding import PositionEncoding
from model.modules.ban.ban import BAN
from model.modules.fusion.fusion import MotionApprFusion, AttFlat

# torch.set_printoptions(threshold=np.inf)


class MASN(nn.Module):

    def __init__(
            self,
            vocab_size,
            s_layers,
            s_embedding,
            resnet_input_size,
            i3d_input_size,
            hidden_size,
            dropout_p=0.0,
            gcn_layers=2,
            answer_vocab_size=None,
            q_max_len=35,
            v_max_len=80,
            ablation='none'):
        super().__init__()

        self.ablation = ablation
        self.q_max_len = q_max_len
        self.v_max_len = v_max_len
        self.hidden_size = hidden_size

        self.compress_appr_local = dropnn.WeightDropLinear(
            resnet_input_size,
            hidden_size,
            weight_dropout=dropout_p,
            bias=False)
        self.compress_motion_local = dropnn.WeightDropLinear(
            i3d_input_size,
            hidden_size,
            weight_dropout=dropout_p,
            bias=False)
        self.compress_appr_global = dropnn.WeightDropLinear(
            resnet_input_size,
            hidden_size,
            weight_dropout=dropout_p,
            bias=False)
        self.compress_motion_global = dropnn.WeightDropLinear(
            i3d_input_size,
            hidden_size,
            weight_dropout=dropout_p,
            bias=False)

        embedding_dim = s_embedding.shape[1] if s_embedding is not None else hidden_size
        self.glove = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if s_embedding is not None:
            print("glove embedding weight is loaded!")
            self.glove.weight = nn.Parameter(torch.from_numpy(s_embedding).float())
        self.glove.weight.requires_grad = False
        self.embedding_proj = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(embedding_dim, hidden_size, bias=False)
        )

        self.sentence_encoder = SentenceEncoderRNN(
            vocab_size,
            hidden_size,
            input_dropout_p=dropout_p,
            dropout_p=dropout_p,
            n_layers=s_layers,
            bidirectional=True,
            rnn_cell='lstm'
        )

        self.bbox_location_encoding = nn.Linear(6, 64)
        self.pos_location_encoding = PositionEncoding(n_filters=64, max_len=self.v_max_len)

        self.appr_local_proj = nn.Linear(hidden_size+128, hidden_size)
        self.motion_local_proj = nn.Linear(hidden_size+128, hidden_size)

        self.pos_enc = PositionEncoding(n_filters=512, max_len=self.v_max_len)
        self.appr_v = nn.Linear(hidden_size*2, hidden_size)
        self.motion_v = nn.Linear(hidden_size*2, hidden_size)

        self.appr_adj = VideoAdjLearner(hidden_size, hidden_size)
        self.appr_gcn = GCN(hidden_size, hidden_size, hidden_size, num_layers=gcn_layers)
        self.motion_adj = VideoAdjLearner(hidden_size, hidden_size)
        self.motion_gcn = GCN(hidden_size, hidden_size, hidden_size, num_layers=gcn_layers)

        self.res_layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.i3d_layernorm = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.appr_vq_interact = BAN(hidden_size, glimpse=4)
        self.motion_vq_interact = BAN(hidden_size, glimpse=4)

        self.motion_appr_fusion = MotionApprFusion(hidden_size, hidden_size, n_layer=1)
        self.attflat = AttFlat(hidden_size, hidden_size, 1, hidden_size)

        if answer_vocab_size is not None:
            self.fc = nn.Linear(hidden_size, answer_vocab_size)
        else:
            self.fc = nn.Linear(hidden_size, 1)

    def forward(self, task, *args):
        # expected sentence_inputs is of shape (batch_size, sentence_len, 1)
        # expected video_inputs is of shape (batch_size, frame_num, video_feature)
        self.task = task
        if task == 'Count':
            return self.forward_count(*args)
        elif task == 'FrameQA':
            return self.forward_frameqa(*args)
        elif task == 'Action' or task == 'Trans':
            return self.forward_trans_or_action(*args)
        elif task == 'MS-QA':
            return self.forward_msqa(*args)

    def model_block(self, res_avg_inp, i3d_avg_inp, res_obj_inp, bbox_inp, i3d_obj_inp,
                    video_length, all_sen_inputs, all_ques_length):

        q_mask = self.make_mask(all_sen_inputs, all_ques_length)
        v_mask = self.make_mask(res_avg_inp[:,:,0], video_length)

        q_emb = F.relu(self.embedding_proj(self.glove(all_sen_inputs)))  # b, q_len, d
        q_output, q_hidden = self.sentence_encoder(q_emb, input_lengths=all_ques_length)
        q_hidden = q_hidden.squeeze()

        bsz, v_len, obj_num = res_obj_inp.size(0), res_obj_inp.size(1), res_obj_inp.size(2)
        q_len = q_output.size(1)
        q_mask = q_mask[:,:q_len]

        # make local and global feature
        res_obj_inp = self.compress_appr_local(res_obj_inp)  # b, v_len, N, d
        i3d_obj_inp = self.compress_motion_local(i3d_obj_inp)  # b, v_len, N, d
        res_avg_inp = self.compress_appr_global(res_avg_inp)  # b, v_len, d
        i3d_avg_inp = self.compress_motion_global(i3d_avg_inp)  # b, v_len, d

        bbox_inp = self.bbox_location_encoding(bbox_inp)  # b, v_len, N, d/8
        pos_inp = self.pos_location_encoding(res_obj_inp.contiguous().view(bsz*obj_num, v_len, -1))  # 1, v_len, 64
        pos_inp = pos_inp.unsqueeze(2).expand(bsz, v_len, obj_num, 64) * v_mask.unsqueeze(2).unsqueeze(3)  # b, v_len, N, d/8

        appr_local = self.appr_local_proj(torch.cat([res_obj_inp, bbox_inp, pos_inp], dim=3))  # b, v_len, N, d
        motion_local = self.motion_local_proj(torch.cat([i3d_obj_inp, bbox_inp, pos_inp], dim=3))  # b, v_len, N, d

        v_len = appr_local.size(1)
        appr_local = appr_local.contiguous().view(bsz*v_len, obj_num, self.hidden_size)
        motion_local = motion_local.contiguous().view(bsz*v_len, obj_num, self.hidden_size)

        res_avg_inp = self.pos_enc(res_avg_inp) + res_avg_inp
        res_avg_inp = res_avg_inp.contiguous().view(bsz*v_len, self.hidden_size)
        res_avg_inp = res_avg_inp.unsqueeze(1).expand_as(appr_local)
        appr_v = self.appr_v(torch.cat([appr_local, res_avg_inp], dim=-1))

        i3d_avg_inp = self.pos_enc(i3d_avg_inp) + i3d_avg_inp
        i3d_avg_inp = i3d_avg_inp.contiguous().view(bsz*v_len, self.hidden_size)
        i3d_avg_inp = i3d_avg_inp.unsqueeze(1).expand_as(motion_local)
        motion_v = self.motion_v(torch.cat([motion_local, i3d_avg_inp], dim=-1))

        appr_v = appr_v.contiguous().view(bsz, v_len*obj_num, self.hidden_size)
        motion_v = motion_v.contiguous().view(bsz, v_len*obj_num, self.hidden_size)
        v_mask_expand = v_mask[:,:v_len].unsqueeze(2).expand(bsz, v_len, obj_num).contiguous().view(bsz, v_len*obj_num)

        # object graph convolution
        appr_adj = self.appr_adj(appr_v, v_mask_expand)
        appr_gcn = self.appr_gcn(appr_v, appr_adj)  # b, v_len*obj_num, d
        motion_adj = self.motion_adj(motion_v, v_mask_expand)
        motion_gcn = self.motion_gcn(motion_v, motion_adj)  # b, v_len*obj_num, d

        # vq interaction
        appr_vq, _ = self.appr_vq_interact(appr_gcn, q_output, v_mask_expand, q_mask)
        motion_vq, _ = self.motion_vq_interact(motion_gcn, q_output, v_mask_expand, q_mask)

        # motion-appr fusion
        U = torch.cat([appr_vq, motion_vq], dim=1)  # b, 2*q_len, d
        q_mask_ = torch.cat([q_mask, q_mask], dim=1)
        U_mask = torch.matmul(q_mask_.unsqueeze(2), q_mask_.unsqueeze(2).transpose(1, 2))

        fusion_out = self.motion_appr_fusion(U, q_hidden, U_mask)
        fusion_out = self.attflat(fusion_out, q_mask_)

        out = self.fc(fusion_out).squeeze()
        return out

    def make_mask(self, seq, seq_length):
        mask = seq
        mask = mask.data.new(*mask.size()).fill_(1)
        for i, l in enumerate(seq_length):
            mask[i][min(mask.size(1)-1, l):] = 0
        mask = Variable(mask)  # b, seq_len
        mask = mask.to(torch.float)
        return mask

    def forward_count(
            self, res_avg_inp, i3d_avg_inp, res_obj_inp, bbox_inp, i3d_obj_inp, video_length,
            all_sen_inputs, all_ques_length, answers):
        # out of shape (batch_size, )
        out = self.model_block(
            res_avg_inp, i3d_avg_inp, res_obj_inp, bbox_inp, i3d_obj_inp, video_length,
            all_sen_inputs, all_ques_length)
        predictions = torch.clamp(torch.round(out), min=1, max=10).long()
        # answers of shape (batch_size, )
        return out, predictions, answers

    def forward_frameqa(
            self, res_avg_inp, i3d_avg_inp, res_obj_inp, bbox_inp, i3d_obj_inp, video_length,
            all_sen_inputs, all_ques_length, answers, answer_type):
        # out of shape (batch_size, num_class)
        out = self.model_block(
            res_avg_inp, i3d_avg_inp, res_obj_inp, bbox_inp, i3d_obj_inp, video_length,
            all_sen_inputs, all_ques_length)

        _, max_idx = torch.max(out, 1)
        # (batch_size, ), dtype is long
        predictions = max_idx
        # answers of shape (batch_size, )
        return out, predictions, answers

    def forward_trans_or_action(
            self, res_avg_inp, i3d_avg_inp, res_obj_inp, bbox_inp, i3d_obj_inp, video_length,
            all_cand_inputs, all_cand_length, answers, row_index):
        all_cand_inputs = all_cand_inputs.permute(1, 0, 2)
        all_cand_length = all_cand_length.permute(1, 0)

        all_out = []
        for idx in range(5):
            # out of shape (batch_size, )
            out = self.model_block(
                res_avg_inp, i3d_avg_inp, res_obj_inp, bbox_inp, i3d_obj_inp, video_length,
                all_cand_inputs[idx], all_cand_length[idx])
            all_out.append(out)
        # all_out of shape (batch_size, 5)
        all_out = torch.stack(all_out, 0).transpose(1, 0)
        _, max_idx = torch.max(all_out, 1)
        # (batch_size, )
        predictions = max_idx

        # answers of shape (batch_size, )
        return all_out, predictions, answers

    def forward_msqa(
            self, res_avg_inp, i3d_avg_inp, res_obj_inp, bbox_inp, i3d_obj_inp, video_length,
            all_sen_inputs, all_ques_length, answers):
        # out of shape (batch_size, num_class)
        out = self.model_block(
            res_avg_inp, i3d_avg_inp, res_obj_inp, bbox_inp, i3d_obj_inp, video_length,
            all_sen_inputs, all_ques_length)

        _, max_idx = torch.max(out, 1)
        # (batch_size, ), dtype is long
        predictions = max_idx
        # answers of shape (batch_size, )
        return out, predictions, answers
