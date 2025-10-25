from math import ceil

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import Conv1d
from vita.modeling.transformer_decoder.vita import VITA, MLP, CrossAttentionLayer, SelfAttentionLayer, FFNLayer


class NoiseFilter(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(C * 2, C // 4),
            nn.ReLU(),
            nn.Linear(C // 4, C),
            nn.Sigmoid()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(C, C, (1, 3), padding=(0, 1), groups=C),
            nn.Conv2d(C, C, (3, 1), padding=(1, 0), groups=C)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(C, C, (1, 5), padding=(0, 2), groups=C),
            nn.Conv2d(C, C, (5, 1), padding=(2, 0), groups=C)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(C, C, (1, 7), padding=(0, 3), groups=C),
            nn.Conv2d(C, C, (7, 1), padding=(3, 0), groups=C)
        )
        self.fuse = nn.Conv2d(C * 4, C, 1)

    def forward(self, x):
        N, C = x.shape

        max_pool = x.max(dim=0, keepdim=True)[0]
        avg_pool = x.mean(dim=0, keepdim=True)
        channel_att = self.mlp(torch.cat([max_pool, avg_pool], dim=1))
        x_channel = x * channel_att

        x_2d = x_channel.unsqueeze(0).unsqueeze(-1)

        branch0 = x_2d
        branch1 = self.conv3(x_2d)
        branch2 = self.conv5(x_2d)
        branch3 = self.conv7(x_2d)

        spatial_feat = torch.cat([branch0, branch1, branch2, branch3], dim=1)
        spatial_att = self.fuse(spatial_feat).sigmoid()

        out = x_channel * spatial_att.squeeze(0).squeeze(-1)

        return out


class GateLayer(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.W_s = nn.Linear(C, C)
        self.W_f = nn.Linear(C, C)
        self.threshold = nn.Parameter(torch.tensor(0.5))
        self.temp = 0.1

    def forward(self, state_token, frame_query):
        Q_s = self.W_s(state_token)
        K_f = self.W_f(frame_query)

        attn = torch.softmax(torch.einsum('nc,mc->nm', Q_s, K_f) / (Q_s.shape[-1] ** 0.5), dim=-1)
        conf = attn.sum(dim=-1)

        soft_gate = torch.sigmoid((conf - self.threshold) / self.temp)

        return soft_gate.unsqueeze(-1), conf


class TGMNVIS(VITA):
    def __init__(self, cfg):
        super().__init__(
            cfg=cfg,
            in_channels=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            aux_loss=cfg.MODEL.VITA.DEEP_SUPERVISION,
        )

        self.num_frames = cfg.MODEL.GENVIS.LEN_CLIP_WINDOW
        hidden_dim = cfg.MODEL.VITA.HIDDEN_DIM

        self.state_token_init = nn.Embedding(self.num_queries, hidden_dim)

        self.temporal_cross_attn = nn.ModuleList()
        self.temporal_gate = nn.ModuleList()
        self.temporal_self_attn = nn.ModuleList()

        self.appearance_cross_attn = nn.ModuleList()
        self.masked_attn = nn.ModuleList()

        for _ in range(self.num_layers):
            self.temporal_cross_attn.append(CrossAttentionLayer(hidden_dim, self.num_heads, 0.0, "relu", False))
            self.temporal_gate.append(GateLayer(hidden_dim))
            self.temporal_self_attn.append(SelfAttentionLayer(hidden_dim, self.num_heads, 0.0, "relu", False))

            self.appearance_cross_attn.append(CrossAttentionLayer(hidden_dim, self.num_heads, 0.0, "relu", False))
            self.masked_attn.append(SelfAttentionLayer(hidden_dim, self.num_heads, 0.0, "relu", False))

        self.noise_filter = NoiseFilter(hidden_dim)

        self.use_mem = cfg.MODEL.GENVIS.USE_MEM
        if self.use_mem:
            self.pre_memory_embed_k = nn.Linear(hidden_dim, hidden_dim)
            self.pre_memory_embed_v = nn.Linear(hidden_dim, hidden_dim)
            self.pre_query_embed_k = nn.Linear(hidden_dim, hidden_dim)
            self.pre_query_embed_v = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, frame_query, pre_memory, output, prev_state_token=None):
        if not self.training:
            frame_query = frame_query[[-1]]

        pre_memory_k = pre_memory["k"]
        pre_memory_v = pre_memory["v"]

        L, BT, fQ, C = frame_query.shape
        B = BT // self.num_frames if self.training else 1
        T = self.num_frames if self.training else BT // B

        frame_query = frame_query.reshape(L * B, T, fQ, C)
        frame_query = frame_query.permute(1, 2, 0, 3).contiguous()
        frame_query = self.input_proj_dec(frame_query)

        if self.window_size > 0:
            pad = int(ceil(T / self.window_size)) * self.window_size - T
            _T = pad + T
            frame_query = F.pad(frame_query, (0, 0, 0, 0, 0, 0, 0, pad))
            enc_mask = frame_query.new_ones(L * B, _T).bool()
            enc_mask[:, :T] = False
        else:
            enc_mask = None

        frame_query = self.encode_frame_query(frame_query, enc_mask)
        frame_query = frame_query[:T].flatten(0, 1)

        if self.use_sim:
            pred_fq_embed = self.sim_embed_frame(frame_query)
            pred_fq_embed = pred_fq_embed.transpose(0, 1).reshape(L, B, T, fQ, C)
        else:
            pred_fq_embed = None

        src = self.src_embed(frame_query)
        dec_pos = self.fq_pos.weight[None, :, None, :].repeat(T, 1, L * B, 1).flatten(0, 1)

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, L * B, 1)

        cQ, LB, C = output.shape

        if prev_state_token is None:
            state_token = self.state_token_init.weight.unsqueeze(1).repeat(1, L * B, 1)
        else:
            state_token = prev_state_token

        appearance_query = output

        if self.use_mem:
            pre_query_k = self.pre_query_embed_k(output)
            pre_query_v = self.pre_query_embed_v(output)

            if pre_memory_k and pre_memory_v:
                pre_memory_k = torch.cat(pre_memory_k).flatten(1, 2)
                pre_memory_v = torch.cat(pre_memory_v).flatten(1, 2)
            else:
                pre_memory_k = torch.empty((0, LB, cQ, C), device=output.device)
                pre_memory_v = torch.empty((0, LB, cQ, C), device=output.device)

            qk_mk = torch.einsum("qbc, mbpc -> bqmp", pre_query_k, pre_memory_k)
            qk_mk = torch.einsum("bqmq -> bqm", qk_mk)
            qk_mk = F.softmax(qk_mk, dim=2)
            qk_mk_mv = torch.einsum("bqm, mbqc-> qbc", qk_mk, pre_memory_v)

            pre_query_v = pre_query_v + qk_mk_mv
            appearance_query = appearance_query + pre_query_v

        decoder_outputs = []

        for i in range(self.num_layers):
            state_token_before = state_token.clone()

            state_token = self.temporal_cross_attn[i](
                state_token, src,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=dec_pos, query_pos=query_embed
            )

            soft_gate, conf = self.temporal_gate[i](state_token, src)
            state_token = soft_gate * state_token + (1 - soft_gate) * state_token_before

            state_token = self.temporal_self_attn[i](
                state_token, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            appearance_query = self.appearance_cross_attn[i](
                appearance_query, src,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=dec_pos, query_pos=query_embed
            )

            _, app_conf = self.temporal_gate[i](appearance_query, src)

            joint_conf = torch.cat([conf.unsqueeze(0), app_conf.unsqueeze(0)], dim=0)
            mask = (joint_conf.unsqueeze(-1) * joint_conf.unsqueeze(-2)) > (self.temporal_gate[i].threshold ** 2)
            mask = mask.view(2 * cQ, 2 * cQ).float()
            mask = mask * -1000
            mask = mask.unsqueeze(0).repeat(L * B, 1, 1)

            joint_feat = torch.cat([state_token, appearance_query], dim=0)
            joint_feat = self.masked_attn[i](
                joint_feat, tgt_mask=mask,
                tgt_key_padding_mask=None,
                query_pos=None
            )

            state_token = joint_feat[:cQ]
            appearance_query = joint_feat[cQ:]

            appearance_query = self.transformer_ffn_layers[i](appearance_query)

            if (self.training and self.aux_loss) or (i == self.num_layers - 1):
                dec_out = self.decoder_norm(appearance_query)
                dec_out = dec_out.transpose(0, 1).view(L, B, self.num_queries, C)
                decoder_outputs.append(dec_out)

        decoder_outputs = torch.stack(decoder_outputs, dim=0)

        pred_cls = self.class_embed(decoder_outputs)
        pred_mask_embed = self.mask_embed(decoder_outputs)
        if self.use_sim and self.sim_use_clip:
            pred_cq_embed = self.sim_embed_clip(decoder_outputs)
        else:
            pred_cq_embed = [None] * self.num_layers

        final_output = decoder_outputs[-1]

        filtered_output = []
        for l_idx in range(L):
            for b_idx in range(B):
                feat = final_output[l_idx, b_idx]
                feat = self.noise_filter(feat)
                filtered_output.append(feat)
        filtered_output = torch.stack(filtered_output, dim=0).view(L, B, self.num_queries, C)

        if self.use_mem:
            memory_input = filtered_output
            pre_memory_k = self.pre_memory_embed_k(memory_input)[None]
            pre_memory_v = self.pre_memory_embed_v(memory_input)[None]

        out = {
            'pred_logits': pred_cls[-1],
            'pred_mask_embed': pred_mask_embed[-1],
            'pred_fq_embed': pred_fq_embed,
            'pred_cq_embed': pred_cq_embed[-1],
            'aux_outputs': self._set_aux_loss(
                pred_cls, pred_mask_embed, pred_cq_embed, pred_fq_embed, [None] * self.num_layers
            ),
            'pre_memory': {"k": pre_memory_k, "v": pre_memory_v},
        }

        return out, appearance_query, state_token