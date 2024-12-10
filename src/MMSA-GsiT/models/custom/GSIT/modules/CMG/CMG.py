import torch
import torch.nn as nn

from easydict import EasyDict

from ..GraphAttentions import GraphFormerEncoder

__all__ = ['CrossModalGraph']

class CrossModalGraph(nn.Module):
    def __init__(self, args):
        super(CrossModalGraph, self).__init__()
        self.args = args
        self.cfg = cfg = args.cmg_cfg

        self.in_embed, self.num_heads = cfg.dst_feature_dim_nheads
        self.bidi = cfg.bidirectional
        if not cfg.bidirectional:
            self.cross_encoder_forward = GraphFormerEncoder(
                embed_dim=self.in_embed,
                num_heads=self.num_heads,
                layers=cfg.nlevels,
                attn_dropout=cfg.attn_dropout,
                relu_dropout=cfg.relu_dropout,
                res_dropout=cfg.res_dropout,
                embed_dropout=cfg.embed_dropout,
                attn_mask=cfg.attn_mask,
                direction='forward'
            )
            self.cross_encoder_backward = GraphFormerEncoder(
                embed_dim=self.in_embed,
                num_heads=self.num_heads,
                layers=cfg.nlevels,
                attn_dropout=cfg.attn_dropout,
                relu_dropout=cfg.relu_dropout,
                res_dropout=cfg.res_dropout,
                embed_dropout=cfg.embed_dropout,
                attn_mask=cfg.attn_mask,
                direction='backward'
            )
            self.self_encoder = GraphFormerEncoder(
                embed_dim=2*self.in_embed,
                num_heads=self.num_heads,
                layers=cfg.nlevels,
                attn_dropout=cfg.attn_dropout,
                relu_dropout=cfg.relu_dropout,
                res_dropout=cfg.res_dropout,
                embed_dropout=cfg.embed_dropout,
                attn_mask=cfg.attn_mask,
                direction='self'
            )
        elif cfg.bidirectional:
            self.cross_encoder_bidi = GraphFormerEncoder(
                embed_dim=2*self.in_embed,
                num_heads=self.num_heads,
                layers=cfg.nlevels,
                attn_dropout=cfg.attn_dropout,
                relu_dropout=cfg.relu_dropout,
                res_dropout=cfg.res_dropout,
                embed_dropout=cfg.embed_dropout,
                attn_mask=cfg.attn_mask,
            )
            self.self_encoder = GraphFormerEncoder(
                embed_dim=2*self.in_embed,
                num_heads=self.num_heads,
                layers=cfg.nlevels,
                attn_dropout=cfg.attn_dropout,
                relu_dropout=cfg.relu_dropout,
                res_dropout=cfg.res_dropout,
                embed_dropout=cfg.embed_dropout,
                attn_mask=cfg.attn_mask,
            )
    
    def forward(self, cat_seq, split, plot_map):
        # self.bidi = True
        if not self.bidi:
            cross_mask_forward = self.build_adj_masked_matrix(
                split, mode='cross', direction='forward'
            ).to(cat_seq.device)
            cross_mask_backward = self.build_adj_masked_matrix(
                split, mode='cross', direction='backward'
            ).to(cat_seq.device)
            self_mask = self.build_adj_masked_matrix(
                split, mode='self'
            ).to(cat_seq.device)

            # if plot_map:
            #     def plot(temp):
            #         import numpy as np
            #         import matplotlib.pyplot as plt
            #         import seaborn as sns
                    
            #         mask_arr = np.array(temp)
            #         plt.figure(figsize=(10, 10), dpi=600)
            #         sns.heatmap(mask_arr, cmap='viridis', cbar=False)
            #         plt.show()       
                
            #     temp_1 = cross_mask_forward.clone().detach().to('cpu')
            #     temp_2 = cross_mask_backward.clone().detach().to('cpu')
            #     temp_3 = self_mask.clone().detach().to('cpu')
            #     plot(temp_1)
            #     plot(temp_2)
            #     plot(temp_3)

            cross_attned_seq_forward, f_w = self.cross_encoder_forward(
                cat_seq, mask=cross_mask_forward, mask_fixer=None, plot_map=False,
                seq_split=split
            )
            cross_attned_seq_forward = cross_attned_seq_forward.permute(1, 0, 2)
            cross_attned_seq_backward, b_w = self.cross_encoder_backward(
                cat_seq, mask=cross_mask_backward, mask_fixer=None, plot_map=False,
                seq_split=split
            )
            cross_attned_seq_backward = cross_attned_seq_backward.permute(1, 0, 2)

            # # Ablation
            # cross_attned_seq_forward, f_w = self.cross_encoder_forward(
            #     cat_seq, mask_fixer=None, plot_map=False
            # )
            # cross_attned_seq_forward = cross_attned_seq_forward.permute(1, 0, 2)
            # cross_attned_seq_backward, b_w = self.cross_encoder_backward(
            #     cat_seq, mask_fixer=None, plot_map=False
            # )
            # cross_attned_seq_backward = cross_attned_seq_backward.permute(1, 0, 2)

            # Building Circular 
            # forward_split = torch.split(cross_attned_seq_forward, split, dim=0)   # [v->t, t->v, v->a]
            # backward_split = torch.split(cross_attned_seq_backward, split, dim=0) # [a->t, a->v, t->a]
            # true_forward = torch.cat([
            #     forward_split[0], backward_split[1], backward_split[2] # [v->t, a->v, t->a]
            # ], dim=0)
            # true_backward = torch.cat([
            #     backward_split[0], forward_split[1], forward_split[2]  # [a->t, t->v, v->a]
            # ], dim=0)

            # bi_direction_seq = torch.cat([true_forward, true_backward], dim=-1)

            bi_direction_seq = torch.cat([
                cross_attned_seq_forward, cross_attned_seq_backward
            ], dim=-1).permute(1, 0, 2)

            self_attned_seq, s_w = self.self_encoder(bi_direction_seq, mask=self_mask, plot_map=plot_map,
                                                     seq_split=split)
            # self_attned_seq, s_w = self.self_encoder(bi_direction_seq, plot_map=plot_map)

            return EasyDict({
                'attned_seq': self_attned_seq,
                'split': list(seq.permute(1, 0, 2) for seq in torch.split(self_attned_seq, split, dim=0)),
                'weights': (f_w, b_w, s_w)
            })
        elif self.bidi:
            cross_mask_bidi = self.build_adj_masked_matrix(
                split, mode='cross', direction='bidirectional'
            ).to(cat_seq.device)
            self_mask = self.build_adj_masked_matrix(
                split, mode='self', direction='bidirectional'
            ).to(cat_seq.device)
            cat_seq = torch.concat([cat_seq, cat_seq], dim=-1)

            cross_attned_seq_bidi, _ = self.cross_encoder_bidi(
                cat_seq, mask=cross_mask_bidi, mask_fixer=None
            )
            self_attned_seq, _ = self.self_encoder(
                cross_attned_seq_bidi, mask=self_mask, mask_fixer=None
            )
            return EasyDict({
                'attned_seq': self_attned_seq,
                'split': list(seq.permute(1, 0, 2) for seq in torch.split(self_attned_seq, split, dim=0))
            })
    

    def build_adj_masked_matrix(self, split, mode='cross', direction='forward'):
        t, v, a = split
        s1 = (0, t)                 # [0, t)
        s2 = (t, t + v)             # [t, t + v)
        s3 = (t + v, t + v + a)     # [t + v, t + v + a)
        sum_len = sum(split)
        mask_list = []
        for idx, split_len in enumerate(split):
            for _ in range(split_len):
                row_mask_tensor = torch.ones(sum_len, dtype=torch.float32)
                if idx == 0:
                    row_mask_tensor[0:s1[1]] = 0
                    if mode == 'cross':
                        if direction == 'forward':
                            row_mask_tensor[s3[0]:] = 0        # Original
                            # row_mask_tensor[s2[0]:s2[1]] = 0 # Structure-1
                            # row_mask_tensor[s3[0]:] = 0      # Structure-2
                            # row_mask_tensor[s3[0]:] = 0      # Structure-3
                        elif direction == 'backward':
                            row_mask_tensor[s2[0]:s2[1]] = 0   # Original
                            # row_mask_tensor[s3[0]:] = 0      # Structure-1
                            # row_mask_tensor[s2[0]:s2[1]] = 0 # Structure-2
                            # row_mask_tensor[s2[0]:s2[1]] = 0 # Structure-3
                            
                elif idx == 1:
                    row_mask_tensor[s2[0]:s2[1]] = 0
                    if mode == 'cross':
                        if direction == 'forward':
                            row_mask_tensor[0:s1[1]] = 0       # Original
                            # row_mask_tensor[0:s1[1]] = 0     # Structure-1
                            # row_mask_tensor[s3[0]:] = 0      # Structure-2
                            # row_mask_tensor[0:s1[1]] = 0     # Structure-3
                        elif direction == 'backward':
                            row_mask_tensor[s3[0]:] = 0        # Original
                            # row_mask_tensor[s3[0]:] = 0      # Structure-1
                            # row_mask_tensor[0:s1[1]] = 0     # Structure-2
                            # row_mask_tensor[s3[0]:] = 0      # Structure-3
                            
                elif idx == 2:
                    row_mask_tensor[s3[0]:s3[1]] = 0
                    if mode == 'cross':
                        if direction == 'forward':
                            row_mask_tensor[s2[0]:s2[1]] = 0   # Original
                            # row_mask_tensor[s2[0]:s2[1]] = 0 # Structure-1
                            # row_mask_tensor[0:s1[1]] = 0     # Structure-2
                            # row_mask_tensor[0:s1[1]] = 0     # Structure-3
                        elif direction == 'backward':
                            row_mask_tensor[0:s1[1]] = 0       # Original
                            # row_mask_tensor[s2[0]:s2[1]] = 0 # Structure-1
                            # row_mask_tensor[s2[0]:s2[1]] = 0 # Structure-2
                            # row_mask_tensor[s2[0]:s2[1]] = 0 # Structure-3
                            
                mask_list.append(row_mask_tensor)
        if mode == 'cross':
            mask = torch.stack(mask_list)
            if direction == 'forward':
                return self.get_mask_neginf_0(mask)
            elif direction == 'backward':
                return self.get_mask_neginf_0(mask)
            elif direction == 'bidirectional':
                return self.get_mask_neginf_0(mask)
            else:
                raise ValueError(
                    'direction must be \'forward\' or \'backward\' or \'bidirectional\''
                )
        elif mode == 'self':
            return self.get_mask_neginf_0(
                torch.abs(torch.stack(mask_list) - 1)
            )
        else:
            raise ValueError(
                r'mode must be \'cross\' or \'self\''
            )
    
    def build_adj_masked_matrix_ablation(self, split, mode='cross', direction='forward'):
        a, b = split
        s1 = (0, a)                 # [0, t)
        s2 = (a, a + b)             # [t, t + v)
        # s3 = (t + v, t + v + a)     # [t + v, t + v + a)
        sum_len = sum(split)
        mask_list = []
        for idx, split_len in enumerate(split):
            for _ in range(split_len):
                row_mask_tensor = torch.ones(sum_len, dtype=torch.float32)
                if idx == 0:
                    row_mask_tensor[0:s1[1]] = 0
                elif idx == 1:
                    row_mask_tensor[s2[0]:s2[1]] = 0   
                mask_list.append(row_mask_tensor)
        if mode == 'cross':
            mask = torch.stack(mask_list)
            if direction == 'forward':
                return self.get_mask_neginf_0(mask)
            elif direction == 'backward':
                return self.get_mask_neginf_0(mask)
            elif direction == 'bidirectional':
                return self.get_mask_neginf_0(mask)
            else:
                raise ValueError(
                    'direction must be \'forward\' or \'backward\' or \'bidirectional\''
                )
        elif mode == 'self':
            return self.get_mask_neginf_0(
                torch.abs(torch.stack(mask_list) - 1)
            )
        else:
            raise ValueError(
                r'mode must be \'cross\' or \'self\''
            )

    def get_mask_neginf_0(self, mask):
        neg_inf = -10e9
        return torch.where(
            mask == 0, neg_inf,
            torch.tensor(0, dtype=torch.float32)
        )
