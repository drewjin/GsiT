import torch
import torch.nn as nn
import torch.nn.functional as F

from rtdl_num_embeddings import PeriodicEmbeddings

__all__ = ['VAEmbeddings']

class VAEmbeddings(nn.Module):
    def __init__(self, args, in_dim, out_dim):
        super().__init__()
        freq = args.va_embed_freq
        feature_wise_embed = args.feature_wise_embed
        res_dropout = args.va_embed_res_dropout
        activation_dropout = args.va_embed_activation_dropout
        
        self.res_dropout = res_dropout
        self.activation_dropout = activation_dropout
        self.tab_embedding = PeriodicEmbeddings(
            n_features=in_dim, d_embedding=feature_wise_embed,
            n_frequencies=freq, lite=False
        )
        self.attn = nn.ModuleList([nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=1, batch_first=True,
            dropout=self.activation_dropout, bias=True) 
            for _ in range(feature_wise_embed)
        ])
        self.hidden_dim = hidden_dim = feature_wise_embed * in_dim
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.fc_1 = nn.Linear(hidden_dim, hidden_dim//4)
        self.fc_2 = nn.Linear(hidden_dim//4, hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)
        self.proj_fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, X):
        embed_X = self.tab_embedding(X).permute(-1, 0, 1, 2)
        x_sp = embed_X.shape
        mask = torch.triu(torch.ones(x_sp[2], x_sp[2]), diagonal=1).bool().to(X.device)

        residual = embed_X.reshape((x_sp[1], x_sp[2], x_sp[0] * x_sp[-1]))
        embed_H = torch.concat([
            _attn(embed_x, embed_x, embed_x, attn_mask=mask)[0] 
            for _attn, embed_x in zip(self.attn, embed_X)
        ], dim=-1)
        embed_H = F.dropout(embed_H, p=self.res_dropout, training=self.training)
        embed_H = self.layer_norm_1(embed_H + residual)

        residual = embed_H
        embed_H = F.relu(self.fc_1(embed_H))
        embed_H = F.dropout(embed_H, p=self.res_dropout, training=self.training)
        embed_H = self.fc_2(embed_H)
        embed_H = F.dropout(embed_H, p=self.res_dropout, training=self.training)
        embed_H = self.layer_norm_2(residual + embed_H)
        embed_O = self.proj_fc(embed_H)
        return embed_O