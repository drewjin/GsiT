import torch
import torch.nn as nn

from .FFNs import FeedForwardNetwork


__all__ = ['DecomposeEnhanceBlock', 'Decomposer', 'DecomposeAttention']


class Decomposer(nn.Module):
    """
        Decomposer: Decompose the input into activities
    """
    def __init__(self, args, split_rate) -> None:
        super(Decomposer, self).__init__()
        self.args = args
        self.split_rate = split_rate

    def forward(self, input):
        if input.shape[-1] != sum(self.split_rate):
            raise ValueError("Input feature dimension does not match the sum of split_rate")
        decomposed_features = []
        current_index = 0
        for rate in self.split_rate:
            feature_slice = input[..., current_index:current_index + rate]
            decomposed_features.append(feature_slice)
            current_index += rate
        return decomposed_features
    
    def get_dec_heads(self):
        return len(self.split_rate)
    

class DecomposeAttention(nn.Module):
    """
        DecomposeAttention: Decomposed Features were used as input to the attention mechanism
        - Ordinary dot product attention (v1)
        - Multi-head attention (v2)
    """
    def __init__(self, args, version, split_rate) -> None:
        super(DecomposeAttention, self).__init__()
        MODEL_MAP = {'v1': self.__dot_product, 'v2': self.__mh_attn, 'v3':self.__mh_attn}
        MODEL_MAP[version](args, split_rate)
    
    def __dot_product(self, args, split_rate):
        nheads = args['num_specific_heads']
        if nheads != 1:
            raise NotImplementedError(
                r"The current implementation only supports nheads equal to 1." 
                r"Please adjust the value of nheads accordingly."
            )
        dropout = args['dropout']
        self.attn_layers = nn.ModuleList([ # NOTE： How to parallelize the computation?
            nn.MultiheadAttention(embed_dim=dec_size, num_heads=nheads, dropout=dropout)
            for dec_size in split_rate
        ])

    def __mh_attn(self, args, split_rate):
        nheads = args['num_specific_heads']
        dropout = args['dropout']
        self.attn_layers = nn.ModuleList([ # NOTE： How to parallelize the computation?
            nn.MultiheadAttention(embed_dim=dec_size, num_heads=nhead, dropout=dropout)
            for dec_size, nhead in zip(split_rate, nheads)
        ])

    def forward(self, X):
        return torch.cat([
            attn_layer_i(query=x_i, key=x_i, value=x_i)[0] 
            for x_i, attn_layer_i in zip(X, self.attn_layers)], dim=-1)


class DecomposeEnhanceLayer(nn.Module):
    """
        DecomposeEnhanceLayer: 
    """
    def __init__(self, args, split_rate, version) -> None:
        super(DecomposeEnhanceLayer, self).__init__()
        self.args = args
        self.use_residual = args.get('residual', True)
        self.use_ffn = args.get('ffn', True)
        self.embed_dim = embed_dim = sum(split_rate)
        self.decomposer = Decomposer(args, split_rate)
        self.dec_attn   = DecomposeAttention(args['decompose_attn'], version, split_rate)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        if self.use_ffn:
            self.ffn = FeedForwardNetwork(args['decompose_ffn'], embed_dim, embed_dim)
            self.layer_norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, X):
        # Decompose Attention
        res_X = X
        dec_X = self.decomposer(X)
        dec_att_X = self.dec_attn(dec_X)
        if self.use_residual:
            H = self.layer_norm1(dec_att_X + res_X)
        else:
            H = self.layer_norm1(dec_att_X)
        
        # Feed Forward Network
        res_H = H
        if self.use_ffn:
            T = self.ffn(H)
            if self.use_residual:
                O = self.layer_norm2(T + res_H)
            else:
                O = self.layer_norm2(T)
        else:
            O = H

        return O
        


class DecomposeEnhanceBlock(nn.Module):
    """
        DecomposeEnhanceBlock:
    """
    def __init__(self, args, split_rate, version) -> None:
        super(DecomposeEnhanceBlock, self).__init__()
        self.args  = args
        num_layers = args.get('num_layers', 1)
        a_sr, v_sr = split_rate.values()
        self.v_layers = nn.Sequential(*[
            DecomposeEnhanceLayer(args, v_sr, version) 
            for _ in range(num_layers)
        ])
        self.a_layers = nn.Sequential(*[
            DecomposeEnhanceLayer(args, a_sr, version) 
            for _ in range(num_layers)
        ])

    def forward(self, X_v, X_a):
        return (self.v_layers(X_v), self.a_layers(X_a))