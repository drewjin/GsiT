import torch.nn as nn

from .FFNs import FeedForwardNetwork


__all__ = ['BaseEnhanceBlock', "SimpleEnhanceBlock"]


class BaseEnhanceLayer(nn.Module):
    """
        EarlyEnhanceLayer
    """
    def __init__(self, args, embed_dim):
        super(BaseEnhanceLayer, self).__init__()
        self.args = args
        self.use_residual = args.get('residual', True)
        self.use_ffn = args.get('ffn', True)
        attn_param = args.get('attn', None)
        ffn_param = args.get('mlp', None)
        proj_dim = attn_param['embed_dim']
        self.proj1 = nn.Linear(embed_dim, proj_dim)
        self.layer_norm1 = nn.LayerNorm(proj_dim)
        self.mh_attn = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=attn_param['num_heads'], dropout=attn_param['dropout'], batch_first=True
        )
        self.layer_norm2 = nn.LayerNorm(proj_dim)
        if args['ffn']:
            self.layer_norm3 = nn.LayerNorm(proj_dim)
            self.ffn = FeedForwardNetwork(ffn_param, proj_dim, proj_dim)
            self.layer_norm4 = nn.LayerNorm(proj_dim)
        self.proj2 = nn.Linear(proj_dim, embed_dim)


    def forward(self, X):
        X = self.proj1(X)
        res_X = X 
        X = self.layer_norm1(X)
        X = self.mh_attn(X, X, X)[0]
        if self.use_residual:
            H = self.layer_norm2(X + res_X)
        else:
            H = self.layer_norm2(X)
        
        res_H = H
        if self.use_ffn:
            H = self.layer_norm3(H)
            T = self.ffn(H)
            if self.use_residual:
                O = self.layer_norm4(T + res_H)
            else:
                O = self.layer_norm4(T)
        else:
            O = H
        
        return self.proj2(O)

class BaseEnhanceBlock(nn.Module):
    """
        EarlyEnhanceBlock
    """
    def __init__(self, args, split_rate):
        super(BaseEnhanceBlock, self).__init__()
        self.args = args
        num_layers = args.get('num_layers', 1)
        a_embed_dim, v_embed_dim = sum(split_rate['audio']), sum(split_rate['vision'])
        self.v_layers = nn.Sequential(*[
            BaseEnhanceLayer(args, v_embed_dim)
            for _ in range(num_layers)
        ])
        self.a_layers = nn.Sequential(*[
            BaseEnhanceLayer(args, a_embed_dim)
            for _ in range(num_layers)
        ])
    def forward(self, X_v, X_a):
        return (self.v_layers(X_v), self.a_layers(X_a))
    

class SimpleEnhanceLayer(nn.Module):
    """
        SimpleEnhanceLayer
    """
    def __init__(self, args, embed_dim):
        super(SimpleEnhanceLayer, self).__init__()
        self.args = args
        self.use_residual = args.get('residual', True)
        self.use_ffn = args.get('ffn', True)
        attn_param = args.get('attn', None)
        proj_dim = attn_param['embed_dim']
        self.proj1 = nn.Linear(embed_dim, proj_dim)
        self.mh_attn = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=attn_param['num_heads'], dropout=attn_param['dropout'], batch_first=True
        )
        self.layer_norm = nn.LayerNorm(proj_dim)
        self.proj2 = nn.Linear(proj_dim, embed_dim)
    
    def forward(self, X):
        X = self.proj1(X)
        res = X
        X = self.mh_attn(X, X, X)[0]
        if self.use_residual:
            H = self.layer_norm(X + res)
        else:
            H = self.layer_norm(X)
        O = self.proj2(H)
        return O
    

class SimpleEnhanceBlock(nn.Module):
    def __init__(self, args, split_rate):
        super(SimpleEnhanceBlock, self).__init__()
        self.args = args
        num_layers = args.get('num_layers', 1)
        a_embed_dim, v_embed_dim = sum(split_rate['audio']), sum(split_rate['vision'])
        self.v_layers = nn.Sequential(*[
            SimpleEnhanceLayer(args, v_embed_dim)
            for _ in range(num_layers)
        ])
        self.a_layers = nn.Sequential(*[
            SimpleEnhanceLayer(args, a_embed_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, X_v, X_a):
        return (self.v_layers(X_v), self.a_layers(X_a))