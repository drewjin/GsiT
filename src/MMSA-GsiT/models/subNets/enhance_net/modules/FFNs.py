import torch.nn as nn


__all__ = ['FeedForwardNetwork', 'LateFFN']


class FeedForwardNetwork(nn.Module):
    def __init__(self, args, in_embed_dim, o_embed_dim) -> None:
        super(FeedForwardNetwork, self).__init__()
        self.args = args
        self.MLP = nn.Sequential(
            nn.Linear(in_embed_dim, 2*in_embed_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(2*in_embed_dim, o_embed_dim),
        )
    
    def forward(self, X):
        return self.MLP(X)
    

class LateFFN(nn.Module):
    def __init__(self, args, v_embed_dim, a_embed_dim):
        super(LateFFN, self).__init__()
        self.v_net = nn.Sequential(
            nn.LayerNorm(v_embed_dim),
            FeedForwardNetwork(args, v_embed_dim, v_embed_dim),
            nn.LayerNorm(v_embed_dim)
        )
        self.a_net = nn.Sequential(
            nn.LayerNorm(a_embed_dim),
            FeedForwardNetwork(args, a_embed_dim, a_embed_dim),
            nn.LayerNorm(a_embed_dim)
        )

    def forward(self, X_v, X_a):
        return (self.v_net(X_v), self.a_net(X_a))