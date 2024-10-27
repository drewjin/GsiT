import torch
import torch.nn as nn

from easydict import EasyDict
from torch.nn.utils.rnn import pack_padded_sequence

from ..xLSTM import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig


__all__ = ["xLSTMProj", "NaiveLSTMProj"]


class xLSTMProj(nn.Module):
    def __init__(self, args):
        super(xLSTMProj, self).__init__()
        self.xlstm = xLSTMBlockStack(self.xlstm_cfg(args))
        self.dropout = nn.Dropout(args.lstm_dropout)
        self.proj = nn.Linear(args.in_embed, args.out_embed)
        # self.proj_conv = nn.Linear(args.in_embed*2, args.out_embed)

    def xlstm_cfg(self, args):
        return xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(), 
            context_length=args.context_length, 
            num_blocks=args.num_blocks, 
            embedding_dim=args.in_embed, 
        )
    
    def forward(self, x):
        hidden_states = self.xlstm(x)
        hidden_states_ = self.dropout(hidden_states)
        last_hidden = hidden_states_.permute(1, 0, 2)[-1]
        proj_output = self.proj(last_hidden)
        # proj_conv = self.proj_conv(conv)
        return EasyDict({
            'pooler_output': proj_output,
            'hidden_states': hidden_states,
            # 'conv': proj_conv,
        })


class  NaiveLSTMProj(nn.Module):
    def __init__(
        self, 
        in_size, 
        hidden_size, 
        out_size, 
        num_layers=1, 
        dropout=0.2, 
        bidirectional=False
    ):
        super(NaiveLSTMProj, self).__init__()
        self.rnn = nn.LSTM(
            in_size, hidden_size, num_layers=num_layers, 
            dropout=dropout, bidirectional=bidirectional, 
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        # packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(x) 
        h = self.dropout(final_states[0][-1])
        proj_output = self.proj(h)
        return EasyDict({
            'pooler_output': proj_output,
            'hidden_states': final_states[0],
        })