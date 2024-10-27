import sys 
sys.path.append('..')

import torch
import torch.nn as nn

from easydict import EasyDict as edict
from transformers import ResNetModel
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig

__all__ = ['TempResNet']

class TempResNet(nn.Module):
    def __init__(self, args) -> None:
        super(TempResNet, self).__init__()
        # self.resnet_weight = r'/home/drew/Desktop/Research/weights/resnet/resnet-18'
        self.resnet_weight = args.resnet
        self.resnet = ResNetModel.from_pretrained(self.resnet_weight)
        self.resnet_output_dim = args.resnet_out
        self.hidden_size = args.resnet_out
        self.xlstm = False
        if args.use_xlstm:
            self.xlstm = True
            self.xlstm_stack = xLSTMBlockStack(self.xlstm_cfg(args))
        else:
            self.lstm = nn.LSTMCell(input_size=self.resnet_output_dim, hidden_size=self.hidden_size)
        self.cls_proj = nn.Linear(self.hidden_size, args.video_out)

    def xlstm_cfg(self, args):
        return xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(), 
            context_length=15, 
            num_blocks=8, 
            embedding_dim=args.resnet_out, 
        )

    def forward(self, x):
        device = x.device
        if not self.xlstm:
            hidden = torch.zeros(x.shape[0], self.hidden_size).to(device)
            cell = torch.zeros(x.shape[0], self.hidden_size).to(device)
            hidden_states = []
            cell_states = [] 
        x = x.permute(1, 0, 4, 2, 3)
        extracted_features = []
        for x_i in x:
            resnet_output = self.resnet(x_i)
            v_feature = resnet_output.pooler_output.squeeze()
            extracted_features.append(v_feature)
            if not self.xlstm:
                prev_hidden, prev_cell = hidden, cell
                new_hidden, new_cell = self.lstm(v_feature, (prev_hidden, prev_cell))
                hidden, cell = new_hidden, new_cell
                hidden_states.append(new_hidden) 
                cell_states.append(new_cell)
        extracted_features = torch.stack(extracted_features).permute(1, 0, 2)
        if self.xlstm:
            xlstm_hidden_states = self.xlstm_stack(extracted_features)
            xlstm_last_hidden_state = xlstm_hidden_states.permute(1, 0, 2)[-1]
            cls_proj_output = self.cls_proj(xlstm_last_hidden_state)
            ans = edict({
                'last_hidden_state': xlstm_last_hidden_state,
                'pooler_output': cls_proj_output,
                'hidden_states': xlstm_hidden_states,
                'extracted_features': extracted_features,
                'cell_states': None
            })
        else:
            hidden_states = torch.stack(hidden_states).permute(1, 0, 2)
            cell_states = torch.stack(cell_states).permute(1, 0, 2)
            last_hidden_state = hidden_states.permute(1, 0, 2)[-1]
            cls_proj_output = self.cls_proj(last_hidden_state)
            ans = edict({
                'last_hidden_state': last_hidden_state,
                'pooler_output': cls_proj_output,
                'hidden_states': hidden_states,
                'extracted_features': extracted_features,
                'cell_states': cell_states
            })
        return ans


if __name__ == '__main__':
    model = TempResNet(edict({
        'use_xlstm': False,
        'resnet': r'/home/drew/Desktop/Research/weights/resnet/resnet-18',
        'resnet_out': 512,
        'video_out': 128
    })).cuda()
    ans = model(torch.randn(32, 15, 64, 64, 3).cuda())
    print(ans)