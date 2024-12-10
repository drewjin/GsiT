import torch
import torch.nn as nn
import torch.nn.functional as F

from thop import profile
from ptflops import get_model_complexity_info
from torch_operation_counter import OperationsCounterMode
from easydict import EasyDict
from transformers import (
    Data2VecAudioModel, 
    BertModel
)

from .modules import (CrossModalGraph,)


__all__ = ['GsiT']
    

class GsiT(nn.Module):
    def __init__(self, args):
        super(GsiT, self).__init__()
        self_super_cfg = args.self_super_cfg
        cmg_cfg = args.cmg_cfg
        # TEXT TOKENIZITION
        self.aligned = args.need_data_aligned

        # BERT MODEL
        with torch.no_grad():
            self.bert_model = BertModel.from_pretrained(args.weight_dir)
        self.text_dropout = cmg_cfg.text_dropout

        audio_temp_dim_in  = self_super_cfg.a_feat_in
        audio_temp_dim_out = self_super_cfg.a_feat_out

        video_temp_dim_in  = self_super_cfg.v_feat_in
        video_temp_dim_out = self_super_cfg.v_feat_out

        # TEMPORAL CONVOLUTION LAYERS
        proj_dim = cmg_cfg.dst_feature_dim_nheads[0]
        self.proj_l = nn.Conv1d(
            cmg_cfg.text_out, proj_dim, 
            kernel_size=cmg_cfg.conv1d_kernel_size_l, 
            padding=0, bias=False
        )
        self.proj_a = nn.Conv1d(
            audio_temp_dim_in, proj_dim, 
            kernel_size=cmg_cfg.conv1d_kernel_size_a, 
            padding=0, bias=False
        )
        self.proj_v = nn.Conv1d(
            video_temp_dim_in, proj_dim, 
            kernel_size=cmg_cfg.conv1d_kernel_size_v, 
            padding=0, bias=False
        )

        # GRAPH MODAL GRAPH
        self.cross_modal_graph = CrossModalGraph(args)

        # xLSTMs
        seq_lens = self_super_cfg.inter_seq_lens
        a_len, v_len = seq_lens[1], seq_lens[2]
        a_kernel = cmg_cfg.conv1d_kernel_size_a
        v_kernel = cmg_cfg.conv1d_kernel_size_v
        a_conv_len = int((a_len - a_kernel) / 1 + 1)
        v_conv_len = int((v_len - v_kernel) / 1 + 1)

        # SELF SUPERVISION ON ENCODED FEATURES
        # the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(6*proj_dim, proj_dim)
        # self.post_fusion_layer_1 = nn.Linear(self_super_cfg.text_embed + 2*proj_dim, args.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(proj_dim, 1)
        # self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, 1)


    def forward(self, text, audio, video, plt=None):
        audio_feat, audio_lengths = audio
        video_feat, video_lengths = video

        input_ids, input_mask, segment_ids = text
        text = self.bert_model(input_ids=input_ids, attention_mask=input_mask)

        text_embedding = text.last_hidden_state
        text_pooler_output = text.pooler_output

        x_l = F.dropout(
            text_embedding.transpose(1, 2), 
            p=self.text_dropout, training=self.training
        )
        # x_a = audio_d2v.last_hidden_state.transpose(1, 2)
        # x_v = video_res.hidden_states.transpose(1, 2)
        x_a = audio_feat.transpose(1, 2)
        x_v = video_feat.transpose(1, 2)

        proj_x_l = self.proj_l(x_l).permute(2, 0, 1)
        proj_x_a = self.proj_a(x_a).permute(2, 0, 1)
        proj_x_v = self.proj_v(x_v).permute(2, 0, 1)

        cat_list = [
            proj_x_l, 
            proj_x_v, 
            proj_x_a
        ]
        # cat_list = [x.permute(2, 0, 1) for x in [x_l, x_v, x_a]]
        cat_seq = torch.cat(cat_list, dim=0)
        cat_split = self.get_seq_split(cat_list)

        fused_output = self.cross_modal_graph(cat_seq=cat_seq, split=cat_split, plot_map=plt)
        split_output = fused_output.split
        
        fusion_h = torch.concat([split.permute(1, 0, 2)[-1] for split in split_output], dim=-1)
        fusion_h = self.post_fusion_dropout(fusion_h)
        # flops_post_fusion_layer_1, param_post_fusion_layer_1 = self.param_flops(
        #     self.post_fusion_layer_1, fusion_h
        # )
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
        
        output_fusion = self.post_fusion_layer_2(fusion_h)

        res = EasyDict({
            'M': output_fusion,
            'Feature_f': fusion_h,
        })
        return res, fused_output, cat_split
    
    def get_seq_split(self, seq_list):
        return [seq.shape[0] for seq in seq_list]
    
    def param_flops(self, model, input):
        macs, _ = profile(model, inputs=(input,))
        param = self.count_params(model)
        return macs, param
    
    def count_params(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)