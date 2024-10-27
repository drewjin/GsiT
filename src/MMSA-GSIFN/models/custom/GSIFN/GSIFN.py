import torch
import torch.nn as nn
import torch.nn.functional as F

from thop import profile
from easydict import EasyDict
from transformers import (
    Data2VecAudioModel, 
    BertModel
)

from .modules import (
    TempResNet, 
    xLSTMProj, 
    NaiveLSTMProj, 
    GsiT,
    XLSTMCFG
)


__all__ = ['GSIFN']
    

class GSIFN(nn.Module):
    def __init__(self, args):
        super(GSIFN, self).__init__()
        self_super_cfg = args.self_super_cfg
        cmg_cfg = args.cmg_cfg
        # TEXT TOKENIZITION
        self.aligned = args.need_data_aligned

        # BERT MODEL
        # with torch.no_grad():
        self.bert_model = BertModel.from_pretrained(args.weight_dir)
        self.text_dropout = cmg_cfg.text_dropout
        # self.bert_embedding = self.bert_model.get_input_embeddings()

        # AUDIO EXTRACTION
        # Using Data2Vec to extract audio temporal features
        # self.audio_model = Data2VecAudioModel.from_pretrained(args.data2vec)
        # # Using LSTMs to extract audio features
        # self.use_xlstm = args.use_xlstm
        # if args.use_xlstm:
        #     self.audio_d2v_xlstm_proj = xLSTMProj(
        #         XLSTMCFG(
        #             context_length=self_super_cfg.inter_seq_lens[2],
        #             in_embed=self_super_cfg.a_d2v_in, 
        #             out_embed=self_super_cfg.a_feat_out, 
        #             num_blocks=args.a_lstm_layers, 
        #             lstm_dropout=args.a_lstm_dropout
        #         ))
        # else:
        #     # TODO
        #     self.audio_proj = NaiveLSTMProj(
        #         args.audio_embed, args.a_lstm_hidden_size, args.audio_out, 
        #         num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout
        #     )

        # VISION EXTRACTION
        # Using TempResNet to extract visual features
        # self.video_model = TempResNet(args)

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
        # seq_lens = self_super_cfg.inter_seq_lens
        # a_len, v_len = seq_lens[1], seq_lens[2]
        # self.proj_a = nn.Linear(audio_temp_dim_in, proj_dim)
        # self.audio_model = xLSTMProj(
        #     XLSTMCFG(
        #         context_length=a_len,
        #         in_embed=proj_dim,
        #         out_embed=audio_temp_dim_out,
        #         num_blocks=args.xlstm_block
        #     )
        # )
        # self.proj_v = nn.Linear(video_temp_dim_in, proj_dim)
        # self.video_model = xLSTMProj(
        #     XLSTMCFG(
        #         context_length=v_len,
        #         in_embed=proj_dim,
        #         out_embed=video_temp_dim_out,
        #         num_blocks=args.xlstm_block
        #     )
        # )

        # GRAPH MODAL GRAPH
        self.cross_modal_graph = GsiT(args)

        # xLSTMs
        seq_lens = self_super_cfg.inter_seq_lens
        a_len, v_len = seq_lens[1], seq_lens[2]
        a_kernel = cmg_cfg.conv1d_kernel_size_a
        v_kernel = cmg_cfg.conv1d_kernel_size_v
        a_conv_len = int((a_len - a_kernel) / 1 + 1)
        v_conv_len = int((v_len - v_kernel) / 1 + 1)
        self.audio_model = xLSTMProj(
            XLSTMCFG(
                context_length=a_conv_len,
                in_embed=proj_dim,
                out_embed=audio_temp_dim_out,
                num_blocks=args.xlstm_block
            )
        )
        self.video_model = xLSTMProj(
            XLSTMCFG(
                context_length=v_conv_len,
                in_embed=proj_dim,
                out_embed=video_temp_dim_out,
                num_blocks=args.xlstm_block
            )
        )
        # Ablation Study naive lstm
        # self.a_kernel = a_kernel
        # self.v_kernel = v_kernel
        # self.audio_model = NaiveLSTMProj(
        #     in_size=proj_dim,
        #     hidden_size=2*proj_dim,
        #     out_size=audio_temp_dim_out,
        #     num_layers=args.xlstm_block,
        #     dropout=args.a_lstm_dropout,
        #     bidirectional=True
        # )
        # self.video_model = NaiveLSTMProj(
        #     in_size=proj_dim,
        #     hidden_size=2*proj_dim,
        #     out_size=video_temp_dim_out,
        #     num_layers=args.xlstm_block,
        #     dropout=args.v_lstm_dropout,
        #     bidirectional=True
        # )

        # SELF SUPERVISION ON ENCODED FEATURES
        # the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(6*proj_dim, proj_dim)
        # self.post_fusion_layer_1 = nn.Linear(self_super_cfg.text_embed + 2*proj_dim, args.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(proj_dim, 1)
        # self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, 1)

        # the classify layer for text
        self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        self.post_text_layer_1 = nn.Linear(self_super_cfg.text_embed, args.post_text_dim)
        self.post_text_layer_2 = nn.Linear(args.post_text_dim, args.post_text_dim//2)
        self.post_text_layer_3 = nn.Linear(args.post_text_dim//2, 1)

        # the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.post_audio_layer_1 = nn.Linear(proj_dim, args.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(args.post_audio_dim, args.post_audio_dim//2)
        self.post_audio_layer_3 = nn.Linear(args.post_audio_dim//2, 1)

        # the classify layer for video
        self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        self.post_video_layer_1 = nn.Linear(proj_dim, args.post_video_dim)
        self.post_video_layer_2 = nn.Linear(args.post_video_dim, args.post_video_dim//2)
        self.post_video_layer_3 = nn.Linear(args.post_video_dim//2, 1)


    def forward(self, text, audio, video, plt=None):
        audio_feat, audio_lengths = audio
        video_feat, video_lengths = video

        # a_mask = torch.ones_like(audio_feat, dtype=torch.long)
        # for idx, true_len in enumerate(audio_lengths):
        #     a_mask[idx, int(true_len):] = 0

        input_ids, input_mask, segment_ids = text
        text = self.bert_model(input_ids=input_ids, attention_mask=input_mask)
        text_embedding = text.last_hidden_state
        text_pooler_output = text.pooler_output

        # if self.aligned:
        #     audio_d2v = self.audio_model(audio_wav)
        #     video_res = self.video_model(video_rgb)
        # else: 
        #     audio_d2v = self.audio_model(audio_wav, attention_mask=a_mask, output_hidden_states=True)
        #     video_res = self.video_model(video_rgb)

        # if self.use_xlstm:
        #     audio_d2v = self.audio_d2v_xlstm_proj(audio_d2v.last_hidden_state)
        # else:
        #     audio_d2v = self.audio_proj(audio_d2v.last_hidden_state, audio_lengths)

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
        # x_a_hs = self.audio_model(self.proj_a(audio_feat))
        # x_v_hs = self.video_model(self.proj_v(video_feat))
        # proj_x_a = x_a_hs.hidden_states.permute(1, 0, 2)
        # proj_x_v = x_v_hs.hidden_states.permute(1, 0, 2)
        # a_pooled = x_a_hs.pooler_output
        # v_pooled = x_v_hs.pooler_output

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
        
        # self.param_flops(self.cross_modal_graph, cat_seq, cat_split)

        a_pooled = self.audio_model(proj_x_a.permute(1, 0, 2)).pooler_output
        v_pooled = self.video_model(proj_x_v.permute(1, 0, 2)).pooler_output

        # self.param_flops(self.audio_model, proj_x_a.permute(1, 0, 2))
        # self.param_flops(self.video_model, proj_x_v.permute(1, 0, 2))

        # Ablation for lstm
        # a_pooled = self.audio_model(proj_x_a.permute(1, 0, 2)).pooler_output
        # v_pooled = self.video_model(proj_x_v.permute(1, 0, 2)).pooler_output

        # Ablation for Graph
        # split_output = [text_pooler_output, v_pooled, a_pooled]
        # fusion_h = torch.cat(split_output, dim=-1)
        
        fusion_h = torch.concat([split.permute(1, 0, 2)[-1] for split in split_output], dim=-1)
        fusion_h = self.post_fusion_dropout(fusion_h)
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
        # text
        text_h = self.post_text_dropout(text_pooler_output)
        text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)
        # audio
        audio_h = self.post_audio_dropout(a_pooled)
        audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
        # vision
        video_h = self.post_video_dropout(v_pooled)
        video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)

        # classifier-fusion
        # x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        # output_fusion = self.post_fusion_layer_3(x_f)
        output_fusion = self.post_fusion_layer_2(fusion_h)
        # classifier-text
        x_t = F.relu(self.post_text_layer_2(text_h), inplace=False)
        output_text = self.post_text_layer_3(x_t)
        # classifier-audio
        x_a = F.relu(self.post_audio_layer_2(audio_h), inplace=False)
        output_audio = self.post_audio_layer_3(x_a)
        # classifier-vision
        x_v = F.relu(self.post_video_layer_2(video_h), inplace=False)
        output_video = self.post_video_layer_3(x_v)

        res = EasyDict({
            'M': output_fusion, 
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            # 'T': output_fusion,
            # 'A': output_fusion,
            # 'V': output_fusion,
            # 'Feature_t': fusion_h,
            # 'Feature_a': fusion_h,
            # 'Feature_v': fusion_h,
            'Feature_f': fusion_h,
        })
        return res, fused_output, cat_split
    
    def get_seq_split(self, seq_list):
        return [seq.shape[0] for seq in seq_list]
    
    def param_flops(self, model, cat_seq):
        macs, params = profile(model, inputs=(cat_seq,))
        print(f"FLOPs: {macs / 1e9:.3f} G ({macs}), Params: {params / 1e6:.3f} M ({params})")