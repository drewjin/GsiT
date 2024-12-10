"""
AMIO -- All Model in One
"""
import torch.nn as nn

from .custom import *
from .multiTask import *
from .singleTask import *
from .missingTask import *
from .subNets import AlignSubNet
from .subNets import EnhanceNet_v1, EnhanceNet_v2, EnhanceNet_v3, VAEmbeddings
from pytorch_transformers import BertConfig

class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        self.args = args
        self.MODEL_MAP = {
            # single-task
            'tfn': TFN,
            'lmf': LMF,
            'mfn': MFN,
            'ef_lstm': EF_LSTM,
            'lf_dnn': LF_DNN,
            'graph_mfn': Graph_MFN,
            'mctn': MCTN,
            'bert_mag': BERT_MAG,
            'mult': MULT,
            'misa': MISA,
            'mfm': MFM,
            'mmim': MMIM,
            'cenet': CENET,
            # multi-task
            'mtfn': MTFN,
            'mlmf': MLMF,
            'mlf_dnn': MLF_DNN,
            'self_mm': SELF_MM,
            'tetfn': TETFN,
            # missing-task
            'tfr_net': TFR_NET,
            # custom
            'gsit': GsiT
        }
        self.need_model_aligned = args.get('need_model_aligned', None)
        # simulating word-align network (for seq_len_T == seq_len_A == seq_len_V)
        if self.need_model_aligned:
            self.alignNet = AlignSubNet(args, 'avg_pool')
            if 'seq_lens' in args.keys():
                args['seq_lens'] = self.alignNet.get_seq_len()
        self.model_name = args['model_name']
        lastModel = self.MODEL_MAP[self.model_name]

        self.need_data_enhancement = args.get('need_data_enhancement', None)
        if self.need_data_enhancement:
            enargs = args.get('enhance_net_args', None)
            self.version = version = enargs['version']
            params = enargs['hyper_params']
            ENHANCENET_MAP = {'v1': EnhanceNet_v1, 'v2': EnhanceNet_v2, 'v3': EnhanceNet_v3}
            self.enhanceNet = ENHANCENET_MAP[version](enargs, params)
        
        self.need_va_embeddings = args.get('need_va_embeddings', None)
        if self.need_va_embeddings:
            self.a_embedding = VAEmbeddings(args, args.feature_dims[1], args.a_out_feature_dim)
            self.v_embedding = VAEmbeddings(args, args.feature_dims[2], args.v_out_feature_dim)
            args.feature_dims[1] = args.a_out_feature_dim
            args.feature_dims[2] = args.v_out_feature_dim

        if args.model_name == 'cenet':
            config = BertConfig.from_pretrained(args.weight_dir, num_labels=1, finetuning_task='sst')
            self.Model = CENET.from_pretrained(args.weight_dir, config=config, pos_tag_embedding=True, senti_embedding=True, polarity_embedding=True, args=args)
        else:
            self.Model = lastModel(args)

    def forward(self, text_x, audio_x, video_x, mode=['TRAIN',None], *args, **kwargs):
        if self.model_name == 'bm_mag_m':
            mamba_text_x, text_x = text_x
        if self.need_model_aligned:
            text_x, audio_x, video_x = self.alignNet(text_x, audio_x, video_x)
        if self.need_data_enhancement:
            if mode[0] == 'TEST':
                import os 
                import warnings
                import numpy as np
                import seaborn as sns
                import matplotlib.pyplot as plt

                from os.path import join

                warnings.filterwarnings(
                    "ignore", 
                    category=RuntimeWarning, 
                    message="invalid value encountered in divide")
                sample = 0
                temp_video_x, temp_audio_x = video_x.clone(), audio_x.clone()

            if self.args['model_name'] in ['self_mm', 'mmim', 'tetfn', 'tfr_net']:
                video_ex, audio_ex = self.enhanceNet(video_x[0], audio_x[0])
                video_x, audio_x = tuple([video_ex] + list(video_x[1:])), tuple([audio_ex] + list(audio_x[1:]))
            else:
                video_x, audio_x = self.enhanceNet(video_x, audio_x)

            if mode[0] == 'TEST':
                v_corr_mat_processed = np.corrcoef(video_x[sample].cpu(), rowvar=False)
                a_corr_mat_processed = np.corrcoef(audio_x[sample].cpu(), rowvar=False)    
                v_corr_mat_org = np.corrcoef(temp_video_x[sample].cpu(), rowvar=False)
                a_corr_mat_org = np.corrcoef(temp_audio_x[sample].cpu(), rowvar=False)
                _, axs = plt.subplots(2, 2, figsize=(15, 10))
                sns.heatmap(v_corr_mat_org, cmap='YlGnBu', square=True, cbar=False, ax=axs[0][0])
                axs[0][0].set_title('Original Video Correlation Coefficients')

                sns.heatmap(a_corr_mat_org, cmap='YlGnBu', square=True, cbar=False, ax=axs[0][1])
                axs[0][1].set_title('Original Audio Correlation Coefficients')

                sns.heatmap(v_corr_mat_processed, cmap='YlGnBu', square=True, cbar=False, ax=axs[1][0])
                axs[1][0].set_title('Processed Video Correlation Coefficients')

                sns.heatmap(a_corr_mat_processed, cmap='YlGnBu', square=True, cbar=False, ax=axs[1][1])
                axs[1][1].set_title('Processed Audio Correlation Coefficients')

                plt.tight_layout()

                img_path = join('results', 'imgs', self.args['dataset_name'], self.args["model_name"])
                if not os.path.exists(img_path):
                    os.makedirs(img_path)
                plt.savefig(join(img_path, f'{self.version}_{mode[0]}_{mode[1]}.png'))
        if self.need_va_embeddings:
            audio_x = self.a_embedding(audio_x)
            video_x = self.v_embedding(video_x)
        if self.model_name == 'bm_mag_m':
            text_x = (mamba_text_x, text_x)
        return self.Model(text_x, audio_x, video_x, *args, **kwargs)
