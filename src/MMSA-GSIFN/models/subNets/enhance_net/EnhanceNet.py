import sys 
sys.path.append('/home/drew/Desktop/Research/MMSA/src/MMSA/models/subNets/enhance_net')

import torch
import torch.nn as nn

from modules import (
    DecomposeEnhanceBlock, 
    DecomposeAttention,
    Decomposer,
    BaseEnhanceBlock, 
    SimpleEnhanceBlock,
    LateFFN
)

__all__ = ['EnhanceNet_v1', 'EnhanceNet_v2', 'EnhanceNet_v3']


class EnhanceNet_v1(nn.Module):
    """
        EnhanceNet_v1: Using only Decomposer and DecomposeAttention(none Multi-Head version or Single-Head version)
    """
    def __init__(self, args, params) -> None:
        super(EnhanceNet_v1, self).__init__()
        split_rate   = args.get('split_rate', None)
        self.params  = params
        self.version = 'v1'

        self.v_sr   = split_rate['vision']
        self.a_sr   = split_rate['audio']

        self.v_dec  = Decomposer(args, self.v_sr)
        self.a_dec  = Decomposer(args, self.a_sr)

        self.v_dec_attn = DecomposeAttention(params, self.version, self.v_sr)
        self.a_dec_attn = DecomposeAttention(params, self.version, self.a_sr)
    
    def forward(self, vision, audio):
        split_v, split_a = self.v_dec(vision), self.a_dec(audio)
        enhanced_v, enhanced_a = self.v_dec_attn(split_v), self.a_dec_attn(split_a)
        return enhanced_v, enhanced_a


class EnhanceNet_v2(nn.Module):
    """
        EnhanceNet_v2:  
        - Decomposer
        - DecomposeAttention(Multi-Head version)
        - GlobalMultiHeadAttention and ResidualConnection
    """
    def __init__(self, args, params) -> None:
        super(EnhanceNet_v2, self).__init__()
        split_rate   = args.get('split_rate', None)
        self.params  = params
        self.version = 'v2'
        if params['use_decomposition']: 
            self.dec_en_net = DecomposeEnhanceBlock(params['decompose'], split_rate, self.version)
        if params['use_early_enhance']:
            self.early_en_net = BaseEnhanceBlock(params['early_enhance'], split_rate)
            self.late_ffns = LateFFN(params['late_ffn'], sum(split_rate['vision']), sum(split_rate['audio']))
        if params['use_late_enhance']:
            self.late_en_net = BaseEnhanceBlock(params['late_enhance'], split_rate)
        if params['global_residual']:
            self.glb_ffns = LateFFN(params['late_ffn'], sum(split_rate['vision']), sum(split_rate['audio']))

    def forward(self, vision, audio):
        glb_res = self.params.get('global_residual', False)
        early = self.params.get('use_early_enhance', True)
        late = self.params.get('use_late_enhance', True)
        dec = self.params.get('use_decomposition', True)

        if glb_res:
            res_org_v, res_org_a = vision, audio
        
        if dec:
            O_dec_v, O_dec_a = self.dec_en_net(vision, audio)
            if late:
                O_l_v, O_l_a = self.late_en_net(O_dec_v, O_dec_a)
                if early:
                    O_e_v, O_e_a = self.early_en_net(vision, audio) 
                    O_v, O_a = self.late_ffns(O_l_v + O_e_v, O_l_a + O_e_a)
                    if glb_res:
                        O_v, O_a = self.glb_ffns(O_v + res_org_v, O_a + res_org_a)
                else:
                    if glb_res:
                        O_v, O_a = self.glb_ffns(O_l_v + res_org_v, O_l_a + res_org_a)
                    else:
                        O_v, O_a = O_l_v, O_l_a
        else:
            if late:
                O_l_v, O_l_a = self.late_en_net(vision, audio)
                if early:
                    O_e_v, O_e_a = self.early_en_net(vision, audio) 
                    O_v, O_a = self.late_ffns(O_l_v + O_e_v, O_l_a + O_e_a)
                    if glb_res:
                        O_v, O_a = self.glb_ffns(O_v + res_org_v, O_a + res_org_a)
                else:
                    if glb_res:
                        O_v, O_a = self.glb_ffns(O_l_v + res_org_v, O_l_a + res_org_a)
                    else:
                        O_v, O_a = O_l_v, O_l_a
            else:
                O_e_v, O_e_a = self.early_en_net(vision, audio)
                if glb_res:
                    O_v, O_a = self.glb_ffns(O_e_v + res_org_v, O_e_a + res_org_a)
                else:
                    O_v, O_a = O_e_v, O_e_a
        return O_v, O_a


class EnhanceNet_v3(nn.Module):
    def __init__(self, args, params) -> None:
        super(EnhanceNet_v3, self).__init__()
        split_rate   = args.get('split_rate', None)
        self.params  = params
        self.version = 'v3'
        self.dec_en_net = DecomposeEnhanceBlock(params['decompose'], split_rate, self.version)
        self.simple_en_net = SimpleEnhanceBlock(params['simple_enhance'], split_rate)
        self.lffn1 = LateFFN(params['late_ffn'], sum(split_rate['vision']), sum(split_rate['audio']))
        self.lffn2 = LateFFN(params['late_ffn'], sum(split_rate['vision']), sum(split_rate['audio']))
    
    def forward(self, X_v, X_a):
        res_X_v, res_X_a = X_v, X_a
        X_dec_v, X_dec_a = self.dec_en_net(X_v, X_a)
        X_s_v, X_s_a = self.simple_en_net(X_v, X_a)
        H_v, H_a = self.lffn1(X_dec_v + X_s_v, X_dec_a + X_s_a)
        O_v, O_a = self.lffn2(H_v + res_X_v, H_a + res_X_a)
        return O_v, O_a

if __name__ == '__main__':
    import os 
    import json
    
    from pathlib import Path
    from easydict import EasyDict as edict
    from argparse import Namespace
    def get_config_regression(
        model_name: str, dataset_name: str, config_file: str = "", cmd_args: Namespace = None,
    ) -> dict:
        """
        Get the regression config of given dataset and model from config file.

        Parameters:
            model_name: Name of model.
            dataset_name: Name of dataset.
            config_file: Path to config file, if given an empty string, will use default config file.

        Returns:
            config (dict): config of the given dataset and model
        """
        if config_file == "":
            config_file = Path(__file__).parent / "config" / "config_regression.json"
        with open(config_file, 'r') as f:
            config_all = json.load(f)
        model_common_args = config_all[model_name]['commonParams']
        model_dataset_args = config_all[model_name]['datasetParams'][dataset_name]
        dataset_args = config_all['datasetCommonParams'][dataset_name]
        # use aligned feature if the model requires it, otherwise use unaligned feature
        if model_common_args['need_data_aligned'] and 'aligned' in dataset_args:
            dataset_args = dataset_args['aligned']
        else:
            dataset_args = dataset_args['unaligned']

        enhance_net_args = {}
        if dataset_name not in ['sims', 'simsv2']:
            dataset_args['need_data_enhancement'] = False
        else:
            en_net = cmd_args.enhance_net
            if isinstance(cmd_args.enhance_net, str):
                temp = []
                for elem in en_net:
                    if elem in {'0', '1', '2'}:
                        temp.append(int(elem))
                en_net = temp
            model_common_args['need_data_enhancement'] = bool(en_net[0])
            enhance_net_args = config_all['enhanceNetParams']
            model_common_args['enhance_net_version'] = en_version = en_net[1]
            if en_version == 1:
                enhance_net_args = {'enhance_net_args':{
                    'split_rate':enhance_net_args['enhancement_split'],
                    'version': 'v1',
                    'hyper_params':enhance_net_args['v1']
                }}
            elif en_version == 2:
                enhance_net_args = {'enhance_net_args':{
                    'split_rate':enhance_net_args['enhancement_split'],
                    'version': 'v2',
                    'hyper_params':enhance_net_args['v2']
                }}

        config = {}
        config['model_name'] = model_name
        config['dataset_name'] = dataset_name
        config.update(dataset_args)
        config.update(model_common_args)
        config.update(model_dataset_args)
        config.update(enhance_net_args)
        config['featurePath'] = os.path.join(config_all['datasetCommonParams']['dataset_root_dir'], config['featurePath'])
        config = edict(config) # use edict for backward compatibility with MMSA v1.0

        maybe_use_transformers = config.get('transformers', None)
        if maybe_use_transformers is not None:
            pretrained_weight_root = config_all['pretrainedWeights']['weights_root_dir']
            if config['transformers'] not in [None, [], '']:
                config['weight_dir'] = os.path.join(pretrained_weight_root, config['transformers'], config['pretrained'])
            else:
                config['weight_dir'] = os.path.join(pretrained_weight_root, config['pretrained'])

            weight_dir = config['weight_dir']
            if config['transformers'] == 'bert':
                weight_dir = weight_dir.split('/')
                cn_bert = 'bert-base-chinese'
                en_bert = 'bert-base-uncased'
                if config['language']== 'cn' and weight_dir[-1] != cn_bert:
                    weight_dir[-1] = cn_bert
                    weight_dir = '/'.join(weight_dir)
                    config['weight_dir'] = weight_dir 
                elif config['language'] == 'en' and weight_dir[-1] != en_bert:
                    weight_dir[-1] = en_bert
                    weight_dir = '/'.join(weight_dir)
                    config['weight_dir'] = weight_dir 

        return config
    
    model = 'mult'
    dataset = 'sims'
    cfg = '/home/drew/Desktop/Research/MMSA/src/MMSA/config/config_regression.json'
    args = get_config_regression(model, dataset, cfg)
    
    vision, audio = torch.randn(16, 50, 709), torch.randn(16, 50, 33)
    enargs = args.get('enhance_net_args', None)
    version = enargs['version']
    params = enargs['hyper_params']
    
    ENHANCENET_MAP = {'v1': EnhanceNet_v1, 'v2': EnhanceNet_v2, 'v3': EnhanceNet_v3}
    enhanceNet = ENHANCENET_MAP[version](enargs, params)
    output = enhanceNet(vision, audio)