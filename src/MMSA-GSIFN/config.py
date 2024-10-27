import json
import os
from pathlib import Path
import random
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
    elif not model_common_args['need_data_aligned'] and model_common_args['use_custom_data']:
        dataset_args = dataset_args['custom_unaligned']
    else:
        dataset_args = dataset_args['unaligned']

    enhance_net_args = {}
    if dataset_name not in ['sims', 'simsv2']:
        model_common_args['need_data_enhancement'] = False
    elif cmd_args.use_embedding != 1:
        en_net = cmd_args.enhance_net
        if isinstance(cmd_args.enhance_net, str):
            temp = []
            for elem in en_net:
                if elem in {'0', '1', '2', '3'}:
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
        elif en_version == 3:
            enhance_net_args = {'enhance_net_args':{
                'split_rate':enhance_net_args['enhancement_split'],
                'version': 'v3',
                'hyper_params':enhance_net_args['v3']
            }}

    va_embedding_args = {}
    if cmd_args.use_embedding == 1:
        model_common_args['need_data_enhancement'] = False
        model_common_args['need_va_embeddings'] = True
        va_embedding_args = config_all['VAEmbeddingsParams']

    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config.update(dataset_args)
    config.update(model_common_args)
    config.update(model_dataset_args)
    config.update(enhance_net_args)
    config.update(va_embedding_args)
    config['featurePath'] = os.path.join(config_all['datasetCommonParams']['dataset_root_dir'], config['featurePath'])
    config = edict(config) # use edict for backward compatibility with MMSA v1.0

    maybe_use_transformers = config.get('transformers', None)
    
    if maybe_use_transformers is None:
        no_transformers_key = True
        use_bert = config.get('use_bert', None)
        if maybe_use_transformers:
            config['transformers'] = 'bert'
    else:
        no_transformers_key = False
    if (maybe_use_transformers is not None or  
       (maybe_use_transformers is None and no_transformers_key and use_bert)):
        pretrained_weight_root = config_all['pretrainedWeights']['weights_root_dir']
        if not no_transformers_key:
            if config['transformers'] not in [[], '']:
                config['weight_dir'] = os.path.join(pretrained_weight_root, config['transformers'], config['pretrained'])
            else:
                config['weight_dir'] = os.path.join(pretrained_weight_root, config['pretrained'])
        else:
            if use_bert:
                config['weight_dir'] = os.path.join(pretrained_weight_root, 'bert', config['pretrained'])
            else:
                config['weight_dir'] = os.path.join(pretrained_weight_root, config['pretrained'])
        if model_name == 'bm_mag_m':
            config['bert_path'] = os.path.join(pretrained_weight_root, 'bert/bert-base-uncased')
        weight_dir = config['weight_dir']
        if ((no_transformers_key and use_bert) or 
            (maybe_use_transformers is not None and maybe_use_transformers == 'bert')):
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


def get_config_tune(
    model_name: str, dataset_name: str, config_file: str = "",
    random_choice: bool = True
) -> dict:
    """
    Get the tuning config of given dataset and model from config file.

    Parameters:
        model_name: Name of model.
        dataset_name: Name of dataset.
        config_file: Path to config file, if given an empty string, will use default config file.
        random_choice: If True, will randomly choose a config from the list of configs.

    Returns:
        config (dict): config of the given dataset and model
    """
    if config_file == "":
        config_file = Path(__file__).parent / "config" / "config_tune.json"
    with open(config_file, 'r') as f:
        config_all = json.load(f)
    model_common_args = config_all[model_name]['commonParams']
    model_dataset_args = config_all[model_name]['datasetParams'][dataset_name] if 'datasetParams' in config_all[model_name] else {}
    model_debug_args = config_all[model_name]['debugParams']
    dataset_args = config_all['datasetCommonParams'][dataset_name]
    # use aligned feature if the model requires it, otherwise use unaligned feature
    dataset_args = dataset_args['aligned'] if (model_common_args['need_data_aligned'] and 'aligned' in dataset_args) else dataset_args['unaligned']

    # random choice of args
    if random_choice:
        for item in model_debug_args['d_paras']:
            if type(model_debug_args[item]) == list:
                model_debug_args[item] = random.choice(model_debug_args[item])
            elif type(model_debug_args[item]) == dict: # nested params, 2 levels max
                for k, v in model_debug_args[item].items():
                    model_debug_args[item][k] = random.choice(v)

    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config.update(dataset_args)
    config.update(model_common_args)
    config.update(model_dataset_args)
    config.update(model_debug_args)
    config['featurePath'] = os.path.join(config_all['datasetCommonParams']['dataset_root_dir'], config['featurePath'])
    

    config = edict(config) # use edict for backward compatibility with MMSA v1.0

    return config


def get_config_all(config_file: str) -> dict:
    """
    Get all default configs. This function is used to export default config file. 
    If you want to get config for a specific model, use "get_config_regression" or "get_config_tune" instead.

    Parameters:
        config_file: "regression" or "tune"
    
    Returns:
        config: all default configs
    """
    if config_file == "regression":
        config_file = Path(__file__).parent / "config" / "config_regression.json"
    elif config_file == "tune":
        config_file = Path(__file__).parent / "config" / "config_tune.json"
    else:
        raise ValueError("config_file should be 'regression' or 'tune'")
    with open(config_file, 'r') as f:
        config_all = json.load(f)
    return edict(config_all)

def get_citations() -> dict:
    """
    Get paper titles and citations for models and datasets.

    Returns:
        cites (dict): {
            models: {
                tfn: {
                    title: "xxx",
                    paper_url: "xxx",
                    citation: "xxx",
                    description: "xxx"
                },
                ...
            },
            datasets: {
                ...
            },
        }
    """
    # TODO: add citations
    config_file = Path(__file__).parent / "config" / "citations.json"
    with open(config_file, 'r') as f:
        cites = json.load(f)
    return cites