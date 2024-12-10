import os 
import sys
from os.path import join
workplace_root = join(*os.path.abspath(os.path.join(os.path.dirname(__file__))).split(os.sep)[:-2])
src_root = join(workplace_root, 'src', 'MMSA')
sys.path.append(workplace_root)
sys.path.append(src_root)

import argparse

from MMSA.run import MMSA_run

seeds = [
    128
    # 114514
    # , 12, 712, 827, 1347, 99, 8, 24
]

def parse_args():    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-m', '--model', type=str, default='gsit', help='Name of model',
                        choices=['lf_dnn', 'ef_lstm', 'tfn', 'mctn','lmf', 'mfn', 'graph_mfn', 'mult', 'bert_mag', 
                                 'misa', 'mfm', 'mlf_dnn', 'mtfn', 'mlmf', 'self_mm', 'mmim','tfr_net','tetfn','cenet',
                                 'bm_mag_m', 'mult_another', 'mmml', 'gsgnet', 'gsit'])
    parser.add_argument('-d', '--dataset', type=str, default='sims',
                        choices=['sims', 'mosi', 'mosei', 'simsv2'], help='Name of dataset')
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='Path to config file. If not specified, default config file will be used.')
    parser.add_argument('-t', '--tune', action='store_true',
                        help='Whether to tune hyper parameters. Default: False')
    parser.add_argument('-tt', '--tune-times', type=int, default=50,
                        help='Number of times to tune hyper parameters. Default: 50')
    # seeds = [129]
    # if parser.parse_args().model == 'cmgformer':
    #     seeds = [128]
    parser.add_argument('-s', '--seeds', action='append', type=int, default=seeds,
                        help='Random seeds. Specify multiple times for multiple seeds. Default: [1111, 1112, 1113, 1114, 1115]')
    parser.add_argument('-n', '--num-workers', type=int, default=20,
                        help='Number of workers used to load data. Default: 4')
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        help='Verbose level of stdout. 0 for error, 1 for info, 2 for debug. Default: 1')
    parser.add_argument('--model-save-dir', type=str, default='saved_models',
                        help='Path to save trained models. Default: "~/MMSA/saved_models"')
    parser.add_argument('--res-save-dir', type=str, default='results',
                        help='Path to save csv results. Default: "~/MMSA/results"')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Path to save log files. Default: "~/MMSA/logs"')
    parser.add_argument('-g', '--gpu-ids', action='append', default=[0],
                        help='Specify which gpus to use. If an empty list is supplied, will automatically assign to the most memory-free gpu. \
                              Currently only support single gpu. Default: []')
    parser.add_argument('-Ft', '--feature-T', type=str, default='',
                        help='Path to custom text feature file. Default: ""')
    parser.add_argument('-Fa', '--feature-A', type=str, default='',
                        help='Path to custom audio feature file. Default: ""')
    parser.add_argument('-Fv', '--feature-V', type=str, default='',
                        help='Path to custom video feature file. Default: ""')
    parser.add_argument('-E', '--enhance-net', type=str, default=[0, 1])
    parser.add_argument('-embed', '--use-embedding', type=int, default=0)
    
    return parser.parse_args()


if __name__ == '__main__':
    cmd_args = parse_args()
    MMSA_run(
        model_name=cmd_args.model,
        dataset_name=cmd_args.dataset,
        config_file=cmd_args.config,
        seeds=cmd_args.seeds,
        is_tune=cmd_args.tune,
        tune_times=cmd_args.tune_times,
        feature_T=cmd_args.feature_T,
        feature_A=cmd_args.feature_A,
        feature_V=cmd_args.feature_V,
        model_save_dir=cmd_args.model_save_dir,
        res_save_dir=cmd_args.res_save_dir,
        log_dir=cmd_args.log_dir,
        gpu_ids=cmd_args.gpu_ids,
        num_workers=cmd_args.num_workers,
        verbose_level=cmd_args.verbose,
        cmd_args=cmd_args
    )
