"""
ATIO -- All Trains in One
"""
from .custom import *
from .multiTask import *
from .singleTask import *
from .missingTask import *

__all__ = ['ATIO']

class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            # single-task
            'tfn': TFN,
            'lmf': LMF,
            'mfn': MFN,
            'ef_lstm': EF_LSTM,
            'lf_dnn': LF_DNN,
            'graph_mfn': Graph_MFN,
            'mult': MULT,
            'mctn': MCTN,
            'bert_mag':BERT_MAG,
            'misa': MISA,
            'mfm': MFM,
            'mmim': MMIM,
            'cenet':CENET,
            # multi-task
            'mtfn': MTFN,
            'mlmf': MLMF,
            'mlf_dnn': MLF_DNN,
            'self_mm': SELF_MM,
            'tetfn': TETFN,
            # missing-task
            'tfr_net': TFR_NET,
            # custom
            'gsifn': GSIFN
        }
    
    def getTrain(self, args, trainer=None):
        # if args.use_accelerate:
        #     return self.TRAIN_MAP[args['model_name']](args, trainer)
        return self.TRAIN_MAP[args['model_name']](args)
