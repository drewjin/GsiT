__all__ = ['XLSTMCFG']

class XLSTMCFG:
    def __init__(
        self,
        context_length,
        in_embed,
        out_embed,
        num_blocks=8,
        lstm_dropout=0.1, 
    ):
        self.context_length = context_length
        self.in_embed = in_embed
        self.out_embed = out_embed
        self.num_blocks = num_blocks
        self.lstm_dropout = lstm_dropout

    