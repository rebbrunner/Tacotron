import configparser, ast, importlib

import numpy as np
import toml
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import betabinom

class Tacotron(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init()
        self.input_size = 2 * decoder['input_size']
        self.attn_rnn_size = decoder['attn_rnn_size']
        self.decoder_rnn_size = decoder['decoder_rnn_size']
        self.n_mels = decoder['n_mels']
        self.reduction_factor = decoder['reduction_factor']

        # self.encoder = Encoder(**encoder)
        # self.decoder_cell = DecoderCell
    
    @classmethod
    def from_pretrained(cls, path, map_location=None, cfg_path='./taco_config.ini'):
        with open(cfg_path) as f:
            cfg = toml.load(f)
        checkpoint = torch.load(path)
        model = cls(**cfg['model'])
        model.load_state_dict(checkpoint['tacotron'])
        model.eval()
        return model
