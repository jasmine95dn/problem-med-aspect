# -*- coding: utf8 -*-

import scripts.nn.processors as processors
import scripts.nn.config as config
from scripts.nn.models import GRUNet, BiLSTMNet, BertNet


class ModPolProcessor:
    def __init__(self, model_config, embedding_config, never_split=None):
        never_split = ['<p>', '</p>'] if not never_split
        self.processor = processors.ModelProcessor(model_config=model_config, model=GRUNet,
                                              embedding_config=embedding_config, never_split=never_split)

    def __call__(self, mode='train', info='start', train_model_path=None, save_path=None, infer_save_path=None):
        self.processor(mode, info, train_model_path, save_path, infer_save_path)

class RelaProcessor:
    def __init__(self, model_config, embedding_config, modtype='within'):

        if modtype == 'within':
            self.processor = processors.ModelProcessor(model_config=model_config, model=BiLSTMNet,
                                                       embedding_config=embedding_config)
        else:
            self.processor = processors.ModelProcessor(model_config=model_config, model=BertNet,
                                                       embedding_config=embedding_config)
    def __call__(self, mode='train', info='start', train_model_path=None, save_path=None, infer_save_path=None):
        self.processor(mode, info, train_model_path, save_path, infer_save_path)


