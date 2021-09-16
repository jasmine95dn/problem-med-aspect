# -*- coding: utf8 -*-

import scripts.nn.processors as processors
from scripts.nn.config import EmbeddingConfig, ModelConfig
from scripts.nn.models import GRUNet, BiLSTMNet, BertNet


class ModPolProcessor:
    def __init__(self, model_config: ModelConfig, embedding_config: EmbeddingConfig):
        """

        :param model_config:
        :param embedding_config:
        """
        self.processor = processors.ModelProcessor(model_config=model_config, model=GRUNet,
                                                    embedding_config=embedding_config)

    def __call__(self, mode='train', cond=False):
        """

        :param mode:
        :param cond:
        :return:
        """
        self.processor(mode, cond)


class RelaProcessor:
    def __init__(self, model_config: ModelConfig, embedding_config: EmbeddingConfig, within_mode: bool = True):
        """

        :param model_config:
        :param embedding_config:
        :param within_mode:
        """
        if within_mode:
            self.processor = processors.ModelProcessor(model_config=model_config, model=BiLSTMNet,
                                                       embedding_config=embedding_config)
        else:
            self.processor = processors.ModelProcessor(model_config=model_config, model=BertNet,
                                                       embedding_config=embedding_config)

    def __call__(self, mode='train', cond=False):
        """

        :param mode:
        :param cond:
        :return:
        """
        self.processor(mode, cond)


