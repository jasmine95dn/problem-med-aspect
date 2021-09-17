# -*- coding: utf8 -*-
import scripts.nn as nn
import scripts.prep as prep
import scripts.eval as evaluator


def run_prep(prepconfig, year='2012'):
    """

    :param prepconfig:
    :param year:
    :return:
    """
    processor = prep.ReformatProcessor(prepconfig, year=year)
    processor()


def run_mod(model_config, embedding_config, mod='modpol', sent_state='within', mode='train', cond=False):
    """

    :param model_config:
    :param embedding_config:
    :param mod:
    :param sent_state:
    :param mode:
    :param cond:
    :return:
    """
    processor = None
    if mod == 'modpol':
        print('Run ModPolProcessor, GRU classifier')
        processor = nn.ModPolProcessor(model_config, embedding_config)
    elif mod == 'rela':
        within_mode = True if sent_state == 'within' else False
        print(f'Run RelaProcessor, {sent_state} model')
        processor = nn.RelaProcessor(model_config, embedding_config, within_mode=within_mode)

    processor(mode=mode, cond=cond)


def run_plot(config):
    """

    :param config:
    :return:
    """
    evaluator.cf_matrix(config)
