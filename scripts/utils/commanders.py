# -*- coding: utf8 -*-
import scripts.nn as nn
import scripts.prep as prep
import scripts.eval.evaluators as evaluators
import scripts.eval.plotters as plotters


def run_prep(prepconfig, year='2012'):
    """

    :param prepconfig:
    :param year:
    :return:
    """
    processor = prep.ReformatProcessor(prepconfig, year=year)
    processor()


def run_mod(model_config, embedding_config, mod='modpol', sent_state='within', mode='train', info='start'):
    """

    :param model_config:
    :param embedding_config:
    :param mod:
    :param sent_state:
    :param mode:
    :param info:
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

    processor(mode=mode, info=info)


def run_eval(**kwargs):
    evaluators.evaluate_error(**kwargs)


def run_plot(**kwargs):
    plotters.cf_matrix_plot(**kwargs)