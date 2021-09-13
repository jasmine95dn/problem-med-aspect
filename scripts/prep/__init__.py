# -*- coding: utf8 -*-
from scripts.prep.reformatter import Reformat2010, Reformat2012
from scripts.nn.config import PrepConfig


class ReformatProcessor:
    def __init__(self, prep_config: PrepConfig, year='2012'):
        """

        :param prep_config:
        :param year:
        """
        self.reformatter = Reformat2012(prep_config) if year == '2012' else Reformat2010(prep_config)

    def __call__(self):
        """

        :return:
        """
        self.reformatter()