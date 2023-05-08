# --*-- coding: utf-8 --*--
"""
This module defines some elements for data
"""
import re
import os
import torch
import json
import pandas as pd


class Entity:
    """
    Class Entity contains cond of an entity, including an entity, its pre-entity (left part)
    and post-entity (right part) in a sentence

    Args:
        entity (str): entity sequence in sentence
        left (str): left context sequence of this entity in sentence
        right (str): right context sequence of this entity in sentence
        ast (str): assertion label for this medical entity

    Attributes:
        entity (str): entity sequence in sentence
        entity_start (int): start position of entity in sentence
        entity_end (int): end position of entity in sentence
        left (str): left context sequence of this entity in sentence
        right (str): right context sequence of this entity in sentence
        ast_label (str): assertion label for this medical entity
        entity_embedding (torch.Tensor): embedding representing entity in sentence
    """
    def __init__(self, **kwargs): #entity: str, start: int,  left: str, right: str, mod: str):

        self.entity = kwargs.get('event')
        self.start = kwargs.get('start')
        self.end = kwargs.get('end')
        self.mod = kwargs.get('event_modality')
        self.pol = kwargs.get('event_polarity')

        self.left = kwargs.get('left')
        self.right = kwargs.get('right')
        
        self.other_event = kwargs.get('other_event')
        self.other_event_type = kwargs.get('other_event_type')
        self.other_start = kwargs.get('other_start')
        self.other_end = kwargs.get('other_end')
        self.other_mod = kwargs.get('other_event_modality')
        self.other_pol = kwargs.get('other_event_polarity')
        
        self.entity_embedding = None

    def set_entity_embedding(self, left_emb: torch.Tensor, entity_emb: torch.Tensor, right_emb: torch.Tensor):
        """
        Set embedding for 3 parts of an entity and combine them together as representation for an entity in sentence

        Args:
            left_emb (torch.Tensor): phrase embedding for left context sequence in sentence
            entity_emb (torch.Tensor): phrase embedding for entity in sentence
            right_emb (torch.Tensor): phrase embedding for right context sequence in sentence

        """
        assert isinstance(left_emb, torch.Tensor)
        assert isinstance(entity_emb, torch.Tensor)
        assert isinstance(right_emb, torch.Tensor)
        self.entity_embedding = torch.cat((left_emb, entity_emb, right_emb), 0)


class Sentence:
    """
    Class Sentence contains information of elements in sentence

    Args:
        line (str): line representing sentence
        sent_id (int): sentence id defined in data

    Attributes:
        replace (str): annotation part to be replaced
        sent_id (int): sentence id defined in data
        entities (dict(str:str)): all entities in a sentence
        sentence (str): sentence in raw form without annotation
    """
    def __init__(self, data: dict, sent_id: int, doc_id: int):

        self.sent_id = sent_id
        self.doc_id = doc_id

        self.entities = {}
        self.__parse_entity(line)

        self.sentence = ''
        self.__parse_sent(line)

    def __parse_entity(self, line: str):
        """
        Parse data used for training in GRU
        3 parts: pre-entity, entity, post-entity

        Args:
            line (str): line to parse entity
        """

        

    def __parse_sent(self, line: str):
        """
        Parse sentence used for embedding as input

        Args:
            line (str): line to parse sentence
        """
        
        


class Data:
    """
    Class Data contains all sentences with annotations.

    Args:
        filename (str): path to preprocessed file with annotations
        task ()
        type_data (str): name of data type (train/test)

    Attributes:
        type_data (str): name of data type (train/test)
        sentences (list[str]): list of all sentences that contain entity of medical problem concept and its assertion
                                annotation
    """
    def __init__(self, filename: str, task='modpol', type_data='train'):

        assert os.path.isfile(filename), f'{filename} does not exist!'

        self.type_data = type_data
        self.sentences = []

        self.__read_file(filename)

    def __read_file(self, filename: str):
        """
        Read a given file and return the related sentences for use

        Args:
            filename (str): name of data file
        """

        with open(filename) as f2r:
            data = json.loads(f2r.read())

        