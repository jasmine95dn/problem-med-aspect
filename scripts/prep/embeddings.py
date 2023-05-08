# --*-- coding: utf-8 --*--

from transformers import BertTokenizer, AutoModel
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings, FlairEmbeddings, StackedEmbeddings, WordEmbeddings
import torch
import sys
from torch.nn import GRU, Linear



class SentenceEmbedding:
    """
    Class SentenceEmbedding defines embedding for a sentence, including token embedding and phrase embedding

    Args:
        sent (str):
        path_to_model (str):
        mode (str):
        n_last_layers (int):

    Attributes:
        tokens
    """
    def __init__(self, sent: str, path_to_model='emilyalsentzer/Bio_ClinicalBERT', mode='concat',
                 n_last_layers=3):
        # pre-define self.tokens and self.embedding
        self.tokens = sent.split('')
        self.embeddings = None

        self.path_to_model = path_to_model

        # re-define self.tokens and self.embedding
        self.get_tokens_embedding(sentence=sent, path_to_model=path_to_model, mode=mode,
                                  n_last_layers=n_last_layers)

    @staticmethod
    def get_vector(output_layers, mode='avg', n_last_layers=3) -> torch.Tensor:
        """
        :param output_layers:
        :param mode:
        :param n_last_layers:
        :return:
        """
        concat = torch.squeeze(torch.cat(output_layers, -1))
        if mode == 'concat':
            return concat
        else:
            n_vectors, n_concat = concat.shape
            if mode == 'avg':
                return torch.mean(torch.reshape(concat, (n_vectors, n_last_layers, n_concat // n_last_layers)), 1)

            elif mode == 'sum':
                return torch.sum(torch.reshape(concat, (n_vectors, n_last_layers, n_concat // n_last_layers)), 1)

            else:
                sys.stderr.write(f'No {mode} available!')
                sys.exit(1)


    @classmethod
    def tokenize(cls, path_to_model, sentences, never_split=['<p>', '</p>'])):

        """BASELINE + TEMPREL!!!!!!!! CHANGE HERE !!!inputs for transformers, sents for flair (for baseline)"""
        
        # load tokenizer for wanted model
        tokenizer = BertTokenizer.from_pretrained(path_to_model, never_split=never_split)

        # load inputs before feeding into embeddings

        inputs = tokenizer(sentences, padding=True, return_tensors='pt')

        # assign decoded sentences
        sents = [tokenizer.convert_ids_to_tokens(sent) for sent in inputs['input_ids']]
        
        return inputs, sents

    def export_from_flair(self, type_='flair', sentence: str = None, sentences: list = None):
        """baseline: a lot of sentences with padding, normal one passed into GRU process each sentence"""



    def export_from_transformers_split(self, path_to_model: str = 'emilyalsentzer/Bio_ClinicalBERT', sent: list):

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(path_to_model)

        # load model
        model = AutoModel.from_pretrained(path_to_model)

        # sent includes 3 parts
        pre, ent, post =  sent

        # 1. use only [CLS] for pre
        # tokenize
        pre_inp = tokenizer(pre, return_tensors='pt')
        # output
        pre_out = model(**pre_inp)
        # last output
        pre_states = pre_out.last_hidden_state[:,:-1,:]

        # 2. no [CLS] and [SEP] for ent
        # tokenize
        ent_inp = tokenizer(ent, return_tensors='pt')
        # output
        ent_out = model(**ent_inp)
        # last output
        ent_states = ent_out.last_hidden_state[:,1:-1,:]

        # 3. use only [SEP] for post
        # tokenize
        post_inp = tokenizer(post, return_tensors='pt')
        # output
        post_out = model(**post_inp)
        # last output
        post_states = post_out.last_hidden_state[:,1:,:]

        return pre_states, ent_states, post_states

    def export_from_transformers(self, path_to_model: str='emilyalsentzer/Bio_ClinicalBERT', sentences: list, inputs):

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(path_to_model)

        # load model
        model = AutoModel.from_pretrained(path_to_model, output_hidden_states=False)

        # create embeddings from pre-trained model for given sentence
        outputs = model(**inputs)

        return ouputs.last_hidden_state

    def combine_bert_flair(self, type_: str, bert, flair):

        type_ = type_.lower()

        # clincalbert flair


        # clinicalbert hunflair

    def get_tokens_embedding(self, sentence: str, path_to_model: str, mode: str, n_last_layers: int) -> torch.Tensor:
        """

        :param sentence:
        :param path_to_model:
        :param mode:
        :param n_last_layers:
        :return:
        """
        # load tokenizer for wanted model
        tokenizer = AutoTokenizer.from_pretrained(path_to_model)

        # load model
        model = AutoModel.from_pretrained(path_to_model, output_hidden_states=True)

        # create embeddings from pre-trained model for given sentence
        inputs = tokenizer(sentence, return_tensors='pt')
        outputs = model(**inputs)

        # assign decoded sentence based on pre-trained embedding
        self.tokens = tokenizer.convert_ids_to_tokens(torch.squeeze(inputs['input_ids']))

        # return token embeddings in given mode
        if mode == 'last':
            self.embeddings = torch.reshape(outputs.last_hidden_state, outputs.last_hidden_state.shape[1:])
        else:
            self.embeddings = self.get_vector(output_layers=outputs.hidden_states, mode=mode, n_last_layers=n_last_layers)

    @staticmethod
    def get_phrase_embedding(phrase: torch.Tensor, hidden_size: int = 256, num_layers: int = 2, drop_out: float = 0):
        """

        :param drop_out:
        :param phrase: size (seq_size, input_size)
        :param hidden_size:
        :param num_layers:
        :returns representation of this phrase size (1, input_size)
        """
        # turn phrase embedding of size (seq_size, input_size) into (1, seq_size, input_size) (batch_size=1 in this
        # case since we only have 1 phrase)
        phrase = torch.unsqueeze(phrase, 0)

        # apply GRU layer(s) and a linear layer to get phrase embedding
        h0 = torch.zeros(num_layers, phrase.size(0), hidden_size) # size of num_layers, batch_size, hidden_size
        model = GRU(input_size=phrase.size(-1), hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                    dropout=drop_out)
        linear = Linear(in_features=hidden_size, out_features=phrase.size(-1))
        out, _ = model(phrase, h0)
        out = linear(out[:, -1])
        return out
