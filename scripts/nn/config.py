# --*-- coding: utf-8 --*--
import torch


class Config:
    def __init__(self, device: torch.device = torch.device('cpu')):
        """

        :param device:
        """
        self.device = device


class PrepConfig(Config):
    def __init__(self, device: torch.device, text_src: str, srcfolder: str, outfolder: str,
                    dataset_type: str = 'train',  to_compress: bool = False, raw_pref: str = '.xml',
                 temprel_tags={'xml': [('<e2>', '</e2>'), ('<e1>', '</e2>')],
                               'nonxml': [('ebs', 'ebe'), ('eas', 'eae')]},
                 window_size=1500
                 ):
        """

        :param device:
        :param text_src:
        :param srcfolder:
        :param outfolder:
        :param dataset_type:
        :param to_compress:
        :param raw_pref:
        :param temprel_tags:
        :param window_size:
        """
        super(PrepConfig, self).__init__(device)

        self.window_size = window_size
        self.temprel_tags = temprel_tags
        self.raw_pref = raw_pref
        self.to_compress = to_compress
        self.dataset_type = dataset_type
        self.outfolder = outfolder
        self.srcfolder = srcfolder
        self.text_src = text_src


class EmbeddingConfig(Config):
    def __init__(self, device: torch.device = torch.device('cpu'), etype: str = 'bert', never_split=None):
        """

        :param device:
        :param etype:
        :param never_split:
        """
        super(EmbeddingConfig, self).__init__(device)

        self.etype = etype
        self.never_split = never_split


class ModelConfig(Config):

    def __init__(self, train_path, test_path, labels=[0, 1, 2], proportion=0.9,  # raw data
                 model_path='bert-base-uncased',  # tokenizer
                 train_model_path='model.pt',  # in case of continue the training from old training
                 batch_size: int = 500, hidden_size: int = 512, num_layers: int = 2, num_epochs: int = 10,
                 device: torch.device = torch.device('cpu'), drop_prob: float = 0.2,  # train hyperparameters
                 freeze_bert: bool = False, finetune=False, bert_hidden_size=768,  # type of model
                 l_rate: float = 0.001, eps: float = 1e-8,  # optimizer hyperparameters
                 model_save_path='out/model', infer_save_path='out/infer'  # output path
                 ):
        super(ModelConfig, self).__init__(device)
        
        # for tokenizer and bertmodel, both in embedding and in finetuning
        self.model_path = model_path

        # training model if continue training process
        self.train_model_path = train_model_path

        # data
        self.train_path = train_path
        self.test_path = test_path
        self.proportion = proportion

        # list of labels
        self.labels = labels
        self.num_classes = len(self.labels)

        # save end model output path
        self.model_save_path = model_save_path
        self.infer_save_path = infer_save_path

        # hyperparameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.l_rate = l_rate
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.eps = eps

        # model type
        self.finetune = finetune

        # for GRU and biLSTM
        self.drop_prob = drop_prob

        # for finetuning
        self.freeze_bert = freeze_bert
        self.input_size = bert_hidden_size

