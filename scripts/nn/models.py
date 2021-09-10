# --*-- coding: utf-8 --*--
"""
Models
"""
import torch.nn as nn
import torch
from transformers import BertModel

class Net(nn.Module):
    def __init__(self, input_size=1024, hidden_size=512, num_labels=2, num_layers=2):
        super(Net, self).__init__()

        self.mode = 'None'
        self.model = None
        self.classifier = None

    def forward(self):
        pass

    def __repr__(self):
        return f'Model {self.mode}'


# Create GRUNet
class GRUNet(Net):
    def __init__(self, input_size:int=1024, hidden_size:int=512, 
                num_labels:int=2, num_layers:int=2, dropout:float=0.2):

        super(GRUNet, self).__init__()

        self.mode = 'GRU'

        # InstantiateGRU Net
        self.model = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                         dropout=dropout, batch_first=True)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, num_labels),
            nn.ReLU())

    def forward(self, x):

        output, _ = self.model(x)

        logits = self.classifier(output[:, -1])

        return logits


# Create LSTMNet
class BiLSTMNet(Net):
    def __init__(self, input_size:int=1024, hidden_size:int=512, 
                    num_labels:int=3, num_layers:int=2, dropout:float=0.2):

        super(BiLSTMNet, self).__init__()

        self.mode = 'BiLSTM'

        # Instantiate a BiLSTM Net
        self.model = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                        bidirectional=True, dropout=dropout, batch_first=True)
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, num_labels),
            nn.Sigmoid())


    def forward(self, x):
        

        output, (hidden, ct) = self.model(x)

        ht = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)

        logits = self.classifier(ht)

        return logits


class BertNet(Net):
    def __init__(self, model:str='bert-base-uncased', 
                    input_size:int=768, hidden_size:int=50, num_labels:int=3,
                     freeze_bert=False):

        super(BertNet, self).__init__()

        self.mode = 'finetuned'

        # Instantiate BERT model
        self.model = BertModel.from_pretrained(model)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Softmax(),
            nn.Linear(hidden_size, num_labels))

        # Freeze the BERT model
        if freeze_bert:
            for param in self.model.parameters():
                param.requires_grad = False


    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        :param:    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        :param:    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        :return:   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """

        # Feed input to BERT
        outputs = self.model(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits
