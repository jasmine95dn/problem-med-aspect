# --*-- coding: utf-8 --*--
"""
Models
"""
import torch.nn as nn
import torch
from transformers import BertModel


# Create GRUNet
class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_labels=2, num_layers=2, dropout=0.2):

        super(GRUNet, self).__init__()

        # InstantiateGRU Net
        self.model = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                         dropout=dropout, batch_first=True)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, num_labels),
            nn.ReLU())

    def forward(self, x):

        output, _ = self.model(x)


        return self.classifier(output[:, -1]) #self.act(self.fc(output[:,-1]))


# Create LSTMNet
class BiLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size=512,
                    num_labels=3, num_layers=2, dropout=0.2):

        super(BiLSTMNet, self).__init__()

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

        return self.classifier(ht) #self.act(self.fc(ht))


class BertNet(nn.Module):
    def __init__(self, model='bert-base-uncased', bert_hidden_size:int=768, hidden_size:int=50, num_labels=3,
                     freeze_bert=False):

        super(BertNet, self).__init__()
        # Instantiate BERT model
        self.model = BertModel.from_pretrained(model)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden_size, hidden_size),
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
