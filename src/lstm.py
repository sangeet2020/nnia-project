import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, target_size, drp):
        super(LSTMTagger, self).__init__()
        
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.hidden2tag = nn.Linear(2*hidden_dim, target_size)
        
    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence)
        lstm_out = self.relu(lstm_out)
        lstm_out = self.dropout(lstm_out)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
