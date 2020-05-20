import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=0.5, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        batch_size = captions.size(0)

        # embeddings and lstm_out
        captions = captions.long()
        captions = captions[:,:-1]
        embeds = self.embedding(captions)
        features = features.unsqueeze(1)
        lstm_in = torch.cat((features, embeds), 1)
        lstm_out, _ = self.lstm(lstm_in)
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)
        
        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = out.view(batch_size, -1, self.vocab_size)
        return out
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        words = []
        for _ in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)
            out = self.dropout(lstm_out)
            out = self.fc(out)
            out = out.view(-1, self.vocab_size)
            _, top_w = out.max(1)
            inputs = self.embedding(top_w.unsqueeze(1))
            w = int(top_w.squeeze().item())
            if w == 1: break
            words.append(w)
        words.append(1)
        return words