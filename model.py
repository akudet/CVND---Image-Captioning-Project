import torch

import torch.nn as nn
import torchvision.models as models


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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size):
        return (torch.zeros((self.num_layers, batch_size, self.hidden_size)),
                torch.zeros((self.num_layers, batch_size, self.hidden_size)))

    def forward(self, features, captions):
        batch_size = features.shape[0]
        embed = self.embed(captions)
        embed = torch.cat((features.unsqueeze(1), embed), 1)

        hc = self.init_hidden(batch_size)
        hc = hc[0].to(features.device), hc[1].to(features.device)
        out, hc = self.lstm(embed, hc)

        out = out.reshape(-1, self.hidden_size)
        out = self.fc(out)
        out = out.reshape(batch_size, -1, self.vocab_size)
        return out[:, :-1]

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass
