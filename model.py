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
        self.bn = nn.BatchNorm1d(embed_size)

        torch.nn.init.xavier_uniform_(self.embed.weight)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
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

        torch.nn.init.xavier_uniform_(self.fc.weight)

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
        results = []
        if not states:
            states = self.init_hidden(inputs.shape[0])
            states = states[0].to(inputs.device), states[1].to(inputs.device)
        for i in range(max_len):
            # [N, 1, H]
            outputs, states = self.lstm(inputs, states)
            # [N, V]
            outputs = self.fc(outputs.squeeze(1))
            _, outputs = outputs.max(1)
            results.append(outputs.item())
            inputs = self.embed(outputs.unsqueeze(1))
        return results
