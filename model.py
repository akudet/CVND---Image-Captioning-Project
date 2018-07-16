import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Sequential(
            nn.Conv2d(2048, embed_size, 1),
            nn.BatchNorm2d(embed_size),
            nn.ReLU(),
        )

    def forward(self, images):
        features = self.resnet(images)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        context_size = 7 * 7

        self.embed = nn.Embedding(vocab_size, embed_size)

        self.attn_rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.cap_rnn = nn.LSTM(2 * embed_size, hidden_size, num_layers, batch_first=True)

        self.attn = nn.Linear(hidden_size, context_size)
        self.cap = nn.Linear(hidden_size, vocab_size)

        torch.nn.init.xavier_uniform_(self.attn.weight)
        torch.nn.init.xavier_uniform_(self.cap.weight)

    def forward(self, features, captions):
        feats, init_feats = self.flat_feats(features)
        embed = self.embed(captions)
        embed = torch.cat((init_feats, embed), 1)

        # (N, S+1, H)
        attn, _ = self.attn_rnn(embed)
        attn = self.linear_forward(self.attn, attn)
        attn_feats = self.attn_feats(attn, feats)

        # (N, S+1, H)
        attn_feats = torch.cat((attn_feats, embed), 2)
        cap, _ = self.cap_rnn(attn_feats)
        cap = self.linear_forward(self.cap, cap)
        return cap[:, :-1]

    def sample(self, inputs, states=None, max_len=20):
        """
        accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)
        """
        results = []
        feats, embed = self.flat_feats(inputs)
        if not states:
            states = (None, None)
        attn_states, cap_states = states
        for i in range(max_len):
            # (N, 1, H)
            attn, attn_states = self.attn_rnn(embed, attn_states)
            attn = self.linear_forward(self.attn, attn)
            attn_feats = self.attn_feats(attn, feats)

            attn_feats = torch.cat((attn_feats, embed), 2)
            cap, cap_states = self.cap_rnn(attn_feats, cap_states)
            cap = self.linear_forward(self.cap, cap)

            _, outputs = cap.squeeze(1).max(1)
            results.append(outputs.item())
            embed = self.embed(outputs.unsqueeze(1))
        return results

    @staticmethod
    def flat_feats(feats):
        N, C, H, W = feats.shape
        feats = feats.transpose(1, 2).transpose(2, 3)
        feats = feats.reshape(N, H * W, C)
        return feats, torch.mean(feats, 1).unsqueeze(1)

    @staticmethod
    def attn_feats(attn, feats):
        attn = F.softmax(attn, 2)
        # (N, S, L, 1)
        attn = torch.unsqueeze(attn, 3)
        # (N, 1, L, E)
        feats = torch.unsqueeze(feats, 1)
        attn_feats = torch.sum(feats * attn, 2)
        return attn_feats

    @staticmethod
    def linear_forward(f, x):
        """ linear forward for last dimension"""
        batch_size = x.shape[0]
        y = x.reshape(-1, f.in_features)
        y = f(y)
        y = y.reshape(batch_size, -1, f.out_features)
        return y
