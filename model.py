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
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.attn_state_size = hidden_size
        self.attn_cnt = 1
        self.attn_len = 7 * 7
        self.cap_state_size = hidden_size
        self.cap_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_size)

        self.attn_rnn = nn.LSTM(self.embed_size, self.attn_state_size, batch_first=True)
        self.attn = nn.Linear(self.attn_state_size, self.attn_cnt * self.attn_len)
        self.cap_rnn = nn.LSTM((self.attn_cnt + 1) * self.embed_size, self.cap_state_size, self.cap_layers,
                               batch_first=True)
        self.cap = nn.Linear(self.cap_state_size, self.vocab_size)

        torch.nn.init.xavier_uniform_(self.attn.weight)
        torch.nn.init.xavier_uniform_(self.cap.weight)

    def forward(self, features, captions):
        feats, init_feats = self.flat_feats(features)
        embed = self.embed(captions)
        embed = torch.cat((init_feats, embed), 1)

        attn, _ = self.attn_rnn(embed)
        attn = self.linear_forward(self.attn, attn)
        attn_feats = self.attn_feats(attn, self.attn_cnt, feats)

        attn_feats = torch.cat((attn_feats, embed), 2)
        cap, _ = self.cap_rnn(attn_feats)
        cap = self.linear_forward(self.cap, cap)
        return cap[:, :-1]

    def sample(self, inputs, states=None, max_len=20):
        """
        accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)
        """
        results = []
        attns = []
        feats, embed = self.flat_feats(inputs)
        if not states:
            states = (None, None)
        attn_states, cap_states = states
        for i in range(max_len):
            # (N, 1, H)
            attn, attn_states = self.attn_rnn(embed, attn_states)
            attn = self.linear_forward(self.attn, attn)
            attns.append(attn)
            attn_feats = self.attn_feats(attn, self.attn_cnt, feats)

            attn_feats = torch.cat((attn_feats, embed), 2)
            cap, cap_states = self.cap_rnn(attn_feats, cap_states)
            cap = self.linear_forward(self.cap, cap)

            _, outputs = cap.squeeze(1).max(1)
            results.append(outputs.item())
            embed = self.embed(outputs.unsqueeze(1))
        return results, attns

    @staticmethod
    def flat_feats(feats):
        batch_size, channels_cnt, height, weight = feats.shape
        feats = feats.transpose(1, 2).transpose(2, 3)
        feats = feats.reshape(batch_size, height * weight, channels_cnt)
        return feats, torch.mean(feats, 1).unsqueeze(1)

    @staticmethod
    def attn_feats(attn, attn_cnt, feats):
        batch_size, attn_len, embed_size = feats.shape
        # (N, S, C, L, 1)
        attn = attn.reshape(batch_size, -1, attn_cnt, attn_len, 1)
        attn = F.softmax(attn, 3)
        # (N, 1, 1, L, E)
        feats = feats.reshape(batch_size, 1, 1, attn_len, embed_size)
        attn_feats = torch.sum(feats * attn, 3)
        return attn_feats.reshape(batch_size, -1, attn_cnt * embed_size)

    @staticmethod
    def linear_forward(f, x):
        """ linear forward for last dimension"""
        batch_size = x.shape[0]
        y = x.reshape(-1, f.in_features)
        y = f(y)
        y = y.reshape(batch_size, -1, f.out_features)
        return y
