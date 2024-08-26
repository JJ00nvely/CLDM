# coding=utf-8
import torch
import torch.nn as nn
import math


def positional_encoding_1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class GPO(nn.Module):
    def __init__(self, d_pe, d_hidden, length):
        super(GPO, self).__init__()
        self.d_pe = d_pe
        self.d_hidden = d_hidden
        self.pe_database = {}
        self.length = length
        self.gru = nn.GRU(self.d_pe, d_hidden, 1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.d_hidden, 1, bias=False)

    def compute_pool_weights(self, features):
        max_len = self.length
        # 포지셔널 인코딩 생성
        pe_max_len = self.get_pe(max_len)
        pes = pe_max_len.unsqueeze(0).to(features.device)
        self.gru.flatten_parameters()
        out, _ = self.gru(pes)
        out_emb = (out[:, :, :out.size(2) // 2] + out[:, :, out.size(2) // 2:]) / 2
        scores = self.linear(out_emb)
        weights = torch.softmax(scores / 0.1, 1)
        return weights
    
    def forward(self, features):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """
        pool_weights = self.compute_pool_weights(features) # 256 ?
        # if self.token_pool == True:
        #     features = features.sort(dim=1, descending=True)[0]
        #     pooled_features = (features * pool_weights).sum(1)
        # else:
            # Token 이 아닌 마지막 차원
        features = features.sort(dim=2, descending=True)[0]
        pool_weights = pool_weights.permute(0,2,1)
        pooled_features = (features * pool_weights).sum(2)
        return pooled_features
    
    def get_pe(self, length):
        """

        :param length: the length of the sequence
        :return: the positional encoding of the given length
        """
        length = int(length)
        if length in self.pe_database:
            return self.pe_database[length]
        else:
            pe = positional_encoding_1d(self.d_pe, length)
            self.pe_database[length] = pe
            return pe