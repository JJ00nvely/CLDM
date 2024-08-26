import copy
import torch.nn as nn
from einops import rearrange

from transformer_pytorch.models.layer.residual_connection_layer import ResidualConnectionLayer


class tempcoder(nn.Module):

    def __init__(self, self_attention, temp_attention, position_ff, norm, video_length,dr_rate=0):
        super(tempcoder, self).__init__()
        self.self_attention = self_attention
        self.temp_attention = temp_attention
        self.position_ff = position_ff
        self.video_length = video_length

        self.residual1 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.residual2 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.residual3 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)

    def forward(self, src): # src_mask
        out = src
        out = self.residual1(out, lambda out: self.self_attention(query=out, key=out, value=out)) # , mask=src_mask
        out = rearrange(out, '(b t) n d -> (b n) t d', t=self.video_length, n=258)
        out = self.residual3(out, lambda out: self.temp_attention(query=out, key=out, value=out))
        out = rearrange(out,'(b n) t d -> (b t) n d', t=self.video_length, n=258)
        out = self.residual2(out, self.position_ff)
        return out
