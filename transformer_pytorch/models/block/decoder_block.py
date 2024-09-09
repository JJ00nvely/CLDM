"""
@author : Hansu Kim(@cpm0722)
@when : 2022-08-21
@github : https://github.com/cpm0722
@homepage : https://cpm0722.github.io
"""

import copy
import torch.nn as nn
import logging
from einops import rearrange
from transformer_pytorch.models.layer.residual_connection_layer import ResidualConnectionLayer


class DecoderBlock(nn.Module):

    def __init__(self, self_attention, cross_attention, temporal_attention, position_ff, norm,use_temp,video_length ,dr_rate=0):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.use_temp =use_temp
        self.residual1 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.cross_attention = cross_attention
        self.residual2 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.video_length = video_length
        if self.use_temp:
            logging.info('Loading the Temporal Layer')
            self.temporal_attention = temporal_attention
            self.residual3 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        else:
            logging.info('Loading the Original Layer')
            pass
        self.position_ff = position_ff 
        self.residual4 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)


    def forward(self, tgt, encoder_out):
        out = tgt
        out = self.residual1(out, lambda out: self.self_attention(query=out, key=out, value=out))
        out = self.residual2(out, lambda out: self.cross_attention(query=out, key=encoder_out, value=encoder_out))
        if self.use_temp:
            out = rearrange(out, '(b t) n d -> (b n) t d', t=self.video_length, n=5)
            out = self.residual3(out, lambda out: self.temporal_attention(query=out, key=out, value=out))
            out = rearrange(out,'(b n) t d -> (b t) n d', t=self.video_length, n=5)
        out = self.residual4(out, self.position_ff)
        return out
