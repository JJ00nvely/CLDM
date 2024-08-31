"""
@author : Hansu Kim(@cpm0722)
@when : 2022-08-21
@github : https://github.com/cpm0722
@homepage : https://cpm0722.github.io
"""

import copy
import torch.nn as nn
import torch


class Encoder(nn.Module):

    def __init__(self, encoder_block, n_layer, norm):
        super(Encoder, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(encoder_block) for _ in range(self.n_layer)])
        self.norm = norm

    def forward(self, src): # src_mask
        # B,532, 256 
        out = src
        emb = src[:,2:,:]
        for layer in self.layers:
            out = out[:,:2,:]
            out = torch.cat((out,emb),dim=1)
            out = layer(out) 
        out = self.norm(out)
        return out[:,:2,:]
    