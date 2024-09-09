import torch
import torch.nn as nn



import torch
import torch.nn as nn
import logging
from .layer.multi_head_attention_layer import MultiHeadAttentionLayer
from .layer.position_wise_feed_forward_layer import PositionWiseFeedForwardLayer
import copy
from transformer_pytorch.models.block.decoder_block import DecoderBlock
from transformer_pytorch.models.model.decoder import Decoder


class Custom(nn.Module):
    def __init__(self, latent_dim=256, num_heads=8, dr_rate=0., num_layers=6, video_length=16, use_temp=False):
        super(Custom, self).__init__()  # Ensure parent class is properly initialized
        self.d_embed = latent_dim
        self.d_ff = 2 * latent_dim
        self.n_layer = num_layers
        self.video_length = video_length
        self.copy = copy.deepcopy
        # Define attention and feed-forward layers
        attention = MultiHeadAttentionLayer(
            d_model=latent_dim,
            h=num_heads,
            qkv_fc=nn.Linear(self.d_embed, latent_dim),
            out_fc=nn.Linear(latent_dim, self.d_embed),
            dr_rate=dr_rate
        )
        position_ff = PositionWiseFeedForwardLayer(
            fc1=nn.Linear(self.d_embed, self.d_ff),
            fc2=nn.Linear(self.d_ff, self.d_embed),
            dr_rate=dr_rate
        )
        
        norm = nn.LayerNorm(self.d_embed, eps=1e-5)

        decoder_block = DecoderBlock(
                                    self_attention = self.copy(attention),
                                    cross_attention = self.copy(attention),
                                    position_ff = self.copy(position_ff),
                                    norm = self.copy(norm),
                                    temporal_attention=self.copy(attention),
                                    use_temp= use_temp,
                                    video_length=self.video_length,
                                    dr_rate = dr_rate)   

        self.decoder = Decoder(
                        decoder_block = decoder_block,
                        n_layer = self.n_layer,
                        norm = self.copy(norm))        
    def forward(self, tgt, memory):
        encoder_out = self.decoder(tgt, memory)
        return encoder_out