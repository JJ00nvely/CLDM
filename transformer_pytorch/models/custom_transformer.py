import torch
import torch.nn as nn



import torch
import torch.nn as nn
import logging
from .model.encoder import Encoder
from .block.encoder_block import EncoderBlock
from .layer.multi_head_attention_layer import MultiHeadAttentionLayer
from .layer.position_wise_feed_forward_layer import PositionWiseFeedForwardLayer
from .block.temp_block import tempcoder
import copy


class Custom(nn.Module):
    def __init__(self, latent_dim=256, num_heads=8, dr_rate=0., num_layers=8, video_length=16, use_temp=False):
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

        # Initialize encoder block and encoder
        encoder_block = EncoderBlock(
            self_attention=self.copy(attention),
            position_ff=self.copy(position_ff),
            norm=self.copy(norm),
            dr_rate=dr_rate
        )
        tempcoder_block = tempcoder(
            self_attention=self.copy(attention),
            temp_attention=self.copy(attention),
            position_ff=self.copy(position_ff),
            norm = self.copy(norm),
            video_length=video_length,
            dr_rate = dr_rate
        )
        
        if use_temp:
            logging.info('Loading the Temporal Encoder')
            self.encoder = Encoder(
            encoder_block = tempcoder_block,
            n_layer = self.n_layer,
            norm=self.copy(norm)
            )
        else:
            logging.info('Loading the Original Encoder')
            self.encoder = Encoder(
            encoder_block=encoder_block,
            n_layer=self.n_layer,
            norm=self.copy(norm)
        )
    def forward(self, src):
        encoder_out = self.encoder(src)
        return encoder_out