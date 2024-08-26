import torch
import torch.nn as nn
from einops import rearrange

class SingleSpaTemp(nn.Module):
    def __init__(self, encoder_block, temp_encoder_block, video_length):
        super(SingleSpaTemp, self).__init__()
        self.encoder_block = encoder_block
        self.temp_encoder_block = temp_encoder_block
        self.b = video_length
    def forward(self, x):
        # Apply the initial encoder block
        x = self.encoder_block(x)
        
        # Reshape the tensor
        x = rearrange(x, '(b t) n d -> (b n) t d', b=self.b)
        
        # Apply the temporary encoder block
        x = self.temp_encoder_block(x)
        
        # Reshape the tensor back to original shape
        x = rearrange(x, '(b n) t d -> (b t) n d', b=self.b)
        
        return x