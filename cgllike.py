from extractor.image import ImageEncoder
from extractor.positional_encoding import build_position_encoding_1d
import torch.nn as nn
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
import logging
import torch
import torch.nn.functional as F
from logger_set import LOG
from transformer_pytorch.models.custom_transformer import Custom
from utils import TimestepEmbedder

class CLDM(ModelMixin, ConfigMixin):
    def __init__(self,
                 latent_dim : int = 256,
                 num_layers : int = 6,
                 num_heads : int = 8,
                 use_temp = False,
                 backbone_name ='resnet50'):
        super(CLDM, self).__init__() 
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.backbone_name= backbone_name
        self.num_heads = num_heads
        self.use_temp = use_temp
        LOG.info("Loading the %s encoder", self.backbone_name)

        self.encoder = ImageEncoder(
            d_model=self.latent_dim,
            backbone_name=self.backbone_name,
            num_layers=self.num_layers,
            pos_emb="sine",
        )
        self.pos_emb_1d = build_position_encoding_1d(
            pos_emb='layout', d_model=self.latent_dim
        )
        self.embed_timestep = TimestepEmbedder(self.latent_dim)
        self.decoder = Custom(latent_dim=self.latent_dim, num_heads = num_heads, num_layers=num_layers, video_length=16, use_temp=self.use_temp)
        # Embedding Bbox for Seperate
        self.x_emb = nn.Sequential(
            nn.Linear(1, self.latent_dim)
        )
        self.y_emb = nn.Sequential(
            nn.Linear(1, self.latent_dim)
        )
        self.w_emb = nn.Sequential(
            nn.Linear(1, self.latent_dim)
        )
        self.h_emb = nn.Sequential(
            nn.Linear(1, self.latent_dim)
        ) 
        self.decode = nn.Linear(self.latent_dim,1)

        self.init_weights()
        
        if self.use_temp ==True:
            self.initialize_out_fc()


    def init_weights(self) -> None:
        for module in [self.encoder.transformer_encoder, self.decoder]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
    
    def initialize_out_fc(self) :
        for i in self.decoder.decoder.layers:
            out_fc_layer = i.temporal_attention.out_fc
            with torch.no_grad():
                out_fc_layer.weight.zero_()
                if out_fc_layer.bias is not None:
                    out_fc_layer.bias.zero_()

    def forward(self, noisy_sample, timesteps):
    
        img = noisy_sample['image']

        src = self.encoder(img)

        box = noisy_sample['box']

        x = box[:,:1]   
        y = box[:,1:2]
        w = box[:,2:3]
        h = box[:,3:4]
        
        x_emb=self.x_emb(x) # B,1,256
        x_emb= x_emb.unsqueeze(1)

        y_emb=self.y_emb(y) # B,1,256
        y_emb= y_emb.unsqueeze(1)

        w_emb=self.w_emb(w) # B,1,256
        w_emb= w_emb.unsqueeze(1)

        h_emb=self.h_emb(h) # B,1,256
        h_emb= h_emb.unsqueeze(1) 

        t_emb = self.embed_timestep(timesteps) # B, 1, 256
        # Time step 임베딩 추가
        layout_enc = torch.cat((t_emb, x_emb,y_emb,w_emb,h_emb), dim=1) # B,4,256

        layout_enc = self.pos_emb_1d(layout_enc) # Batch,Box[Token], Feature

        latent = self.decoder(tgt=layout_enc, memory= src,) # Batch,Box[Token], Feature

        decode = self.decode(latent[:,1:,:]).squeeze(-1) # B, 4 (x,y,w,h)
    
        return decode # {"loss": loss, 'output' : decode}
    