import torch
import torch.nn as nn
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from utils import PositionalEncoding, TimestepEmbedder
from transformer.SubLayer import MultiHeadAttention, PositionwiseFeedForward
import logging
from transformers import Dinov2Model
from aggr.gpo import GPO
from transformer_pytorch.models.custom_transformer import Custom
from extractor.image import ImageFeatureExtractor 
from einops import rearrange

class CLDM(ModelMixin, ConfigMixin):
    def __init__(self,latent_dim=256, num_heads=8, dropout_r=0.,num_layers=6,activation='gelu',
                 cond_emb_size=256,gpo =True, use_temp = False, video_length=16):
        super().__init__()

        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.dropout_r = dropout_r
        self.gpo = gpo
        self.use_temp = use_temp
        self.video_length = video_length

        logging.info('Using Temporal Attention Layer: %s', self.use_temp)

        # if self.gpo ==True:
        #     logging.info('Using GPO')
        #     self.length = latent_dim
        #     self.gpool = GPO(32,32, self.length)
        # else :
        #     # add additional code if we are not use gpool
        #     pass

        self.extractor = ImageFeatureExtractor(backbone_name='resnet50')
        self.seq_pos_enc = PositionalEncoding(self.latent_dim, 0.)
        self.embed_timestep = TimestepEmbedder(cond_emb_size)

        # Adjust Here when there are some problem with Transformer
        self.transformer = Custom(latent_dim=self.latent_dim, num_heads = num_heads, dr_rate = dropout_r, num_layers=num_layers, video_length=video_length, use_temp=self.use_temp)
        
        if self.use_temp == True :
            self.initialize_out_fc()
        self.decode = nn.Linear(512, 4)
        
        self.loc_emb = nn.Sequential(
            nn.Linear(2, latent_dim),
        )
        self.size_emb = nn.Sequential(
            nn.Linear(2, latent_dim),)
        # self.loc_decode = nn.Sequential(
        #     nn.Linear(latent_dim, 2),
        # )
        # self.size_decode = nn.Sequential(
        #     nn.Linear(latent_dim, 2),
        # )
        

    def initialize_out_fc(self):
        # initialize the project layer's in Additional Temporal Attention Layer
        for i in self.transformer.encoder.layers:
            out_fc_layer = i.temp_attention.out_fc
            with torch.no_grad():
                out_fc_layer.weight.zero_()
                if out_fc_layer.bias is not None:
                    out_fc_layer.bias.zero_()


    def forward(self, noisy_sample, timesteps): # src : image embedding

        image = noisy_sample['image']
        diff_box = noisy_sample['box']
        xy = diff_box[:,:2]
        wh = diff_box[:,2:]

        loc_emb = self.loc_emb(xy)
        loc_emb = loc_emb.unsqueeze(1)

        size_emb = self.size_emb(wh)
        size_emb = size_emb.unsqueeze(1)

        box_emb = torch.cat((loc_emb,size_emb),dim=1) # B 2 256

        ftrs = self.extractor(image) # B 529 256 -> Visual Feature Embedding : )
        enc_output = torch.cat((box_emb,ftrs), dim=1) # B, 531,256
        t_emb = self.embed_timestep(timesteps) # B, 1, 256
        enc_output = torch.cat((enc_output,t_emb),dim=1) # B,532,256
        
        # PE encode
        enc_output = self.seq_pos_enc(enc_output) # enc_output 과 동일한 B,S,F
        enc_output=self.transformer(enc_output) # B,532,256
        # 여기서 B, 532, 256이 들어가니까 제일 앞에 2칸을 떼어서 사용해야함 한 번 들어가면 두개의 embedding으로 나눠서 사용할 수 있게
        # Output 이 B,2,256 으로 나옴
        # enc_output = self.gpool(enc_output) # B,258 / -> Token 의 각 Feature 를 Aggregation 을 수행
        # -> 여기서부터 추가 수정 MLP 2개 넣어서 Regression 후에 Output Noise 를 빼다가져갈 수 있도록 수행 -> 이거 코드 작성하면서 DLT 코드랑 다시 천천히 비교하기
        # 처음 Noise 를 추가해서 loc emb 하고 size emb 어떻게 제대로 나누었는지 box 에다가 Noise 를 한꺼번에 주는지 아니면 x,y,w,h 를 다 따뤄 nOISE 를 주는지 체크하기
        enc_output = rearrange(enc_output,'b c d -> b 1 (c d)' )
        output_noise = self.decode(enc_output) # B, 4
        return output_noise
        # 추가적으로 생각 가능한 것이 transformer block 별로 conditioned 이 들어갈 수 있도록 작성이 가능할 것
        # Cross Attention 방식으로 ?

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


