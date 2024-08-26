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

class CLDM(ModelMixin, ConfigMixin):
    def __init__(self,latent_dim=256, num_heads=8, dropout_r=0.,num_layers=6,activation='gelu',
                 cond_emb_size=256,gpo =True, use_temp = False, video_length=16):
        super().__init__()

        logging.info('Loading the Vision Encoder')
        self.visual_encoder  = Dinov2Model.from_pretrained("facebook/dinov2-base")
        logging.info("freeze vision encoder")
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder = self.visual_encoder.eval()
        self.visual_encoder.train = disabled_train
        logging.info('Loading Dino-v2 Done')

        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.dropout_r = dropout_r
        self.gpo = gpo
        self.use_temp = use_temp
        self.video_length = video_length


        logging.info(self.use_temp)

        
        if self.gpo ==True:
            logging.info('Using GPO')
            self.length = latent_dim
            self.gpool = GPO(32,32, self.length)
        else :
            # add additional code if we are not use gpool
            pass

        self.image_proj = nn.Linear(768, latent_dim)
        self.seq_pos_enc = PositionalEncoding(self.latent_dim, 0.)
        self.embed_timestep = TimestepEmbedder(cond_emb_size)

        # Adjust Here when there are some problem with Transformer
        self.transformer = Custom(latent_dim=self.latent_dim, num_heads = num_heads, dr_rate = dropout_r, num_layers=num_layers, video_length=video_length, use_temp=self.use_temp)
        


        # self.initialize_out_fc()

        self.decode = nn.Linear(258, 4)
        
        self.x_emb = nn.Sequential(
            nn.Linear(1, int(latent_dim)),
        )
        self.y_emb = nn.Sequential(
            nn.Linear(1, int(latent_dim)),
        )
        self.w_emb = nn.Sequential(
            nn.Linear(1, int(latent_dim)),
        )
        self.h_emb = nn.Sequential(
            nn.Linear(1, int(latent_dim)),
        )


    def initialize_out_fc(self):
        # out_fc 레이어 접근 및 가중치 초기화
        out_fc_layer = self.transformer.tempcoder_block.self_attention.out_fc
        with torch.no_grad():
            out_fc_layer.weight.zero_()
            if out_fc_layer.bias is not None:
                out_fc_layer.bias.zero_()

    # 로더에서 선처리하도록 만들어둠
    def encode_img(self,image):
        device = image.device
        img_emb = self.visual_encoder(image).last_hidden_state.to(device)
        img_emb = img_emb[:,1:,:]
        return img_emb

    def forward(self, noisy_sample, timesteps): # src : image embedding
        image = noisy_sample['image']
        diff_box = noisy_sample['box']


        src = self.encode_img(image)  # B,256,768 

        x = diff_box[:,0].unsqueeze(-1)
        y = diff_box[:,1].unsqueeze(-1)
        w = diff_box[:,2].unsqueeze(-1)
        h = diff_box[:,3].unsqueeze(-1)
        
        x_emb = self.x_emb(x)
        y_emb = self.y_emb(y)
        w_emb = self.w_emb(w)
        h_emb = self.h_emb(h)

        x_emb = x_emb.unsqueeze(1)
        y_emb = y_emb.unsqueeze(1)
        w_emb = w_emb.unsqueeze(1)
        h_emb = h_emb.unsqueeze(1)


        box_emb = torch.cat((x_emb,y_emb,w_emb,h_emb), dim=1) # B, 4, 256
        img_emb = self.image_proj(src) # B,256,256
        t_emb = self.embed_timestep(timesteps) # B, 1, 256
        embeded = torch.cat((box_emb,img_emb,t_emb), dim=1)
        embeded = self.seq_pos_enc(embeded)


        enc_output=self.transformer(embeded) # B,258,256 -> DLT 에서는 Timestep 에 관련된 것을 빼고 MLP 를 태운 것으로 확인 
        # B , 4 ,256 -> 이거 256 에서 1 로 바꾸고 바로 NOISE ESTIMATE ? 
        enc_output = self.gpool(enc_output) # B,258 / -> Token 의 각 Feature 를 Aggregation 을 수행
        output_noise = self.decode(enc_output) # B, 4
        return output_noise
        # 추가적으로 생각 가능한 것이 transformer block 별로 conditioned 이 들어갈 수 있도록 작성이 가능할 것
        # Cross Attention 방식으로 ?

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self