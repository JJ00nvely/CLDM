import logging
import os

import fsspec
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# from image2layout.train.global_variables import PRECOMPUTED_WEIGHT_DIR
PRECOMPUTED_WEIGHT_DIR = "./cache/PRECOMPUTED_WEIGHT_DIR"
from torch import Tensor
from torchvision.models.feature_extraction import create_feature_extractor

from extractor.positional_encoding import build_position_encoding_2d

logger = logging.getLogger(__name__)

NORMALIZE = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
RESNET_WEIGHT = {
    "resnet18": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
    "resnet50": torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
}

class ImageEncoder(nn.Module):
    """
    This design follows encoder part of CGL-GAN (https://arxiv.org/abs/2205.00303)
    (i) extract image features
    (ii) flatten them into 1D sequence
    (iii) consider interaction using standard Transformer Encoder
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        backbone_name: str = "resnet50",
        num_layers: int = 6,
        dropout: float = 0.1,
        pos_emb: str = "sine",
        dim_feedforward: int = 2048,
    ) -> None:
        super().__init__()
        self.extractor = ImageFeatureExtractor(
            d_model=d_model, backbone_name=backbone_name
        )
        logger.info(f"Build ImageEncoder with {pos_emb=}, {d_model=}")
        self.pos_emb = build_position_encoding_2d(pos_emb, d_model)
        self.transformer_encoder = nn.TransformerEncoder(  # type: ignore
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                dropout=dropout,
                norm_first=True,
                dim_feedforward=dim_feedforward,
            ),
            num_layers=num_layers,
        )

    def init_weight(self) -> None:
        self.extractor.init_weight()
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return

    def forward(self, image: Tensor) -> Tensor:
        h = self.extractor(image.clone())
        h = self.pos_emb(h)
        h = self.transformer_encoder(h)
        return h  # type: ignore


class ImageFeatureExtractor(nn.Module):
    """
    This design follows encoder part of CGL-GAN (https://arxiv.org/abs/2205.00303)
    """

    def __init__(self, d_model: int = 256, backbone_name: str = "resnet18") -> None:
        super().__init__()
        return_nodes = {"layer4": "layer4", "layer3": "layer3"}
        model = getattr(torchvision.models, backbone_name)(
            weights=RESNET_WEIGHT[backbone_name]
        )
        self.body = create_feature_extractor(model, return_nodes=return_nodes)
        num_channels_dict = {
            "resnet18": {"layer3": 256, "layer4": 512},
            "resnet50": {"layer3": 1024, "layer4": 2048},
        }

        self.conv11 = nn.Conv2d(
            num_channels_dict[backbone_name]["layer4"], d_model // 2, 1
        )
        self.conv22 = nn.Conv2d(
            num_channels_dict[backbone_name]["layer3"], d_model // 2, 1
        )
        self.conv33 = nn.Conv2d(d_model // 2, d_model // 2, 1)


    def init_weight(self) -> None:
        for conv in [self.conv11, self.conv22, self.conv33]:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: Tensor) -> Tensor:
        x[:, 0:3] = NORMALIZE(x[:, 0:3])
        h = self.body(x)
        l3, l4 = h["layer3"], h["layer4"]
        f_up = F.interpolate(self.conv11(l4), l3.size()[2:], mode="bilinear")
        h = torch.cat(
            [f_up, self.conv33(f_up + self.conv22(l3))], dim=1
        )  # [b, c, h, w]
        # Comment out because sine positional embedding needs spatial information
        # h = rearrange(h, "b c h w -> b (h w) c")
        return h  # type: ignore