from transformers import AutoImageProcessor
import torch

class DinoImageProcessor:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    
    def __call__(self,item):
        return torch.tensor(self.processor(item)['pixel_values'][0])