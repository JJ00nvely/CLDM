from data_utils import norm_bbox
from torch.utils.data import Dataset
import json
import random
import torch
import transformers
from PIL import Image
import os
import numpy as np
import logging
from torchvision import transforms


class ImageLayout(Dataset):
    def __init__(self, file='/nas2/lait/1000_Members/jjoonvely/carla_new/pre_seg_combined.json',type = 'train'):
        if type == 'val':
            file = '/nas2/lait/1000_Members/jjoonvely/carla_new/pre_seg_combined_val.json'
        # 나중에 구축할 데이터셋에 맞게 frmae_list 에 이름 추가하는 부분 수정 필요
        with open(file, 'r') as f:
            self.raw_data = json.load(f)
        self.path = '/nas2/lait/1000_Members/jjoonvely/carla_new'
        self.frame_list = []
        self.box_list = []
        self.transform = transforms.Compose([
        transforms.Resize((360,360)),
        transforms.ToTensor()
        ])
        for idx in self.raw_data.keys():
            if self.raw_data[idx]['type']==[["vehicle.tesla.cybertruck"]]:
                pass
            else:
                for img in self.raw_data[idx]['inpaint_frames_seg']:
                    # Window 에서 전처리하면서 생기는 \\ 처리
                    self.frame_list.append(img.replace('\\','/'))
                for box in self.raw_data[idx]['boxes']:
                    self.box_list.append(box)
    def __len__(self):
        return len(self.frame_list)
    def __getitem__(self, index):
        img_path = os.path.join(self.path,self.frame_list[index])
        img = Image.open(img_path)
        W, H = img.size
        img = self.transform(img)[:3,:,:]
        box = self.box_list[index]
        box = norm_bbox(H, W, box)
        box = np.array(box)
        box = ((box*2)-1)
        box = torch.tensor(box, dtype=torch.float32)       
        sample = {'image' : img , 'box' : box , 'box_cond': box.clone(), 'sr' : img_path}
        return sample
