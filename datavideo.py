from data_utils import norm_bbox
from torch.utils.data import Dataset
import json
import random
import torch
import transformers
from PIL import Image
import os
from transformers import AutoImageProcessor
import numpy as np
import logging
from transformers import Dinov2Model

class VideoLayout(Dataset):
    
    def __init__(self, file='/nas2/lait/1000_Members/jjoonvely/carla_new/seg_video.json',type = 'train'):
        # Load raw data from JSON file

        if type == 'val':
            file = '/nas2/lait/1000_Members/jjoonvely/carla_new/seg_video_val.json'

        with open(file, 'r') as f:
            self.raw_data = json.load(f)
        self.path = '/nas2/lait/1000_Members/jjoonvely/carla_new'
        self.frame_list = []
        self.box_list = []
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        # Process the raw data to fill frame_list and box_list
        for idx in self.raw_data.keys():
            frames = self.raw_data[idx]['inpaint_frames_seg']
            boxes = self.raw_data[idx]['boxes']
            num_frames = len(self.raw_data[idx]['seg_frame'])
            if num_frames > 85:
                i = 0
                self.frame_list.append(frames[i:i+16])
                self.box_list.append(boxes[i:i+16])
            else:
                num = (num_frames // 16)
                for i in range(num):
                    self.frame_list.append(frames[i*16:16*(i+1)])
                    self.box_list.append(boxes[i*16:16*(i+1)])
    def __len__(self):
        return len(self.frame_list)
    def __getitem__(self, index):
        frame_list = self.frame_list[index]
        images = []
        fr = []

        for i in frame_list:
            c = i.replace('\\', '/')
            dir = os.path.join(self.path, c)
            with Image.open(dir) as img:

                img_arr = np.array(img)
                
                # Ensure image is in RGBA format
                if img_arr.ndim == 3 and img_arr.shape[2] == 4:
                    # Process images with the image processor
                    src  = self.image_processor(img)['pixel_values'][0]
                    # src = torch.tensor(src)  # Ensure it is a NumPy array
                    
                    # Append to lists
                    images.append(src)
                    fr.append(img_arr)
                else:
                    print(f"Unexpected image format or mode: {img.mode}")

        # Debugging output
        # print(f"Number of images loaded: {len(fr)}")

        if len(fr) != 16:
            raise ValueError(f"Expected 16 frames but got {len(fr)}. Check the data preparation.")
        
        # Get image size from the first NumPy array
        H, W = fr[0].shape[:2]
        
        # Convert images to tensor
        images = torch.stack([torch.tensor(image).float() for image in images])
        
        # Convert fr to uint8 and then to tensor
        fr = torch.stack([torch.tensor(image, dtype=torch.uint8) for image in fr])
        
        # Process box coordinates
        box = self.box_list[index]
        box = [norm_bbox(H, W, bx) for bx in box]
        box = np.array(box)
        box = ((box * 2) - 1)
        box = torch.tensor(box, dtype=torch.float32)
        
        # Create and return the sample
        sample = {'image': images, 'box': box, 'box_cond': box.clone(), 'sr': fr}
        return sample