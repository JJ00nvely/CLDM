import math
import os
import random
import shutil
import datetime
import numpy as np
import torch
import wandb
import logging
from PIL import Image, ImageDraw
from accelerate import Accelerator
from diffusers import get_scheduler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from diffusers import DDPMScheduler
import torch.nn.functional as F
from logger_set import LOG
import time
import cv2
from transformers import AutoImageProcessor, Dinov2Model
from einops import rearrange
import io
import tempfile
from loss import bbox_pair_tv_loss
from einops import rearrange
from util.video import video_make

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

# sample 하는 부분 추가 처리 ?.
# from utils import plot_sample

class TrainLoopCLDM:
    def __init__(self, accelerator: Accelerator, model, diffusion: DDPMScheduler, train_data,val_data,
                 opt_conf, save_interval,
                 log_interval: int,
                 device: str = 'cpu',
                 resume_from_checkpoint: str = None):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.accelerator = accelerator
        self.opt_conf = opt_conf
        self.save_interval = save_interval
        self.diffusion = diffusion
        self.log_interval = log_interval
        self.device = self.accelerator.device
        self.model.deivce = self.accelerator.device

        optimizer = torch.optim.AdamW(model.parameters(), lr=opt_conf.lr, betas=opt_conf.betas,
                                      weight_decay=opt_conf.weight_decay, eps=opt_conf.epsilon)
        train_loader = DataLoader(train_data, batch_size=opt_conf.batch_size,
                                  shuffle=True, num_workers=opt_conf.num_workers)
        val_loader = DataLoader(val_data, batch_size=4,
                                shuffle=False, num_workers=opt_conf.num_workers)
        
        lr_scheduler = get_scheduler(opt_conf.lr_scheduler,
                                     optimizer,
                                     num_warmup_steps=opt_conf.num_warmup_steps * opt_conf.gradient_accumulation_steps,
                                     num_training_steps=(len(train_loader) * opt_conf.num_epochs))
        
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler = self.accelerator.prepare(
            model, optimizer, train_loader, val_loader, lr_scheduler
        )


        LOG.info((model.device, self.device))

        self.total_batch_size = opt_conf.batch_size * accelerator.num_processes * opt_conf.gradient_accumulation_steps
        self.num_update_steps_per_epoch = math.ceil(len(train_loader) / opt_conf.gradient_accumulation_steps)
        self.max_train_steps = opt_conf.num_epochs * self.num_update_steps_per_epoch

        if self.accelerator.is_main_process:
            LOG.info("***** Running training *****")
            LOG.info(f"  Num examples = {len(train_data)}")
            LOG.info(f"  Num Epochs = {opt_conf.num_epochs}")
            LOG.info(f"  Instantaneous batch size per device = {opt_conf.batch_size}")
            LOG.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
            LOG.info(f"  Gradient Accumulation steps = {opt_conf.gradient_accumulation_steps}")
            LOG.info(f"  Total optimization steps = {self.max_train_steps}")

        self.global_step = 0
        self.first_epoch = 0
        self.resume_from_checkpoint = False
        self.start_time =time.time()


    def train(self):
        train_val = iter(self.train_dataloader)
        iter_val = iter(self.val_dataloader)

        for epoch in range(self.first_epoch, self.opt_conf.num_epochs):
            self.train_epoch(epoch)
            LOG.info(f"Current Epoch={epoch}")
            if epoch % 2 == 0:
                sample1 = next(train_val)
                sample2 = next(iter_val)
                img_bbox, img_bbox1, img_bbox2, img_bbox3 = self.generate_images(sample1)
                img_bbox_val, img_bbox1_val, img_bbox2_val, img_bbox3_val = self.generate_val(sample2)
                # 각 이미지 리스트를 비디오 파일로 변환

                local_video_dir = './videos'
                local_video_dir_val = './videos_val'

                video_make(img_bbox,local_video_dir,1,epoch)
                video_make(img_bbox1,local_video_dir,2,epoch)
                video_make(img_bbox2,local_video_dir,3,epoch)
                video_make(img_bbox3,local_video_dir,4,epoch)
                video_make(img_bbox_val,local_video_dir_val,1,epoch)
                video_make(img_bbox1_val,local_video_dir_val,2,epoch)
                video_make(img_bbox2_val,local_video_dir_val,3,epoch)
                video_make(img_bbox3_val,local_video_dir_val,4,epoch)
            else:
                pass

            
    def sample2dev(self, sample):
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                sample[k] = v.to(self.device)
            elif isinstance(v, dict):
                for k1, v1 in v.items():
                    if isinstance(v1, torch.Tensor):
                        sample[k][k1] = v1.to(self.device)

                        
    def train_epoch(self, epoch):
        self.model.train()
        device = self.model.device

        progress_bar = tqdm(total=self.num_update_steps_per_epoch, disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        losses = {} 

        for step, batch in enumerate(self.train_dataloader):
            # Skip steps until we reach the resumed step
            if self.resume_from_checkpoint and epoch == self.first_epoch and step < self.resume_step:
                if step % self.opt_conf.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            self.sample2dev(batch)
            batch['image'] = rearrange(batch['image'], 'b f c w h -> (b f) c w h' )
            batch['box'] = rearrange(batch['box'], 'b f h -> (b f) h' )
            # Sample noise that we'll add to the boxes
            noise = torch.randn(batch['box'].shape).to(device)
            bsz = batch['box'].shape[0] 
            # Sample a random timestep for each layout
            
            t = torch.randint(
                0, self.diffusion.config.num_train_timesteps, (bsz,), device=device
            ).long()

            # Noise part in here 
            diff_box   = self.diffusion.add_noise(batch['box'], noise ,t)
            # rewrite box with noised version, origina l box is still in batch['box_cond']
            batch['box'] = diff_box

            # to(device)
            # Run the model on the noisy layouts
            with self.accelerator.accumulate(self.model):

                noise_pred= self.model(batch, t)
                # noise or sample ? 
                loss_mse = F.mse_loss(rearrange(batch['box_cond'], 'b c d -> (b c) d'), noise_pred)
                # tv_loss = bbox_pair_tv_loss(noise_pred, frames=16, tv_weight=0.05)
                loss = loss_mse #+ tv_loss
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()

                self.optimizer.zero_grad()
            losses.setdefault("loss_mse", []).append(loss_mse.detach().item())
            # losses.setdefault("tv_loss", []).append(tv_loss.detach().item())
            
            if self.accelerator.sync_gradients & self.accelerator.is_main_process:
                progress_bar.update(1)
                self.global_step += 1
                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0],
                        "step": self.global_step} # "tv_loss":tv_loss.detach().item(),
                progress_bar.set_postfix(**logs)

            if self.global_step % self.log_interval == 0:
                wandb.log({k: np.mean(v) for k, v in losses.items()}, step=self.global_step)
                wandb.log({"lr": self.lr_scheduler.get_last_lr()[0]}, step=self.global_step)

        progress_bar.close()
        self.accelerator.wait_for_everyone()

        save_path = self.opt_conf.ckpt_dir / f"checkpoint-{epoch}/"
        # delete folder if we have already 5 checkpoints
        if self.opt_conf.ckpt_dir.exists():
            ckpts = list(self.opt_conf.ckpt_dir.glob("checkpoint-*"))
            # sort by epoch
            ckpts = sorted(ckpts, key=lambda x: int(x.name.split("-")[1]))
            # 20개 이상일 때 가장 옛날거 제거 
            if len(ckpts) > 20:
                LOG.info(f"Deleting checkpoint {ckpts[0]}")
                shutil.rmtree(ckpts[0])
        if epoch%self.save_interval==0 or epoch==self.opt_conf.num_epochs-1 & self.accelerator.is_main_process:
            self.accelerator.save_model(self.model, save_path, "50GB", safe_serialization=False)
        # self.model.save_pretrained(save_path) 
        LOG.info(f"Saving checkpoint to {save_path}")
        
        # unwrapped_model = self.accelerator.unwrap_model(self.model)
        # save_state_dict = self.accelerator.get_state_dict(self.model)
        # unwrapped_model.save_pretrained(f'test{self.global_step}',is_main_process=self.accelerator.is_main_process, max_shard_size=10,
        #                                         save_function=self.accelerator.save,
        #                                         state_dict=save_state_dict)
        # # 이렇게 저장할 시 추후 DDP 래퍼를 처리해야할 수 있다고 하니까 만약 문제가 생길 시 해당 부분 처리할 수 있도록해야함
        # self.accelerator.save_state(save_path)
        # self.model.save_pretrained(save_path, max_shard_size=5)
        LOG.info(f"Saving checkpoint to {save_path}")


####################################################################################################################################################################################

    # Validation 체크를 위한 code 작성
    def generate_images(self, sample):
        sample1 = {'image': sample['image'][0], 'box':sample['box'][0], 'sr':sample['sr'][0]}
        sample2 = {'image': sample['image'][2], 'box':sample['box'][2], 'sr':sample['sr'][2]}
        sample3 = {'image': sample['image'][4], 'box':sample['box'][4], 'sr':sample['sr'][4]}
        sample4 = {'image': sample['image'][6], 'box':sample['box'][6], 'sr':sample['sr'][6]}

        predicted = self.sample_from_model(sample1)
        predicted_ = self.sample_from_model(sample2)
        predicted__ = self.sample_from_model(sample3)
        predicted___ = self.sample_from_model(sample4)

        original_box = sample['box_cond'][0]
        original_box_ = sample['box_cond'][2]
        original_box__ = sample['box_cond'][4]
        original_box___ = sample['box_cond'][6]

        if sample['sr'][0].is_cuda:
            src = sample['sr'][0].cpu().numpy()
            src1 = sample['sr'][2].cpu().numpy()
            src2 = sample['sr'][4].cpu().numpy()
            src3 = sample['sr'][6].cpu().numpy()
        else:
            src = sample['sr'][0].numpy()
            src1 = sample['sr'][2].numpy()
            src2 = sample['sr'][4].numpy()
            src3 = sample['sr'][6].numpy()

        src_list = []
        src_list1 = []
        src_list2 = []
        src_list3 = []

        for i in src:
            src_ = Image.fromarray(i, 'RGBA')
            src_list.append(src_)
        box  = predicted
        box = box.cpu().numpy()
        box = (box + 1) / 2

        box_  = original_box
        box_ = box_.cpu().numpy()
        box_ = (box_ + 1) / 2

        images_with_bbox  = []
        for i in range(box.shape[0]):
            source =src_list[i].copy()
            width, height =source.size
            cx, cy, w, h =  box[i]
            ocx,ocy,ow,oh = box_[i]
            x = int((cx - w / 2) * width)
            y = int((cy - h / 2) * height)
            x2 = int((cx + w / 2) * width)
            y2 = int((cy + h / 2) * height)

            ox = int((ocx - ow / 2) * width)
            oy = int((ocy - oh / 2) * height)
            ox2 = int((ocx + ow / 2) * width)
            oy2 = int((ocy + oh / 2) * height)
            draw = ImageDraw.Draw(source)
            draw.rectangle([x, y, x2, y2], outline="red", width=1)
            draw.rectangle([ox, oy, ox2, oy2], outline="blue", width=1)
            images_with_bbox.append(source)

        for i in src1:
            src_ = Image.fromarray(i, 'RGBA')
            src_list1.append(src_)
        box  = predicted_
        box = box.cpu().numpy()
        box = (box + 1) / 2

        box_  = original_box_
        box_ = box_.cpu().numpy()
        box_ = (box_ + 1) / 2

        images_with_bbox1  = []
        for i in range(box.shape[0]):
            source =src_list1[i].copy()
            width, height =source.size
            cx, cy, w, h =  box[i]
            ocx,ocy,ow,oh = box_[i]
            x = int((cx - w / 2) * width)
            y = int((cy - h / 2) * height)
            x2 = int((cx + w / 2) * width)
            y2 = int((cy + h / 2) * height)
        
            ox = int((ocx - ow / 2) * width)
            oy = int((ocy - oh / 2) * height)
            ox2 = int((ocx + ow / 2) * width)
            oy2 = int((ocy + oh / 2) * height)

            draw = ImageDraw.Draw(source)
            draw.rectangle([x, y, x2, y2], outline="red", width=1)
            draw.rectangle([ox, oy, ox2, oy2], outline="blue", width=1)
            images_with_bbox1.append(source)

        for i in src2:
            src_ = Image.fromarray(i, 'RGBA')
            src_list2.append(src_)
        box  = predicted__
        box = box.cpu().numpy()
        box = (box + 1) / 2

        box_  = original_box__
        box_ = box_.cpu().numpy()
        box_ = (box_ + 1) / 2

        images_with_bbox2  = []
        for i in range(box.shape[0]):
            source =src_list2[i].copy()
            width, height =source.size
            cx, cy, w, h =  box[i]
            ocx,ocy,ow,oh = box_[i]

            x = int((cx - w / 2) * width)
            y = int((cy - h / 2) * height)
            x2 = int((cx + w / 2) * width)
            y2 = int((cy + h / 2) * height)

            ox = int((ocx - ow / 2) * width)
            oy = int((ocy - oh / 2) * height)
            ox2 = int((ocx + ow / 2) * width)
            oy2 = int((ocy + oh / 2) * height)

            draw = ImageDraw.Draw(source)
            draw.rectangle([x, y, x2, y2], outline="red", width=1)
            draw.rectangle([ox, oy, ox2, oy2], outline="blue", width=1)
            images_with_bbox2.append(source)
            
        for i in src3:
            src_ = Image.fromarray(i, 'RGBA')
            src_list3.append(src_)

        box  = predicted___
        box = box.cpu().numpy()
        box = (box + 1) / 2

        box_  = original_box___
        box_ = box_.cpu().numpy()
        box_ = (box_ + 1) / 2

        images_with_bbox3  = []
        for i in range(box.shape[0]):
            source =src_list3[i].copy()
            width, height =source.size
            cx, cy, w, h =  box[i]
            ocx,ocy,ow,oh = box_[i]
            x = int((cx - w / 2) * width)
            y = int((cy - h / 2) * height)
            x2 = int((cx + w / 2) * width)
            y2 = int((cy + h / 2) * height)

            ox = int((ocx - ow / 2) * width)
            oy = int((ocy - oh / 2) * height)
            ox2 = int((ocx + ow / 2) * width)
            oy2 = int((ocy + oh / 2) * height)
    
            draw = ImageDraw.Draw(source)
            draw.rectangle([x, y, x2, y2], outline="red", width=1)
            draw.rectangle([ox, oy, ox2, oy2], outline="blue", width=1)
            images_with_bbox3.append(source)
        return images_with_bbox,  images_with_bbox1, images_with_bbox2, images_with_bbox3

    def generate_val(self,sample):
        sample1 = {'image': sample['image'][0], 'box':sample['box'][0], 'sr':sample['sr'][0]}
        sample2 = {'image': sample['image'][1], 'box':sample['box'][1], 'sr':sample['sr'][1]}
        sample3 = {'image': sample['image'][2], 'box':sample['box'][2], 'sr':sample['sr'][2]}
        sample4 = {'image': sample['image'][3], 'box':sample['box'][3], 'sr':sample['sr'][3]}
        predicted = self.sample_from_model(sample1)
        predicted_ = self.sample_from_model(sample2)
        predicted__ = self.sample_from_model(sample3)
        predicted___ = self.sample_from_model(sample4)
        if sample['sr'][0].is_cuda:
            src = sample['sr'][0].cpu().numpy()
            src1 = sample['sr'][1].cpu().numpy()
            src2 = sample['sr'][2].cpu().numpy()
            src3 = sample['sr'][3].cpu().numpy()
        else:
            src = sample['sr'][0].numpy()
            src1 = sample['sr'][1].numpy()
            src2 = sample['sr'][2].numpy()
            src3 = sample['sr'][3].numpy()

        src_list = []
        src_list1 = []
        src_list2 = []
        src_list3 = []

        for i in src:
            src_ = Image.fromarray(i, 'RGBA')
            src_list.append(src_)
        box  = predicted
        box = box.cpu().numpy()
        box = (box + 1) / 2
        images_with_bbox  = []
        for i in range(box.shape[0]):
            source =src_list[i].copy()
            width, height =source.size
            cx, cy, w, h =  box[i]
            x = int((cx - w / 2) * width)
            y = int((cy - h / 2) * height)
            x2 = int((cx + w / 2) * width)
            y2 = int((cy + h / 2) * height)
            draw = ImageDraw.Draw(source)
            draw.rectangle([x, y, x2, y2], outline="red", width=2)
            images_with_bbox.append(source)

        for i in src1:
            src_ = Image.fromarray(i, 'RGBA')
            src_list1.append(src_)
        box  = predicted_
        box = box.cpu().numpy()
        box = (box + 1) / 2
        images_with_bbox1  = []
        for i in range(box.shape[0]):
            source =src_list1[i].copy()
            width, height =source.size
            cx, cy, w, h =  box[i]
            x = int((cx - w / 2) * width)
            y = int((cy - h / 2) * height)
            x2 = int((cx + w / 2) * width)
            y2 = int((cy + h / 2) * height)
            draw = ImageDraw.Draw(source)
            draw.rectangle([x, y, x2, y2], outline="red", width=2)
            images_with_bbox1.append(source)

        for i in src2:
            src_ = Image.fromarray(i, 'RGBA')
            src_list2.append(src_)
        box  = predicted__
        box = box.cpu().numpy()
        box = (box + 1) / 2
        images_with_bbox2  = []
        for i in range(box.shape[0]):
            source =src_list2[i].copy()
            width, height =source.size
            cx, cy, w, h =  box[i]
            x = int((cx - w / 2) * width)
            y = int((cy - h / 2) * height)
            x2 = int((cx + w / 2) * width)
            y2 = int((cy + h / 2) * height)
            draw = ImageDraw.Draw(source)
            draw.rectangle([x, y, x2, y2], outline="red", width=2)
            images_with_bbox2.append(source)
            
        for i in src3:
            src_ = Image.fromarray(i, 'RGBA')
            src_list3.append(src_)
        box  = predicted___
        box = box.cpu().numpy()
        box = (box + 1) / 2
        images_with_bbox3  = []
        for i in range(box.shape[0]):
            source =src_list3[i].copy()
            width, height =source.size
            cx, cy, w, h =  box[i]
            x = int((cx - w / 2) * width)
            y = int((cy - h / 2) * height)
            x2 = int((cx + w / 2) * width)
            y2 = int((cy + h / 2) * height)
            draw = ImageDraw.Draw(source)
            draw.rectangle([x, y, x2, y2], outline="red", width=2)
            images_with_bbox3.append(source)
        return images_with_bbox,  images_with_bbox1, images_with_bbox2, images_with_bbox3
    
    
    def sample_from_model(self, sample):

        # sample['image'] = rearrange(batch['image'], 'b f c w h -> (b f) c w h' )
        # sample['box'] = rearrange(batch['box'], 'b f h -> (b f) h' )
        shape = sample['box'].shape
        model = self.accelerator.unwrap_model(self.model)
        model.eval()

        noisy_batch = {
            'image':sample['image'],
            'box':torch.rand(*shape, dtype=torch.float32, device=self.device)        
        }


        for i in range(self.diffusion.num_train_timesteps)[::-1]:

            t = torch.tensor([i]*shape[0], device=self.device)

            with torch.no_grad():
                # denoise for step t.
                noise_pred = model(noisy_batch, timesteps=t)
                bbox_pred = self.diffusion.step(noise_pred, t[0].detach().item(),  noisy_batch['box'], return_dict=True)
                
                # updata denoised box
                noisy_batch['box'] = bbox_pred.prev_sample
        
        return bbox_pred.pred_original_sample
    
    def init_optimizer(self, model):
        p_wd, p_non_wd = [], []
        num_parameters = 0
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n: 
                p_non_wd.append(p)
            else:
                p_wd.append(p)
            num_parameters += p.data.nelement()
        self.logger.info("number of trainable parameters: %d" % num_parameters)
        optim_params = [
            {
                "params": p_wd,
                "weight_decay": self.weight_decay,
            },
            {"params": p_non_wd, "weight_decay": 0},
        ]
        optimizer = torch.optim.AdamW(
            optim_params,
            lr=self.lr,
            weight_decay= self.weight_decay,
            betas=(self.beta1, self.beta2),
        )
        return optimizer