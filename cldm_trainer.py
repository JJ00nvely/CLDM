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
from loss import giou


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
                                shuffle=True, num_workers=opt_conf.num_workers)
        
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

        if resume_from_checkpoint:
            LOG.print(f"Resuming from checkpoint {resume_from_checkpoint}")
            accelerator.load_state(resume_from_checkpoint)
            last_epoch = int(resume_from_checkpoint.split("-")[1])
            self.global_step = last_epoch * self.num_update_steps_per_epoch
            self.first_epoch = last_epoch
            self.resume_step = 0
            
        self.start_time =time.time()


    def train(self):
        train_val = iter(self.train_dataloader)
        iter_val = iter(self.val_dataloader)
        for epoch in range(self.first_epoch, self.opt_conf.num_epochs):
            self.train_epoch(epoch)
            LOG.info(f"Current Epoch={epoch}")
            if epoch% 1 ==0:
                sample1 = next(train_val)
                sample2 = next(iter_val)
                img_bbox = self.generate_images(sample1)
                img_iter = self.generate_iteratvie(sample2)
                wandb_images = [wandb.Image(img, caption=f'pred_{i}') for i, img in enumerate(img_bbox)]
                wandb_images_iter = [wandb.Image(img, caption=f'pred_{i}') for i, img in enumerate(img_iter)]
                # wandb_images_iter = [wandb.Image(img, caption=f'iter_pred') for img in img_iter]
                wandb.log({"pred": wandb_images})
                wandb.log({"iter_pred": wandb_images_iter})
            else:
                pass
        self.accelerator.end_training()
        self.accelerator.wait_for_everyone()
        total_time = time.time() - self.start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        LOG.info(f"Training Time {total_time_str}")

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

            # Sample noise that we'll add to the boxes
            noise = torch.randn(batch['box'].shape).to(device)
            bsz = batch['box'].shape[0] 
            # Sample a random timestep for each layout
            t = torch.randint(
                0, self.diffusion.config.num_train_timesteps, (bsz,), device=device
            ).long()

            # Noise part in here 
            diff_box   = self.diffusion.add_noise(original_samples= batch['box'], timesteps=t, noise=noise)
            # rewrite box with noised version, origina l box is still in batch['box_cond']
            batch['box'] = diff_box

            # to(device)
            # Run the model on the noisy layouts
            with self.accelerator.accumulate(self.model):
                noise_pred= self.model(batch, t)
                # Change for Predict Box
                loss_giou = 1 - giou(batch['box_cond'], noise_pred)
                loss_giou = loss_giou.sum() / bsz
                loss_mse = F.mse_loss(batch['box_cond'], noise_pred)
                loss = 0.1* loss_giou+  loss_mse
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            losses.setdefault("MSE", []).append(loss_mse.detach().item())
            losses.setdefault("GIOU", []).append(loss_giou.detach().item())

            if self.accelerator.sync_gradients & self.accelerator.is_main_process:
                progress_bar.update(1)
                self.global_step += 1
                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0],
                        "step": self.global_step}
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
        sample = {'image': sample['image'][:8], 'box':sample['box'][:8], 'sr':sample['sr'][:8], 'box_cond': sample['box_cond'][:8]}
        predicted = self.sample_from_model(sample)
        src = sample['sr']
        src_list = []
        for i in src:
            src_ = Image.open(i)
            src_list.append(src_)
        box  = predicted
        box = box.cpu().numpy()
        box = (box + 1) / 2
        original_box = sample['box_cond'].cpu().numpy()
        original_box = (original_box + 1) / 2
        images_with_bbox  = []
        for i in range(box.shape[0]):
            source =src_list[i].copy()
            width, height =source.size
            cx, cy, w, h =  box[i]
            ocx,ocy,ow,oh = original_box[i]
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
        return images_with_bbox

    def generate_iteratvie(self,sample):
        b_list = []
        b_list_ = []
        b_list__ = []
        b_list___ = []
        sample_ = {'image': sample['image'][0:1], 'box':sample['box'][0:1], 'sr':sample['sr'][0:1]}
        sample__ = {'image': sample['image'][1:2], 'box':sample['box'][1:2], 'sr':sample['sr'][1:2]}
        sample___ = {'image': sample['image'][2:3], 'box':sample['box'][2:3], 'sr':sample['sr'][2:3]}
        sample____ = {'image': sample['image'][3:4], 'box':sample['box'][3:4], 'sr':sample['sr'][3:4]}
        images_with_bbox  = []

        for i in range(5):
            predicted_ = self.sample_from_model(sample_)
            b_list.append(predicted_)
        box = torch.stack(b_list).squeeze(1)
        box =  box.cpu().numpy()
        box = (box + 1) / 2
        src = Image.open(sample_['sr'][0])
        width, height =src.size
        for i in range(len(box)):
            cx, cy, w, h =  box[i]
            x = int((cx - w / 2) * width)
            y = int((cy - h / 2) * height)
            x2 = int((cx + w / 2) * width)
            y2 = int((cy + h / 2) * height)
            draw = ImageDraw.Draw(src)
            draw.rectangle([x, y, x2, y2], outline="red", width=2)
        images_with_bbox.append(src)

        for i in range(5):
            predicted_ = self.sample_from_model(sample__)
            b_list_.append(predicted_)
        box = torch.stack(b_list_).squeeze(1)
        box =  box.cpu().numpy()
        box = (box + 1) / 2
        src1 = Image.open(sample__['sr'][0])
        width, height =src1.size
        for i in range(len(box)):
            cx, cy, w, h =  box[i]
            x = int((cx - w / 2) * width)
            y = int((cy - h / 2) * height)
            x2 = int((cx + w / 2) * width)
            y2 = int((cy + h / 2) * height)
            draw = ImageDraw.Draw(src1)
            draw.rectangle([x, y, x2, y2], outline="red", width=2)
        images_with_bbox.append(src1)

        for i in range(5):
            predicted_ = self.sample_from_model(sample___)
            b_list__.append(predicted_)
        box = torch.stack(b_list__).squeeze(1)
        box =  box.cpu().numpy()
        box = (box + 1) / 2
        src2 = Image.open(sample___['sr'][0])
        width, height =src2.size
        for i in range(len(box)):
            cx, cy, w, h =  box[i]
            x = int((cx - w / 2) * width)
            y = int((cy - h / 2) * height)
            x2 = int((cx + w / 2) * width)
            y2 = int((cy + h / 2) * height)
            draw = ImageDraw.Draw(src2)
            draw.rectangle([x, y, x2, y2], outline="red", width=2)
        images_with_bbox.append(src2)

        for i in range(5):
            predicted_ = self.sample_from_model(sample____)
            b_list___.append(predicted_)
        box = torch.stack(b_list___).squeeze(1)
        box =  box.cpu().numpy()
        box = (box + 1) / 2
        src3 = Image.open(sample____['sr'][0])
        width, height =src3.size
        for i in range(len(box)):
            cx, cy, w, h =  box[i]
            x = int((cx - w / 2) * width)
            y = int((cy - h / 2) * height)
            x2 = int((cx + w / 2) * width)
            y2 = int((cy + h / 2) * height)
            draw = ImageDraw.Draw(src3)
            draw.rectangle([x, y, x2, y2], outline="red", width=2)
        images_with_bbox.append(src3)

        return images_with_bbox      
    
    def sample_from_model(self, sample):
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
                # print(noise_pred)

                box_pred = self.diffusion.step(noise_pred, t[0].detach().item(), noisy_batch['box'], return_dict=True)

                # updata denoised box

                noisy_batch['box'] = box_pred.prev_sample

        
        return box_pred.pred_original_sample
