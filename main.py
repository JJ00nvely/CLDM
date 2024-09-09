import os
import torch
from logger_set import LOG
from absl import flags, app
from accelerate import Accelerator
import wandb
from ml_collections import config_flags
from cgllike import CLDM
from cldm_trainer import TrainLoopCLDM
from utils import set_seed
from dataset import ImageLayout
from diffusers import DDPMScheduler


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", "Training configuration.",
                                lock_config=False)
flags.DEFINE_string("workdir", default='test', help="Work unit directory.")
flags.mark_flags_as_required(["config"])


def main(*args, **kwargs):

    config = init_job()

    LOG.info("Loading data.")

    dataset = ImageLayout(type='train')
    iter_val = ImageLayout(type='val')

    accelerator = Accelerator(
        split_batches=config.optimizer.split_batches,
        gradient_accumulation_steps=config.optimizer.gradient_accumulation_steps,
        mixed_precision=config.optimizer.mixed_precision,
        project_dir=config.log_dir,
    )

    LOG.info(accelerator.state)

    LOG.info("Creating model and diffusion process...")

    # Edit attention layer code for torch.nn.trasformer like
    model = CLDM(latent_dim=config.latent_dim, num_layers = config.num_layers, 
                num_heads=config.num_heads, backbone_name=config.backbone_name
                ).to(accelerator.device)
               
    noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps, prediction_type=config.prediction_type,clip_sample=config.clip_sample)
    
    if accelerator.is_main_process:
        LOG.info("Prediction type : %s ", config.prediction_type) 
        LOG.info("Starting training...")

    TrainLoopCLDM(accelerator=accelerator, model=model, diffusion=noise_scheduler,
                 train_data=dataset, val_data=iter_val,  opt_conf=config.optimizer, save_interval = 10,
                 log_interval=config.log_interval, 
                 device=accelerator.device, resume_from_checkpoint=config.resume_from_checkpoint).train()
    
def init_job():
    config = FLAGS.config
    config.log_dir = config.log_dir / FLAGS.workdir
    config.optimizer.ckpt_dir = config.log_dir / 'checkpoints'
    config.optimizer.samples_dir = config.log_dir / 'samples'

    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.optimizer.samples_dir, exist_ok=True)
    os.makedirs(config.optimizer.ckpt_dir, exist_ok=True)
    set_seed(config.seed)
    wandb.init(project='layoutgen', name=FLAGS.workdir,
               mode='online',
               save_code=True, magic=True, config={k: v for k,v in config.items() if k != 'optimizer'})
    wandb.run.log_code(".")
    wandb.config.update(config.optimizer)
    return config


if __name__ == '__main__':
    app.run(main)