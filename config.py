import ml_collections
import torch
from path import Path

# NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=3 python main2.py --config config.py --workdir baseline_video
# NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=3 python main.py --config config.py --workdir check1

def get_config():
    """Gets the default hyperparameter configuration."""

    config = ml_collections.ConfigDict()
    config.log_dir = Path('/workspace/joonsm/City_Layout/log_dir')
    # Exp info
    config.dataset_path = Path("/nas2/lait/1000_Members/jjoonvely/carla_new")
    config.train_json = config.dataset_path / 'seg_video.json'
    config.val_json = config.dataset_path / 'seg_video_val.json' 
    config.resume_from_checkpoint = None
    # Training info
    config.seed = 42
    # Model Specific
    config.latent_dim = 256
    config.num_heads = 8
    config.dropout_r = 0.
    config.cond_emb_size =  256
    config.num_layers = 6
    # Training info
    config.log_interval = 100
    config.use_temp= True

    config.optimizer = ml_collections.ConfigDict()
    config.optimizer.num_gpus = torch.cuda.device_count()
    
    config.optimizer.mixed_precision = 'no'
    config.optimizer.gradient_accumulation_steps = 1
    config.optimizer.betas = (0.95, 0.999)
    config.optimizer.epsilon = 1e-8
    config.optimizer.weight_decay = 1e-6

    config.optimizer.lr_scheduler = 'cosine'
    config.optimizer.num_warmup_steps = 2_000
    config.optimizer.lr = 0.0001

    config.optimizer.num_epochs = 30000
    config.optimizer.batch_size = 16
    config.optimizer.split_batches = False
    config.optimizer.num_workers = 8

    config.optimizer.lmb = 5

    if config.optimizer.num_gpus == 0:
        config.device = 'cpu'
    else:
        config.device = 'cuda'
    return config