# for inpainting methods，generate background images from sepcific seeds

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os, random, json, logging
import sys
sys.path.append("/data/yangyang/mixofshow")
from mixofshow.data import build_dataloader, build_dataset
from mixofshow.models import build_model
from mixofshow.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from mixofshow.utils.options import dict2str, parse_options
from torch.utils.data.distributed import DistributedSampler
from os import path as osp

# base_model inference
# model_base = "/data/yangyang/mixofshow/experiments/pretrained_models/anything-v4.0"
# pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16, use_safetensors=True)

# scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.scheduler = scheduler

# pipe.to("cuda")




# parse options, set distributed setting, set ramdom seed
opt, _ = parse_options("/data/yangyang/mixofshow", is_train=False)

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

# mkdir and initialize loggers
make_exp_dirs(opt)
log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
logger = get_root_logger(logger_name='mixofshow', log_level=logging.INFO, log_file=log_file)
logger.info(get_env_info())
logger.info(dict2str(opt))

# create test dataset and dataloader
test_loaders = []
for _, dataset_opt in sorted(opt['datasets'].items()):
    test_set = build_dataset(dataset_opt)
    test_sampler = DistributedSampler(test_set, opt['world_size'], opt['rank'], shuffle=False)
    test_loader = build_dataloader(
        test_set,
        dataset_opt,
        num_gpu=opt['num_gpu'],
        dist=opt['dist'],
        sampler=test_sampler,
        seed=opt['manual_seed'])
    logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
    test_loaders.append(test_loader)

# create model
model = build_model(opt)


for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info(f'Testing {test_set_name}...')
    model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'], seed=opt['val']['seed'])