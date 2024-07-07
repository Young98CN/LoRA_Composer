# for inpainting methods，generate background images from sepcific seeds

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os, random, json, logging
import sys
sys.path.append("/data/yangyang/mixofshow")
from torch.utils.data.distributed import DistributedSampler
from os import path as osp

cfg = json.load(open("/data/yangyang/mixofshow/eval_cfg/gen_bg/gen_real.json", "r"))

model_path = cfg[0]["base_model_path"]
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, use_safetensors=True)

scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = scheduler
pipe.safety_checker = None

pipe.to("cuda")

for idx in range(1, len(cfg)):
    random.seed(cfg[idx]["seed"])  # 设置随机数种子
    random_seed = [random.randint(1, 20000) for _ in range(50)]

    for i in random_seed:
        image = pipe(
        cfg[idx]["prompt"], 
        negative_prompt=cfg[idx]["neg_prompt"],
        num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=1, generator=torch.Generator().manual_seed(i), height=768, width=1024).images
        output_folder = osp.join("/data/yangyang/mixofshow/results/bg", cfg[idx]["exp_name"])
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        image[0].save("{}/{}.png".format(output_folder, i))