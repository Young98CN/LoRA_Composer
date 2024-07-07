import os
import random
import re
import torch
from torch.utils.data import Dataset
from mixofshow.data.prompt_dataset import PromptDataset

from mixofshow.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class BG_PromptDataset(PromptDataset):
    'A simple dataset to prepare the prompts to generate class images on multiple GPUs.'

    def __init__(self, opt):
        super().__init__(opt)

    def __len__(self):
        return len(self.prompts_to_generate)

    def __getitem__(self, index):
        prompt, indice = self.prompts_to_generate[index]
        example = {}
        example['prompts'] = prompt
        example['indices'] = indice
        return example
