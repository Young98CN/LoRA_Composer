from diffusers.models.cross_attention import CrossAttention
import abc

import cv2
import numpy as np
import torch

from PIL import Image
from typing import Union, Tuple, List, Dict, Optional
import torch.nn.functional as nnf
import os, shutil
import matplotlib.pyplot as plt

# def aggregate_attention(
#         controller, res: int, from_where: List[str], only_cross: bool, select: int
#     ):
#         out_cross = []
#         out_self = []
#         attention_maps = controller.attention_store
#         # num_pixels = res**2
#         num_pixels = res * res * 2
#         for location in from_where:
#             if not only_cross:
#                 for item in attention_maps[f"{location}_self"]:
#                     if item.shape[1] == num_pixels:
#                         self_maps = item.view(-1, res, res*2, res, res*2)
#                         out_self.append(self_maps)
#             if len(controller.region_prompt) == 0:  # 普通cross attention
#                 for item in attention_maps[f"{location}_cross"]:
#                     if item.shape[1] == num_pixels:
#                         cross_maps = item.reshape(
#                             1, -1, res, res * 2, item.shape[-1]
#                         )[select]
#                         out_cross.append(cross_maps)
#             else:  # region lora corss-attention
#                 for item in attention_maps[f"{location}_cross_region"]:
#                     if item.shape[1] == num_pixels:
#                         cross_maps = item.reshape(
#                             1, -1, res, res * 2, item.shape[-1]
#                         )[select]
#                         out_cross.append(cross_maps)
        
#         if not only_cross:
#             out_self = torch.cat(out_self, dim=0)  # res*res*2, res*res*2 的那一层self-attn
#             out_self = out_self.sum(0) / out_self.shape[0]
        
#         if len(controller.region_prompt) == 0:     
#             out_cross = torch.cat(out_cross, dim=0)  # res * res*2 的那一层特征图
#             out_cross = out_cross.sum(0) / out_cross.shape[0]
#             return out_cross, out_self
#         else:  # region lora corss-attention
#             items_out_cross = []
#             for item_idx in range(len(controller.region_prompt)):
#                 item_atten_map = torch.zeros_like(cross_maps)
#                 for layer_atten_idx in range(item_idx, len(out_cross), len(controller.region_prompt)):  # item在不同decoder层中的attention map相加
#                     item_atten_map += out_cross[layer_atten_idx]
#                 items_out_cross.append(item_atten_map.mean(0))
#             return items_out_cross, out_self

def aggregate_attention(
        controller, from_where: List[str], only_cross: bool, select: int
    ):
        out_cross_region = []
        out_cross = []
        out_self = []
        attention_maps = controller.attention_store
        # num_pixels = res**2
        res = controller.resolution
        num_pixels = res[0] * res[1]
        for location in from_where:
            if not only_cross:
                for item in attention_maps[f"{location}_self"]:
                    if item.shape[1] == num_pixels:
                        self_maps = item.view(-1, res[0], res[1], res[0], res[1])
                        out_self.append(self_maps)
            # bg cross attention
            for item in attention_maps[f"{location}_cross"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(
                        1, -1, res[0], res[1], item.shape[-1]
                    )[select]
                    out_cross.append(cross_maps)
            if len(controller.region_prompt) != 0:  # region lora corss-attention
                for item in attention_maps[f"{location}_cross_region"]:
                    if item.shape[1] == num_pixels:
                        cross_maps = item.reshape(
                            1, -1, res[0], res[1], item.shape[-1]
                        )[select]
                        out_cross_region.append(cross_maps)  # 长度为(2+3)*n_concepts, 2down_block, 3up_block
        
        if not only_cross:
            out_self = torch.cat(out_self, dim=0)  # res*res*2, res*res*2 的那一层self-attn
            out_self = out_self.sum(0) / out_self.shape[0]
        
        out_cross_bg = torch.cat(out_cross, dim=0)  # res * res*2 的那一层特征图
        out_cross_bg = out_cross_bg.sum(0) / out_cross_bg.shape[0]
        
        if len(controller.region_prompt) != 0:  # region lora corss-attention
            items_out_cross = []
            for item_idx in range(len(controller.region_prompt)):
                item_atten_map = torch.zeros_like(cross_maps)
                for layer_atten_idx in range(item_idx, len(out_cross_region), len(controller.region_prompt)):  # item在不同decoder层中的attention map相加
                    item_atten_map += out_cross_region[layer_atten_idx]
                items_out_cross.append(item_atten_map.mean(0) / (len(out_cross_region) / len(controller.region_prompt)))
        return items_out_cross, out_self, out_cross_bg



class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str, is_region: bool=False):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str, is_region: bool=False):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet, is_region)  # 更新latent使用
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, num_att_layers):
        self.cur_step = 0
        self.num_att_layers = num_att_layers
        self.cur_att_layer = 0
        self.self_attn_mask_list = None  # 用于保存self attn 人物mask

class AttentionStore(AttentionControl):
    def __init__(self, mode='all', resolution=[16,32]):
        if 'self' == mode or 'cross' == mode:
            num_att_layers = 16
        else:
            num_att_layers = 32
        super(AttentionStore, self).__init__(num_att_layers)
        self.step_store = self.get_empty_store()
        self.attn_mask_store = None
        self.attention_store = {}
        self.region_prompt = []  # 存储region_prompt
        self.all_step_store = {}  # 存储所有step的attention map
        self.resolution = resolution  # 定义存储的attention 分辨率
        self.resolution_flat = resolution[0] * resolution[1]  # 定义存储的attention 分辨率
    
    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
            "down_cross_region": [],
            "mid_cross_region": [],
            "up_cross_region": [],
        }

    def forward(self, attn, is_cross: bool, place_in_unet: str, is_region: bool=False):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}{'_region' if is_region else ''}"
        # if attn.shape[1] <= 32**2:
        if attn.shape[1] == self.resolution_flat:  # 这里会记录x4一层的2个up和3个down block，不会记录mid
            self.step_store[key].append(attn)  # 记录16*32的attention map
        return attn

    def between_steps(self):
        self.attention_store = self.step_store  # NOTE 这里与原先不一样，直接将每一个step复写attention_store，不像之前的算均值
        step_attn = self.get_empty_store()
        for k,v in self.step_store.items():
            step_attn[k] = [i.clone().detach().cpu() for i in v]
        self.all_step_store[self.cur_step] = step_attn  # 存储每一步的attention map

        self.step_store = self.get_empty_store()

    def save_self_mask(self, seq_len, mask_list):
        # key = f"{place_in_unet}_self_mask"
        if seq_len == self.resolution_flat:
            self.attn_mask_store = mask_list  # 记录16*32的mask
            
    def save_region_prompt(self, region_prompt):
        self.region_prompt.append(region_prompt)

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

def text_under_image(
    image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * 0.2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img

def draw_rectangle(image, mask, refector=16):
    # 使用numpy的where函数找到掩码的非零元素的位置
    rows, cols = np.where(mask)
    # 计算掩码的边界框，乘缩放倍数
    min_row, max_row = min(rows)*refector, max(rows)*refector,
    min_col, max_col = min(cols)*refector, max(cols)*refector,
    cv2.rectangle(image, (min_col, min_row), (max_col, max_row), (0, 0, 255), 1)
    return image


def region_att_mean_score(image, mask):
    # 使用numpy的where函数找到掩码的非零元素的位置
    rows, cols = np.where(mask)
    # 计算掩码的边界框，乘缩放倍数
    min_row, max_row = min(rows), max(rows),
    min_col, max_col = min(cols), max(cols),
    att_mean_score = image[min_row:max_row, min_col:max_col].mean()
    return att_mean_score
    
def compute_centroid(image, mask):
    feat_height, feat_width = image.shape
    fg_img = image.clone()
    fg_img[mask != 1] = 0
    center_y = (fg_img.sum(1) * torch.linspace(0,feat_height-1,feat_height)).sum() / fg_img.sum()
    center_x = (fg_img.sum(0) * torch.linspace(0,feat_width-1,feat_width)).sum() / fg_img.sum()
    return [center_x, center_y]

    
def view_images(
    images: Union[np.ndarray, List],
    num_rows: int = 1,
    offset_ratio: float = 0.02,
    display_image: bool = True,
) -> Image.Image:
    """Displays a list of images in a grid."""
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = (
        np.ones(
            (
                h * num_rows + offset * (num_rows - 1),
                w * num_cols + offset * (num_cols - 1),
                3,
            ),
            dtype=np.uint8,
        )
        * 255
    )
    for i in range(num_rows):
        for j in range(num_cols):
            image_[
                i * (h + offset) : i * (h + offset) + h :,
                j * (w + offset) : j * (w + offset) + w,
            ] = images[i * num_cols + j]

    pil_img = Image.fromarray(image_)

    return pil_img
        
class Visualizer:
    def __init__(self, pipeline, lora_pipelines, controller:AttentionStore, draw_region=True):
        self.pipeline = pipeline
        self.lora_pipelines = lora_pipelines
        self.controller = controller
        self.controller.attention_store = {}
        self.draw_region = draw_region

    def get_average_attention(self):
        average_attention = {
            key: [
                item / self.controller.cur_step
                for item in self.controller.attention_store[key]
            ]
            for key in self.controller.attention_store
        }
        return average_attention

    # def aggregate_attention(
    #     self, res: int, from_where: List[str], only_cross: bool, select: int
    # ):
    #     out_self = []
    #     out_cross = []
    #     # attention_maps = self.get_average_attention()
    #     attention_maps = self.controller.attention_store
    #     # num_pixels = res**2
    #     num_pixels = res * res * 2
    #     for location in from_where:
    #         if not only_cross:
    #             for item in attention_maps[f"{location}_self"]:
    #                 if item.shape[1] == num_pixels:
    #                     self_maps = item.view(-1, res, res*2, res, res*2)
    #                     out_self.append(self_maps)
    #         if len(self.controller.region_prompt) == 0:        
    #             for item in attention_maps[f"{location}_cross"]:
    #                 if item.shape[1] == num_pixels:
    #                     cross_maps = item.reshape(
    #                         1, -1, res, res * 2, item.shape[-1]
    #                     )[select]
    #                     out_cross.append(cross_maps)
    #         else:
    #             for item in attention_maps[f"{location}_cross_region"]:
    #                 if item.shape[1] == num_pixels:
    #                     cross_maps = item.reshape(
    #                         1, -1, res, res * 2, item.shape[-1]
    #                     )[select]
    #                     out_cross.append(cross_maps)
        
    #     if not only_cross:
    #         out_self = torch.cat(out_self, dim=0)  # res*res*2, res*res*2 的那一层self-attn
    #         out_self = out_self.sum(0) / out_self.shape[0]
        
    #     if len(self.controller.region_prompt) == 0:     
    #         out_cross = torch.cat(out_cross, dim=0)  # res * res*2 的那一层特征图
    #         out_cross = out_cross.sum(0) / out_cross.shape[0]
    #         return out_cross, out_self
    #     else:
    #         items_out_cross = []
    #         for item_idx in range(len(self.controller.region_prompt)):
    #             item_atten_map = torch.zeros_like(cross_maps)
    #             for layer_atten_idx in range(item_idx, len(out_cross), len(self.controller.region_prompt)):  # region lora的数量与prompt的数量相同
    #                 item_atten_map += out_cross[layer_atten_idx]
    #             items_out_cross.append(item_atten_map.mean(0))
    #         return items_out_cross, out_self

    
    def aggregate_attention(
        self, from_where: List[str], only_cross: bool, select: int
    ):
        out_cross_region = []
        out_self = []
        out_cross = []
        # attention_maps = self.get_average_attention()
        attention_maps = self.controller.attention_store
        # num_pixels = res**2
        res = self.resolution
        num_pixels = res[0] * res[1]
        for location in from_where:
            if not only_cross:
                for item in attention_maps[f"{location}_self"]:
                    if item.shape[1] == num_pixels:
                        self_maps = item.view(-1, res[0], res[1], res[0], res[1])
                        out_self.append(self_maps)

            for item in attention_maps[f"{location}_cross"]:  # bg cross_attention map
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(
                        1, -1, res[0], res[1], item.shape[-1]
                    )[select]
                    out_cross.append(cross_maps)
            if len(self.controller.region_prompt) != 0:  # region lora cross_attention map
                for item in attention_maps[f"{location}_cross_region"]:
                    if item.shape[1] == num_pixels:
                        cross_maps = item.reshape(
                            1, -1, res[0], res[1], item.shape[-1]
                        )[select]
                        out_cross_region.append(cross_maps)
        
        if not only_cross:
            out_self = torch.cat(out_self, dim=0)  # res*res*2, res*res*2 的那一层self-attn
            out_self = out_self.sum(0) / out_self.shape[0]
    
        out_cross_bg = torch.cat(out_cross, dim=0)  # res * res*2 的那一层特征图
        out_cross_bg = out_cross_bg.sum(0) / out_cross_bg.shape[0]
        
        if len(self.controller.region_prompt) != 0:
            items_out_cross = []
            for item_idx in range(len(self.controller.region_prompt)):
                item_atten_map = torch.zeros_like(cross_maps)
                for layer_atten_idx in range(item_idx, len(out_cross_region), len(self.controller.region_prompt)):  # region lora的数量与prompt的数量相同
                    item_atten_map += out_cross_region[layer_atten_idx]
                items_out_cross.append(item_atten_map.mean(0) / (len(out_cross_region) / len(self.controller.region_prompt)))
        return items_out_cross, out_self, out_cross_bg


    
    # def aggregate_attention_step(  # 用于画每一步的atten-map
    #     self, res: int, from_where: List[str], only_cross: bool, select: int, attention_store_step):
    #     out_cross = []
    #     out_self = []
    #     # attention_maps = self.get_average_attention()
    #     attention_maps = attention_store_step
    #     # num_pixels = res**2
    #     num_pixels = res * res * 2
    #     for location in from_where:
    #         if not only_cross:
    #             for item in attention_maps[f"{location}_self"]:
    #                 if item.shape[1] == num_pixels:
    #                     self_maps = item.view(-1, res, res*2, res, res*2)
    #                     out_self.append(self_maps)
    #         if len(self.controller.region_prompt) == 0:        
    #             for item in attention_maps[f"{location}_cross"]:
    #                 if item.shape[1] == num_pixels:
    #                     cross_maps = item.reshape(
    #                         1, -1, res, res * 2, item.shape[-1]
    #                     )[select]
    #                     out_cross.append(cross_maps)
    #         else:
    #             for item in attention_maps[f"{location}_cross_region"]:
    #                 if item.shape[1] == num_pixels:
    #                     cross_maps = item.reshape(
    #                         1, -1, res, res * 2, item.shape[-1]
    #                     )[select]
    #                     out_cross.append(cross_maps)
        
    #     if not only_cross:
    #         out_self = torch.cat(out_self, dim=0)  # res*res*2, res*res*2 的那一层self-attn
    #         out_self = out_self.sum(0) / out_self.shape[0]
        
    #     if len(self.controller.region_prompt) == 0:     
    #         out_cross = torch.cat(out_cross, dim=0)  # res * res*2 的那一层特征图
    #         out_cross = out_cross.sum(0) / out_cross.shape[0]
    #         return out_cross, out_self
    #     else:
    #         items_out_cross = []
    #         for item_idx in range(len(self.controller.region_prompt)):
    #             item_atten_map = torch.zeros_like(cross_maps)
    #             for layer_atten_idx in range(item_idx, len(out_cross), len(self.controller.region_prompt)):  # region lora的数量与prompt的数量相同
    #                 item_atten_map += out_cross[layer_atten_idx]
    #             items_out_cross.append(item_atten_map.mean(0))
    #         return items_out_cross, out_self
        
    def aggregate_attention_step(  # 用于画每一步的atten-map
        self, from_where: List[str], only_cross: bool, select: int, attention_store_step):
        out_cross_region = []
        out_cross = []
        out_self = []
        # attention_maps = self.get_average_attention()
        attention_maps = attention_store_step
        # num_pixels = res**2
        res = self.controller.resolution
        num_pixels = res[0] * res[1]
        for location in from_where:
            if not only_cross:
                for item in attention_maps[f"{location}_self"]:
                    if item.shape[1] == num_pixels:
                        self_maps = item.view(-1, res[0], res[1], res[0], res[1])
                        out_self.append(self_maps)

            for item in attention_maps[f"{location}_cross"]:  # bg cross_attention map
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(
                        1, -1, res[0], res[1], item.shape[-1]
                    )[select]
                    out_cross.append(cross_maps)
            if len(self.controller.region_prompt) != 0:  # region lora cross_attention map
                for item in attention_maps[f"{location}_cross_region"]:
                    if item.shape[1] == num_pixels:
                        cross_maps = item.reshape(
                            1, -1, res[0], res[1], item.shape[-1]
                        )[select]
                        out_cross_region.append(cross_maps)
        
        if not only_cross:
            out_self = torch.cat(out_self, dim=0)  # res*res*2, res*res*2 的那一层self-attn
            out_self = out_self.sum(0) / out_self.shape[0]
    
        out_cross_bg = torch.cat(out_cross, dim=0)  # res * res*2 的那一层特征图
        out_cross_bg = out_cross_bg.sum(0) / out_cross_bg.shape[0]
        
        if len(self.controller.region_prompt) != 0:
            items_out_cross = []
            for item_idx in range(len(self.controller.region_prompt)):
                item_atten_map = torch.zeros_like(cross_maps)
                for layer_atten_idx in range(item_idx, len(out_cross_region), len(self.controller.region_prompt)):  # region lora的数量与prompt的数量相同
                    item_atten_map += out_cross_region[layer_atten_idx]
                items_out_cross.append(item_atten_map.mean(0) / (len(out_cross_region) / len(self.controller.region_prompt)))
        return items_out_cross, out_self, out_cross_bg

    
    def save_cross_attention_vis(self, prompt, attention_maps, bg_attention_maps, path):
        bg_attention_maps = bg_attention_maps.detach().cpu()
        # tokens = self.pipeline.tokenizer.encode(prompt)
        tokens = self.pipeline.tokenizer(prompt).input_ids[0]
        images = []
        for i in range(len(tokens)):
            image = bg_attention_maps[:, :, i]
            reg_param = image.max()
            # reg_param = torch.ones_like(image.max())
            image = 255 * image / reg_param
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            if self.controller.resolution == [16, 32]:
                image = np.array(Image.fromarray(image).resize((512, 256)))
            else:
                image = np.array(Image.fromarray(image).resize((512, 384)))
            reg_param_txt = str(reg_param).split(',')[0].split('(')[1]
            image = text_under_image(image, self.pipeline.tokenizer.decode(int(tokens[i])))
            image = text_under_image(image, "reg=" + reg_param_txt)
            images.append(image)
        vis = view_images(np.stack(images, axis=0))
        vis.save(os.path.join(path,"cross_attn.png"))
        
        # region_cross_attn
        if len(self.controller.region_prompt) != 0:
            for k, prompt in enumerate(self.controller.region_prompt):
                images = []
                tokens = self.lora_pipelines[k].tokenizer(prompt).input_ids
                for i in range(len(tokens)):
                    image = attention_maps[k][:, :, i].detach().cpu()
                    centroid = compute_centroid(image, self.controller.attn_mask_store[k])
                    if self.controller.resolution == [16, 32]:
                        refector = 256 // image.shape[0]
                    else:
                        refector = 384 // image.shape[0]
                    reg_param = image.max()
                    # reg_param = torch.ones_like(image.max())
                    region_mean = region_att_mean_score(image, self.controller.attn_mask_store[k])
                    image = image.unsqueeze(-1).expand(*image.shape, 3)
                    image = 255 * image / reg_param
                    image = image.numpy().astype(np.uint8)
                    if self.controller.resolution == [16, 32]:
                        image = np.array(Image.fromarray(image).resize((512, 256)))
                    else:
                        image = np.array(Image.fromarray(image).resize((512, 384)))
                    if self.draw_region:
                        image = draw_rectangle(image, self.controller.attn_mask_store[k], refector)
                        image = cv2.circle(image, (int(centroid[0]*refector), int(centroid[1]*refector)), 2, (0, 0, 255), -1)
                        region_mean_txt = str(region_mean).split(',')[0].split('(')[1]
                        image = text_under_image(image, "reg=" + reg_param_txt + "mean=" + region_mean_txt)
                    reg_param_txt = str(reg_param).split(',')[0].split('(')[1]
                    image = text_under_image(image, self.lora_pipelines[k].tokenizer.decode(int(tokens[i])))
                    images.append(image)
                    
                vis = view_images(np.stack(images, axis=0))
                vis.save(os.path.join(path,f"item{k}_cross_attn.png"))
        # else:
        #     attention_maps = attention_maps.detach().cpu()
        #     # tokens = self.pipeline.tokenizer.encode(prompt)
        #     tokens = self.pipeline.tokenizer(prompt).input_ids[0]
        #     images = []
        #     for i in range(len(tokens)):
        #         image = attention_maps[:, :, i]
        #         # reg_param = image.max()
        #         reg_param = torch.ones_like(image.max())
        #         image = 255 * image / reg_param
        #         image = image.unsqueeze(-1).expand(*image.shape, 3)
        #         image = image.numpy().astype(np.uint8)
        #         image = np.array(Image.fromarray(image).resize((512, 256)))
        #         image = text_under_image(
        #             image, self.pipeline.tokenizer.decode(int(tokens[i])) + str(reg_param).split(',')[0].split('(')[1]
        #         )
        #         images.append(image)
        #     vis = view_images(np.stack(images, axis=0))
        #     vis.save(os.path.join(path,"cross_attn.png"))
        
        
    def save_self_attention_vis(self, attention_maps, path):
        images = []
        background_mask = torch.zeros_like(self.controller.attn_mask_store[0])
        for i, mask in enumerate(self.controller.attn_mask_store):
            # attention_maps(16,32,16,32), mask(16,32),获取前景query对其他所有点的平均注意力值
            image = attention_maps[mask != 0].mean(0)
            if self.controller.resolution == [16, 32]:
                refector = 256 // image.shape[0]
            else:
                refector = 384 // image.shape[0]
            reg_param = image.max()
            region_mean = region_att_mean_score(image, mask)
            image = 255 * image / reg_param
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            if self.controller.resolution == [16, 32]:
                image = np.array(Image.fromarray(image).resize((512, 256)))
            else:
                image = np.array(Image.fromarray(image).resize((512, 384)))
            if self.draw_region:
                image = draw_rectangle(image, mask, refector)
                region_mean_txt = str(region_mean).split(',')[0].split('(')[1]
                image = text_under_image(image, "reg=" + reg_param_txt + "mean=" + region_mean_txt)
                
            reg_param_txt = str(reg_param).split(',')[0].split('(')[1]
            image = text_under_image(image, str(i)+" area")
            images.append(image)
            # 计算背景mask
            background_mask += mask
    
        image_bg = attention_maps[background_mask == 0].mean(0)
        reg_param = image_bg.max()
        image_bg = 255 * image_bg / reg_param
        image_bg = image_bg.unsqueeze(-1).expand(*image_bg.shape, 3)
        image_bg = image_bg.numpy().astype(np.uint8)
        if self.controller.resolution == [16, 32]:
            image_bg = np.array(Image.fromarray(image_bg).resize((512, 256)))
        else:
            image_bg = np.array(Image.fromarray(image_bg).resize((512, 384)))
        for mask in self.controller.attn_mask_store:
            if self.draw_region:
                image_bg = draw_rectangle(image_bg, mask, refector)
        reg_param_txt = str(reg_param).split(',')[0].split('(')[1]
        image_bg = text_under_image(image_bg, "bg_area")
        if self.draw_region:
            image_bg = text_under_image(image_bg, "reg=" + reg_param_txt)
        images.append(image_bg)
        # 存储前景self-atten map
        vis = view_images(np.stack(images, axis=0))
        vis.save(os.path.join(path,f"self_attn.png"))
    
    def visualize(self, prompt, path, only_cross, save_every_step=True):
        if save_every_step:
            if os.path.exists(path):  # 清理文件
                shutil.rmtree(path)
            for k, item in enumerate(self.controller.all_step_store.values()):
                cross_agg_attn, self_agg_attn, bg_cross_agg_attn = self.aggregate_attention_step(from_where=("up", "down", "mid"), only_cross=only_cross, select=0, attention_store_step=item)
                step_save_path = os.path.join(path, str(k))
                if not os.path.exists(step_save_path):
                    os.makedirs(step_save_path)
                self.save_cross_attention_vis(
                    prompt,
                    attention_maps=cross_agg_attn,
                    bg_attention_maps=bg_cross_agg_attn,
                    path=step_save_path,
                )
                if not only_cross:
                    self.save_self_attention_vis(
                        attention_maps=self_agg_attn.detach().cpu(),
                        path=step_save_path,
                )
        
        else:
            cross_agg_attn, self_agg_attn, bg_cross_agg_attn = self.aggregate_attention(  # 16是倒数第二层特征图, 当不可视化region_lora时cross_agg_attn=[]
                    from_where=("up", "down", "mid"), only_cross=only_cross, select=0
                )
            self.save_cross_attention_vis(
                    prompt,
                    attention_maps=cross_agg_attn,
                    bg_attention_maps=bg_cross_agg_attn,
                    path=step_save_path,
                )
            if not only_cross:
                self.save_self_attention_vis(
                    attention_maps=self_agg_attn.detach().cpu(),
                    path=path,
            )
                
            
        