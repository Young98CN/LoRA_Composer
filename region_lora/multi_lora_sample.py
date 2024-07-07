import math
import numpy as np
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from einops import rearrange
from PIL import Image
from PIL import Image as PIL_Image
from torch import einsum
from typing import Any, Callable, Dict, List, Optional, Union

from mixofshow.utils.diffusers_sample_util import bind_concept_prompt
from diffusers import StableDiffusionAdapterPipeline
import re, os
import copy
from region_lora.utils.gaussian_smoothing import GaussianSmoothing,AverageSmoothing
from torch.nn import functional as F
from region_lora.utils.attn_util import aggregate_attention, AttentionStore
from torch.nn.functional import cosine_similarity


def generate_gaussian_matrix(size, center=[-0.0,0.0], sigma_x=1.0, sigma_y=1.0):
    x, y = torch.meshgrid(torch.linspace(-1, 1, size[0]), torch.linspace(-1, 1, size[1]))
    
    dx = x - center[0]
    dy = y - center[1]
    
    gaussian = torch.exp(-((dx**2 / (2.0 * sigma_x**2)) + (dy**2 / (2.0 * sigma_y**2))))

    gaussian = gaussian / torch.max(gaussian)

    return gaussian

def _compute_max_attention_per_index(attention_maps: List[torch.Tensor],
                                    indices_to_alter: List[List],
                                    smooth_attentions: bool = False,
                                    sigma: float = 0.5,
                                    kernel_size: int = 3,
                                    bbox: List[int] = None,
                                    top_k_ratio=0.2,
                                    self_att_map: torch.Tensor = None,
                                    self_att_mask: List[torch.Tensor] = None,
                                    fill_loss=True,
                                    gassion_sample=True,
                                    bg_loss=True,
                                    ) -> List[torch.Tensor]:
        """ Computes the maximum attention value for each of the tokens we wish to alter. """

        max_indices_list_fg = []
        max_indices_list_bg = []
        dist_x = []
        dist_y = []
        local_box_mean = []
        # Concept Injection Constraints
        for cnt, (item_attention_map, item_indices_to_alter) in enumerate(zip(attention_maps, indices_to_alter)):
            attention_for_text = item_attention_map[:, :, 1:-1]
            for i in item_indices_to_alter:
                image = attention_for_text[:, :, i]
                feat_height = image.shape[0]
                feat_width = image.shape[1]
                start_h, start_w, end_h, end_w = bbox[cnt][-1]
                x1, y1, x2, y2 = math.ceil(start_h * feat_height), math.ceil(
                        start_w * feat_width), math.floor(end_h * feat_height), math.floor(end_w * feat_width)
                # coordinates to masks
                obj_mask = torch.zeros_like(image).cuda()
                ones_mask = torch.ones([x2 - x1, y2 - y1], dtype=obj_mask.dtype).to(obj_mask.device)
                obj_mask[x1:x2, y1:y2] = ones_mask
                
                if smooth_attentions:
                    smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                    padding_size = (kernel_size - 1) // 2
                    input = F.pad(image.unsqueeze(0).unsqueeze(0), (padding_size, padding_size, padding_size, padding_size), mode='reflect').cuda()
                    image = smoothing(input).squeeze(0).squeeze(0)

                k = (obj_mask.sum() * top_k_ratio).long()
                gassian = generate_gaussian_matrix([x2 - x1, y2 - y1], center=[-0.0,0.0], sigma_x=1.0, sigma_y=1.0).cuda()
                obj_mask_gassian = obj_mask.clone()
                obj_mask_gassian[x1:x2, y1:y2] *= gassian
                
                # gassion_sample
                if gassion_sample:
                    # print("gassion_sample")
                    fg_img = image * obj_mask_gassian
                else:
                    fg_img = image * obj_mask
                local_box_mean.append(fg_img[x1:x2, y1:y2].detach().mean().cpu())
                # Concept Enhancement Constraints
                fg_loss = fg_img.reshape(-1).topk(k)[0].mean()
                max_indices_list_fg.append(fg_loss)

                # fill_loss
                if fill_loss:
                    # print("fill_loss")
                    gt_proj_x = torch.max(obj_mask, dim=0)[0]
                    gt_proj_y = torch.max(obj_mask, dim=1)[0]
                    dist_x.append(2*F.l1_loss((image*obj_mask).max(dim=0)[0], gt_proj_x))
                    dist_y.append(2*F.l1_loss((image*obj_mask).max(dim=1)[0], gt_proj_y))
                
            # Region Perceptual Restriction
            obj_self_att_mask = self_att_mask[cnt]  # self-atten map [16,32]
            
            fg_self_att = self_att_map[obj_self_att_mask != 0]  # forground query attention map [N,16,32]
            fg_bg_self_att = fg_self_att[:, obj_self_att_mask == 0].reshape(-1).topk(k)[0].mean()  # [N,N']
            if bg_loss:
                # print("bg_loss")
                max_indices_list_bg.append(10*(fg_bg_self_att))
            
        return max_indices_list_fg, max_indices_list_bg, dist_x, dist_y, local_box_mean

def _aggregate_and_get_max_attention_per_token( attention_store: AttentionStore,
                                                indices_to_alter: List[List],
                                                smooth_attentions: bool = False,
                                                sigma: float = 0.5,
                                                kernel_size: int = 3,
                                                bbox: List[int] = None,
                                                top_k_ratio = 0.2,
                                                fill_loss=True,
                                                gassion_sample=True,
                                                bg_loss=True,
                                                ):
    """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
    item_cross_att_maps, self_att_map, bg_cross_att_maps = aggregate_attention(
        controller=attention_store,
        from_where=("up", "down", "mid"),
        only_cross=False,
        select=0)
    max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y, local_box_mean = _compute_max_attention_per_index(  # 计算前景和背景像素的idx
        attention_maps=item_cross_att_maps,
        indices_to_alter=indices_to_alter,
        smooth_attentions=smooth_attentions,
        sigma=sigma,
        kernel_size=kernel_size,
        bbox=bbox,
        top_k_ratio=top_k_ratio,
        self_att_map=self_att_map,
        self_att_mask=attention_store.attn_mask_store,
        fill_loss=fill_loss,
        gassion_sample=gassion_sample,
        bg_loss=bg_loss,
    )
    return max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y, local_box_mean

def _compute_loss(max_attention_per_index_fg: List[torch.Tensor], max_attention_per_index_bg: List[torch.Tensor],
                    dist_x: List[torch.Tensor], dist_y: List[torch.Tensor], return_losses: bool = False, local_box_mean=None) -> torch.Tensor:
    """ Computes the attend-and-excite loss using the maximum attention value for each token. """
    factors = [8] * len(local_box_mean)
    losses_fg = [max(0, 1. - curr_max) for curr_max in max_attention_per_index_fg]
    losses_fg = [curr_loss * factor for curr_loss, factor in zip(losses_fg,factors)]
    losses_bg = [max(0, curr_max) for curr_max in max_attention_per_index_bg]
    loss = sum(losses_fg) + sum(losses_bg) + sum(dist_x) + sum(dist_y)
    if return_losses:
        return max(losses_fg), losses_fg
    else:
        return max(losses_fg), loss

def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
    """ Update the latent according to the computed loss. """
    torch.autograd.set_detect_anomaly(True) 
    grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
    latents = latents - step_size * grad_cond
    return latents
    
def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL_Image.Image):
        image = [image]

    if isinstance(image[0], PIL_Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=Image.LANCZOS))[None, :] for i in image]
        image = np.concatenate(image, axis=0)

        if len(image.shape) == 3:
            image = image[..., None]

        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


def replace_latents(attention_maps, region_list, n, lora_token_idx_list, latents):
    down_sample_scale = latents.shape[-1] // attention_maps[0].shape[1]
    latents_clone = latents.detach().clone()
    for cnt, (item_attention_map, item_indices_to_alter) in enumerate(zip(attention_maps, lora_token_idx_list)):
        item_attention_map = item_attention_map.detach().clone()
        item_attention_map = item_attention_map[:, :, 1:-1]
        
        for i, token_idx in enumerate(item_indices_to_alter):
            lora_atte_score = item_attention_map[:, :, token_idx]
            feat_height = lora_atte_score.shape[0]
            feat_width = lora_atte_score.shape[1]
            
            start_h, start_w, end_h, end_w = region_list[cnt][-1]
            start_h, start_w, end_h, end_w = math.ceil(start_h * feat_height), math.ceil(
                    start_w * feat_width), math.floor(end_h * feat_height), math.floor(end_w * feat_width)
            
            kernel_size = (end_h-start_h, end_w-start_w)
            kernel = torch.ones(1, 1, *kernel_size, dtype=lora_atte_score.dtype, device=lora_atte_score.device)
            
            if len(lora_atte_score.shape) == 2:
                lora_atte_score = lora_atte_score.unsqueeze(0).unsqueeze(0)
            if i == 0:
                scores = F.conv2d(lora_atte_score, kernel, stride=1)
            else:
                scores += F.conv2d(lora_atte_score, kernel, stride=1)

        topk_indices = scores.flatten().topk(n).indices

        # compute box
        for k, idx in enumerate(topk_indices):
            top_x = int(idx / scores.shape[3])
            top_y = int(idx % scores.shape[3])
            bottom_x = top_x + kernel_size[0]
            bottom_y = top_y + kernel_size[1]
            
            if k == 0:
                res = latents[:, :, top_x*down_sample_scale:bottom_x*down_sample_scale, top_y*down_sample_scale:bottom_y*down_sample_scale].clone()
            else:
                res += latents[:, :, top_x*down_sample_scale:bottom_x*down_sample_scale, top_y*down_sample_scale:bottom_y*down_sample_scale].clone()
        
        latents_clone[:, :, start_h*down_sample_scale:end_h*down_sample_scale, start_w*down_sample_scale:end_w*down_sample_scale] = res 
            
    latents_clone = ((latents_clone - latents_clone.mean()) / latents_clone.std()) * latents.std() + latents.mean()  # norm
    return latents_clone


def register_region_aware_attention(model, layerwise_embedding=False):

    def get_new_forward(cross_attention_idx):

        def region_rewrite(self, hidden_states, query, region_list, height, width, attention_store, unet_type):

            def get_region_mask(region_list, feat_height, feat_width):
                exclusive_mask = torch.zeros((feat_height, feat_width))
                for region in region_list:
                    start_h, start_w, end_h, end_w = region[2]
                    start_h, start_w, end_h, end_w = math.ceil(start_h * feat_height), math.ceil(
                        start_w * feat_width), math.floor(end_h * feat_height), math.floor(end_w * feat_width)
                    exclusive_mask[start_h:end_h, start_w:end_w] += 1
                return exclusive_mask

            dtype = query.dtype
            seq_lens = query.shape[1]
            downscale = math.sqrt(height * width / seq_lens)

            feat_height, feat_width = int(height // downscale), int(width // downscale)
            region_mask = get_region_mask(region_list, feat_height, feat_width)

            query = rearrange(query, 'b (h w) c -> b h w c', h=feat_height, w=feat_width)
            hidden_states = rearrange(hidden_states, 'b (h w) c -> b h w c', h=feat_height, w=feat_width)

            new_hidden_state = torch.zeros_like(hidden_states)
            new_hidden_state[:, region_mask == 0, :] = hidden_states[:, region_mask == 0, :]
            replace_ratio = 1.0
            new_hidden_state[:, region_mask != 0, :] = (1 - replace_ratio) * hidden_states[:, region_mask != 0, :]
            
            # Region-Aware LoRA Injection
            for idx, region in enumerate(region_list):
                region_key, region_value, region_box, decoder = region
                if self.upcast_attention:
                    query = query.float()
                    region_key = region_key.float()

                start_h, start_w, end_h, end_w = region_box
                start_h, start_w, end_h, end_w = math.ceil(start_h * feat_height), math.ceil(
                    start_w * feat_width), math.floor(end_h * feat_height), math.floor(end_w * feat_width)


                attention_region = einsum('b h w c, b n c -> b h w n', query[:, start_h:end_h, start_w:end_w, :],
                                        region_key) * self.scale
                
                
                if self.upcast_softmax:
                    attention_region = attention_region.float()

                attention_region = attention_region.softmax(dim=-1)
                attention_region = attention_region.to(dtype)
                
                # save attention score
                if attention_store is not None:
                    if idx + 1 != len(region_list):
                        attention_store.cur_att_layer -= 1  # Reset counter because there are multiple region_lora
                    attn_map = einsum('b h w c, b n c -> b h w n', query, region_key) * self.scale
                    attn_map = attn_map.softmax(dim=-1)
                    attn_map = attn_map.to(dtype)
                        
                    attn_map = rearrange(attn_map, 'b h w c -> b (h w) c')
                    attention_store(attn_map, True, unet_type, is_region=True)

                hidden_state_region = einsum('b h w n, b n c -> b h w c', attention_region, region_value)
                
                hidden_state_region = rearrange(hidden_state_region, 'b h w c -> b (h w) c')
                hidden_state_region = decoder.batch_to_head_dim(hidden_state_region)
                hidden_state_region = decoder.to_out[0](hidden_state_region)
                hidden_state_region = decoder.to_out[1](hidden_state_region)
                hidden_state_region = rearrange(hidden_state_region, 'b (h w) c -> b h w c', h=end_h - start_h, w=end_w - start_w)
                
                new_hidden_state[:, start_h:end_h, start_w:end_w, :] += \
                    replace_ratio * (hidden_state_region / (
                        region_mask.reshape(
                            1, *region_mask.shape, 1)[:, start_h:end_h, start_w:end_w, :]
                    ).to(query.device))

            new_hidden_state = rearrange(new_hidden_state, 'b h w c -> b (h w) c')
            return new_hidden_state

        def new_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            batch_size, sequence_length, _ = hidden_states.shape
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                is_cross = False
                encoder_hidden_states = hidden_states
            else:
                is_cross = True
                if layerwise_embedding:
                    encoder_hidden_states = encoder_hidden_states[:, cross_attention_idx, ...]
                else:
                    encoder_hidden_states = encoder_hidden_states

            if self.cross_attention_norm:
                encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            
            region_list = []
            pipe_lora = cross_attention_kwargs['pipe_lora_list']
            # Region-Aware LoRA Injection (prepare K and V)
            for region, lora_idx in zip(cross_attention_kwargs['region_list'], cross_attention_kwargs['lora_pipe_idx']):
                if cross_attention_idx < 6:
                    cross_attention_lora = pipe_lora[lora_idx].unet.down_blocks[cross_attention_idx // 2].attentions[cross_attention_idx % 2].transformer_blocks[0].attn2
                    
                elif cross_attention_idx > 6:
                    up_id = cross_attention_idx - 7
                    cross_attention_lora = pipe_lora[lora_idx].unet.up_blocks[up_id // 3 + 1].attentions[up_id % 3].transformer_blocks[0].attn2
                else:
                    cross_attention_lora = pipe_lora[lora_idx].unet.mid_block.attentions[0].transformer_blocks[0].attn2
                if layerwise_embedding: 
                    region_key = cross_attention_lora.to_k(region[0][:, cross_attention_idx, ...])
                    region_value = cross_attention_lora.to_v(region[0][:, cross_attention_idx, ...])
                else:
                    region_key = cross_attention_lora.to_k(region[0])
                    region_value = cross_attention_lora.to_v(region[0])
                region_key = cross_attention_lora.head_to_batch_dim(region_key)
                region_value = cross_attention_lora.head_to_batch_dim(region_value)
                region_list.append((region_key, region_value, region[1], cross_attention_lora))
            
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)

            if cross_attention_idx < 6:
                unet_type = 'down'
            elif cross_attention_idx > 6:
                unet_type = 'up'
            else:
                unet_type = 'mid'

            if not is_cross:
                seq_lens = query.shape[1]
                height=cross_attention_kwargs['height']
                width=cross_attention_kwargs['width']
                downscale = math.sqrt(height * width / seq_lens)

                # 0: prepare mask
                feat_height, feat_width = int(height // downscale), int(width // downscale)
                attn_mask_list = []
                attn_mask_list_vis = []
                background_mask = torch.ones((feat_height, feat_width))
                for region in region_list:
                    attn_mask = torch.zeros((feat_height, feat_width))
                    start_h, start_w, end_h, end_w = region[2]
                    start_h, start_w, end_h, end_w = math.ceil(start_h * feat_height), math.ceil(
                        start_w * feat_width), math.floor(end_h * feat_height), math.floor(end_w * feat_width)
                    attn_mask[start_h:end_h, start_w:end_w] += 1
                    attn_mask_list.append(rearrange(attn_mask, 'h w -> (h w)'))
                    attn_mask_list_vis.append(attn_mask)
                    background_mask -= attn_mask
                if cross_attention_kwargs['attention_store'] is not None:
                    cross_attention_kwargs['attention_store'].save_self_mask(seq_lens, attn_mask_list_vis)  # for visualization

                # Concept Region Mask
                if cross_attention_kwargs['self_attn_mask']:
                    attention_mask = torch.zeros(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device)
                    # self-attn mask_other    
                    for k, mask in enumerate(attn_mask_list):
                        other_attn_mask = copy.deepcopy(attn_mask_list)
                        other_attn_mask.pop(k)
                        for mask_mat in other_attn_mask:
                            other_role_mask = mask_mat.type(torch.bool).to(query.device)
                            other_role_decay = other_role_mask * -1e6
                            attention_mask[:, mask != 0, :] += other_role_decay
                    attention_scores = torch.baddbmm(attention_mask, query, key.transpose(-1, -2),beta=1,alpha=self.scale)
                    attention_probs = attention_scores.softmax(dim=-1)
                    attention_probs = attention_probs.to(query.dtype)
            
            # save attention map
            if cross_attention_kwargs['attention_store'] is not None:
                attention_store = cross_attention_kwargs['attention_store']
                if cross_attention_kwargs['vis_region_rewrite_decoder']:
                    if not is_cross:  # save self attention
                        attention_store(attention_probs, is_cross, unet_type)
                    else:
                        attention_store.cur_att_layer -= 1  # Reset counter
                        attention_store(attention_probs, is_cross, unet_type)  # cross attention map (before rewrite)
                else:  # save self attention and cross attention
                    attention_store(attention_probs, is_cross, unet_type)
            
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)  # reshape

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)
            
            if is_cross:    
                if cross_attention_kwargs['step']<50:
                    if not cross_attention_kwargs['vis_region_rewrite_decoder']:
                        attention_store = None
         
                    hidden_states = region_rewrite(
                        self,
                        hidden_states=hidden_states,
                        query=query,
                        region_list=region_list,
                        height=cross_attention_kwargs['height'],
                        width=cross_attention_kwargs['width'],
                        attention_store=attention_store,
                        unet_type=unet_type,
                        )

            return hidden_states

        return new_forward

    def change_forward(unet, cross_attention_idx):  # omit proceesor in new diffusers
        for name, layer in unet.named_children():
            if layer.__class__.__name__ == 'CrossAttention':
                bound_method = get_new_forward(cross_attention_idx).__get__(layer, layer.__class__)
                setattr(layer, 'forward', bound_method)
                if name == 'attn2':
                    cross_attention_idx += 1
            else:
                cross_attention_idx = change_forward(layer, cross_attention_idx)
        return cross_attention_idx

    # use this to ensure the order
    cross_attention_idx = change_forward(model.unet.down_blocks, 0)
    cross_attention_idx = change_forward(model.unet.mid_block, cross_attention_idx)
    _ = change_forward(model.unet.up_blocks, cross_attention_idx)


def encode_region_prompt_pplus(self,
                               prompt,
                               new_concept_cfg_list,
                               bg_new_concept_cfg,
                               device,
                               num_images_per_prompt,
                               do_classifier_free_guidance,
                               negative_prompt=None,
                               prompt_embeds: Optional[torch.FloatTensor] = None,
                               negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                               height=512,
                               width=512,
                               pipeline_lora_list: List[StableDiffusionAdapterPipeline] = None,
                               attention_store = None):
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    assert batch_size == 1, 'only sample one prompt once in this version'

    if prompt_embeds is None:
        region_list_positive = [] 
        context_prompt, region_list = prompt[0][0], prompt[0][1]

        context_prompt = bind_concept_prompt([context_prompt],  bg_new_concept_cfg)
        context_prompt_input_ids = self.tokenizer(
            context_prompt,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        ).input_ids

        prompt_embeds = self.text_encoder(context_prompt_input_ids.to(device), attention_mask=None)[0]
        prompt_embeds = rearrange(prompt_embeds, '(b n) m c -> b n m c', b=batch_size)
        prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, layer_num, seq_len, _ = prompt_embeds.shape

        if negative_prompt is None:
            negative_prompt = [''] * batch_size

        negative_prompt_input_ids = self.tokenizer(
            negative_prompt,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt').input_ids

        negative_prompt_embeds = self.text_encoder(
            negative_prompt_input_ids.to(device),
            attention_mask=None,
        )[0]

        negative_prompt_embeds = (negative_prompt_embeds).view(batch_size, 1, seq_len, -1).repeat(1, layer_num, 1, 1)
        negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # check the corespondence of prompt and lora pipeline
        new_concept_key = [i.keys() for i in new_concept_cfg_list]
        lora_token_idx = [[]] * len(region_list)
        lora_pipe_idx = []
        for idx, region in enumerate(region_list):
            region_prompt, region_neg_prompt, pos = region
            for concept_idx, key in enumerate(new_concept_key):
                pattern = r'<(.*?)>'
                matche = re.findall(pattern, str(key))[0]
                if matche in region_prompt:
                    tokens = region_prompt.split()
                    token_index = tokens.index('<' + matche + '>')
                    concept_lora_token_idx = [token_index, token_index + 1]
                    break
            lora_token_idx[concept_idx] = concept_lora_token_idx  # save position of lora-token
            lora_pipe_idx.append(concept_idx)  
            region_prompt = bind_concept_prompt([region_prompt], new_concept_cfg_list[concept_idx])
            if attention_store is not None:
                attention_store.save_region_prompt(region_prompt[0])
            region_prompt_input_ids = pipeline_lora_list[concept_idx].tokenizer(
                region_prompt,
                padding='max_length',
                max_length=pipeline_lora_list[concept_idx].tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt').input_ids
            region_embeds = pipeline_lora_list[concept_idx].text_encoder(region_prompt_input_ids.to(device), attention_mask=None)[0]
            region_embeds = rearrange(region_embeds, '(b n) m c -> b n m c', b=batch_size)
            region_embeds.to(dtype=pipeline_lora_list[concept_idx].text_encoder.dtype, device=device)
            bs_embed, layer_num, seq_len, _ = region_embeds.shape

            if region_neg_prompt is None:
                region_neg_prompt = [''] * batch_size
            region_negprompt_input_ids = pipeline_lora_list[concept_idx].tokenizer(
                region_neg_prompt,
                padding='max_length',
                max_length=pipeline_lora_list[concept_idx].tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt').input_ids
            region_neg_embeds = pipeline_lora_list[concept_idx].text_encoder(region_negprompt_input_ids.to(device), attention_mask=None)[0]
            region_neg_embeds = (region_neg_embeds).view(batch_size, 1, seq_len, -1).repeat(1, layer_num, 1, 1)
            region_neg_embeds.to(dtype=pipeline_lora_list[concept_idx].text_encoder.dtype, device=device)
            region_list[idx] = (torch.cat([region_neg_embeds, region_embeds]), pos)
            region_list_positive.append([region_embeds, pos])

    return prompt_embeds, region_list, region_list_positive, lora_token_idx, lora_pipe_idx


@torch.no_grad()
def Regionally_T2IAdaptor_Sample(self,
                                 prompt: Union[str, List[str]] = None,
                                 new_concept_cfg_list=None,
                                 bg_new_concept_cfg=None,
                                 keypose_adapter_input: Union[torch.Tensor, PIL_Image.Image,
                                                              List[PIL_Image.Image]] = None,
                                 keypose_adaptor_weight=1.0,
                                 region_keypose_adaptor_weight='',
                                 sketch_adapter_input: Union[torch.Tensor, PIL_Image.Image,
                                                             List[PIL_Image.Image]] = None,
                                 sketch_adaptor_weight=1.0,
                                 region_sketch_adaptor_weight='',
                                 height: Optional[int] = None,
                                 width: Optional[int] = None,
                                 pipe_lora_list = None,
                                 attention_store = None,
                                 self_attn_mask = True,  # Concept Region Mask
                                 fill_loss=True,  # L_fill
                                 gassion_sample=True,  # gaussion_sample in Concept Enhancement Constraints
                                 bg_loss=True,  # L_region
                                 vis_region_rewrite_decoder = True, #  visualization option: true displays rewrite decoder, false displays base model decoder
                                 num_inference_steps: int = 50,
                                 guidance_scale: float = 7.5,
                                 negative_prompt: Optional[Union[str, List[str]]] = None,
                                 num_images_per_prompt: Optional[int] = 1,
                                 eta: float = 0.0,
                                 generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                                 latents: Optional[torch.FloatTensor] = None,
                                 prompt_embeds: Optional[torch.FloatTensor] = None,
                                 negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                                 output_type: Optional[str] = 'pil',
                                 return_dict: bool = True,
                                 callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                                 callback_steps: int = 1,
                                 cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                                 image_guidance=False # Whether to use image_guidance (canny or pose)
                                 ):
    if image_guidance:
        print("use image_guidance")
    
    if new_concept_cfg_list is None:
        # register region aware attention for sd embedding
        register_region_aware_attention(self, layerwise_embedding=False)
    else:
        # register region aware attention for pplus embedding
        register_region_aware_attention(self, layerwise_embedding=True)

    # 0. Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

    if keypose_adapter_input is not None:
        keypose_input = preprocess(keypose_adapter_input).to(self.device)
        keypose_input = keypose_input.to(self.keypose_adapter.dtype)
    else:
        keypose_input = None

    if sketch_adapter_input is not None:
        sketch_input = preprocess(sketch_adapter_input).to(self.device)
        sketch_input = sketch_input.to(self.sketch_adapter.dtype)
    else:
        sketch_input = None

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    # region_list = (torch.cat([region_neg_embeds, region_embeds]), pos)
    # prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    # prompt = [(prompt, region_collection)]
    prompt_embeds, region_list, region_list_positive, lora_token_idx, lora_pipe_idx = encode_region_prompt_pplus(  # 将prompt输入过text encoder得到embedding
        self,
        prompt,
        new_concept_cfg_list,
        bg_new_concept_cfg,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        height=height,
        width=width,
        pipeline_lora_list=pipe_lora_list,
        attention_store=attention_store,)
    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = self.unet.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
   
    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Denoising loop
    if keypose_input is not None:
        keypose_adapter_state = self.keypose_adapter(keypose_input)
        keys = keypose_adapter_state.keys()
    else:
        keypose_adapter_state = None

    if sketch_input is not None:
        sketch_adapter_state = self.sketch_adapter(sketch_input)
        keys = sketch_adapter_state.keys()
    else:
        sketch_adapter_state = None

    adapter_state = keypose_adapter_state if keypose_adapter_state is not None else sketch_adapter_state
    
    adapter_state_positive = keypose_adapter_state if keypose_adapter_state is not None else sketch_adapter_state

    if do_classifier_free_guidance and image_guidance:
        for k in keys:
            if keypose_adapter_state is not None:
                feat_keypose = keypose_adapter_state[k]
                spatial_adaptor_weight = keypose_adaptor_weight * torch.ones(*feat_keypose.shape[2:]).to(
                    feat_keypose.dtype).to(feat_keypose.device)

                if region_keypose_adaptor_weight != '':
                    keypose_region_list = region_keypose_adaptor_weight.split('|')

                    for region_weight in keypose_region_list:
                        region, weight = region_weight.split('-')
                        region = eval(region)
                        weight = eval(weight)
                        feat_height, feat_width = feat_keypose.shape[2:]
                        start_h, start_w, end_h, end_w = region
                        start_h, end_h = start_h / height, end_h / height
                        start_w, end_w = start_w / width, end_w / width

                        start_h, start_w, end_h, end_w = math.ceil(start_h * feat_height), math.ceil(
                            start_w * feat_width), math.floor(end_h * feat_height), math.floor(end_w * feat_width)

                        spatial_adaptor_weight[start_h:end_h, start_w:end_w] = weight
                feat_keypose = spatial_adaptor_weight * feat_keypose

            else:
                feat_keypose = 0

            if sketch_adapter_state is not None:
                feat_sketch = sketch_adapter_state[k]
                spatial_adaptor_weight = sketch_adaptor_weight * torch.ones(*feat_sketch.shape[2:]).to(
                    feat_sketch.dtype).to(feat_sketch.device)

                if region_sketch_adaptor_weight != '':
                    sketch_region_list = region_sketch_adaptor_weight.split('|')

                    for region_weight in sketch_region_list:
                        region, weight = region_weight.split('-')
                        region = eval(region)
                        weight = eval(weight)
                        feat_height, feat_width = feat_sketch.shape[2:]
                        start_h, start_w, end_h, end_w = region
                        start_h, end_h = start_h / height, end_h / height
                        start_w, end_w = start_w / width, end_w / width

                        start_h, start_w, end_h, end_w = math.ceil(start_h * feat_height), math.ceil(
                            start_w * feat_width), math.floor(end_h * feat_height), math.floor(end_w * feat_width)

                        spatial_adaptor_weight[start_h:end_h, start_w:end_w] = weight
                feat_sketch = spatial_adaptor_weight * feat_sketch
            else:
                feat_sketch = 0

            adapter_state[k] = torch.cat([feat_keypose + feat_sketch] * 2, dim=0)
            adapter_state_positive[k] = torch.cat([feat_keypose + feat_sketch], dim=0)

    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    loss_pre = 1.0e3
    refine_flag = True
    scale_range = np.linspace(1.0, 0.5, len(self.scheduler.timesteps))
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        if image_guidance:
            self.unet.sideload_processor.update_sideload(adapter_state_positive)  # T2i-adapter
        # Latent Re-initialization
        _ = self.unet(
                    latents,
                    timesteps[0],
                    encoder_hidden_states=prompt_embeds[1].unsqueeze(0),
                    cross_attention_kwargs={
                        'region_list': region_list_positive,
                        'height': height,
                        'width': width,
                        'step': 0,
                        'pipe_lora_list': pipe_lora_list,
                        'attention_store': attention_store,
                        'self_attn_mask': self_attn_mask,
                        'vis_region_rewrite_decoder': vis_region_rewrite_decoder,
                        'lora_token_idx': lora_token_idx,
                        'lora_pipe_idx': lora_pipe_idx,
                    }).sample

        item_cross_att_maps, self_att_map, bg_cross_att_maps = aggregate_attention(
            controller=attention_store,
            from_where=("up", "down", "mid"),
            only_cross=True,
            select=0)
        latents = replace_latents(item_cross_att_maps, region_list_positive, 1, lora_token_idx, latents)
        latents = latents * self.scheduler.init_noise_sigma
        # denoising process            
        for i, t in enumerate(timesteps):
            # if i < 25:  # max_iter_to_alter
            if refine_flag:
                with torch.enable_grad():
                    lr = 20 * np.sqrt(scale_range[i])
                    latents = latents.clone().detach().requires_grad_(True)
                    if image_guidance:
                        self.unet.sideload_processor.update_sideload(adapter_state_positive)  # T2i-adapter
                    
                    # Forward pass of denoising with text conditioning
                    _ = self.unet(
                    latents,
                    t,
                    encoder_hidden_states=prompt_embeds[1].unsqueeze(0),
                    cross_attention_kwargs={
                        'region_list': region_list_positive,
                        'height': height,
                        'width': width,
                        'step': i,
                        'pipe_lora_list': pipe_lora_list,
                        'attention_store': attention_store,
                        'self_attn_mask': self_attn_mask,
                        'vis_region_rewrite_decoder': vis_region_rewrite_decoder,
                        'lora_token_idx': lora_token_idx,
                        'lora_pipe_idx': lora_pipe_idx,
                    }).sample
                    self.unet.zero_grad()
                    for pipe_lora in pipe_lora_list:
                        pipe_lora.unet.zero_grad()
                    
                    # Get max activation value for each subject token
                    max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y, local_box_mean = _aggregate_and_get_max_attention_per_token(
                        attention_store=attention_store,
                        indices_to_alter=lora_token_idx,
                        smooth_attentions=True,
                        sigma=0.5,
                        kernel_size=3,
                        bbox=region_list,
                        top_k_ratio=0.2,
                        fill_loss=fill_loss,
                        gassion_sample=gassion_sample,
                        bg_loss=bg_loss,
                    )

                    # Perform gradient update

                    _, loss = _compute_loss(max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y, local_box_mean=local_box_mean)
                    if loss != 0:
                        latents = _update_latent(latents=latents, loss=loss, step_size=lr)
                    # print(f'Iteration {i} | Loss: {loss:0.4f}')
                    if (loss_pre < loss and i >= 5) or i >= 25:
                        refine_flag = False
                        # print("stop_refine in step {}".format(i))
                    else:
                        loss_pre = loss

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            if refine_flag and image_guidance:
                self.unet.sideload_processor.update_sideload(adapter_state)  # T2i-adapter
            # for lora in pipe_lora_list:
            #     lora.unet.sideload_processor.update_sideload(adapter_state)
            
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs={  # 这个在新定义的unet-foward中使用
                    'region_list': region_list,
                    'height': height,
                    'width': width,
                    'step': i,
                    'pipe_lora_list': pipe_lora_list,
                    'attention_store': attention_store,
                    'self_attn_mask': self_attn_mask,
                    'vis_region_rewrite_decoder': vis_region_rewrite_decoder,
                    'lora_pipe_idx': lora_pipe_idx,
                }
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    if output_type == 'latent':
        image = latents
        has_nsfw_concept = None
    elif output_type == 'pil':
        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # 10. Convert to PIL
        image = self.numpy_to_pil(image)
    else:
        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

    # Offload last model to CPU
    if hasattr(self, 'final_offload_hook') and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
