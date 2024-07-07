import argparse
import copy
import hashlib
import os.path
import torch
from diffusers import Adapter, StableDiffusionAdapterPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mixofshow.archs.edlora_override import revise_unet_attention_forward
from mixofshow.utils.diffusers_sample_util import StableDiffusion_PPlus_Sample, StableDiffusion_Sample
from region_lora.multi_lora_sample import Regionally_T2IAdaptor_Sample  # 用于可视化
from region_lora.utils.attn_util import AttentionStore, Visualizer
import glob
import re, time
import cv2
NUM_CROSS_ATTENTION_LAYERS = 16


def inference_image(pipe,
                    input_prompt,
                    input_neg_prompt=None,
                    generator=None,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    sketch_adaptor_weight=1.0,
                    region_sketch_adaptor_weight='',
                    keypose_adaptor_weight=1.0,
                    region_keypose_adaptor_weight='',
                    pipeline_type='sd',
                    **extra_kargs):
    if pipeline_type == 'adaptor_pplus' or pipeline_type == 'adaptor_sd':
        keypose_condition = extra_kargs.pop('keypose_condition')
        if keypose_condition is not None:
            keypose_adapter_input = [keypose_condition] * len(input_prompt)
        else:
            keypose_adapter_input = None

        sketch_condition = extra_kargs.pop('sketch_condition')
        if sketch_condition is not None:
            sketch_adapter_input = [sketch_condition] * len(input_prompt)
        else:
            sketch_adapter_input = None

        new_concept_cfg_list = extra_kargs.pop('new_concept_cfg_list')
        images = Regionally_T2IAdaptor_Sample(
            pipe,
            prompt=input_prompt,
            negative_prompt=input_neg_prompt,
            new_concept_cfg_list=new_concept_cfg_list,
            keypose_adapter_input=keypose_adapter_input,
            keypose_adaptor_weight=keypose_adaptor_weight,
            region_keypose_adaptor_weight=region_keypose_adaptor_weight,
            sketch_adapter_input=sketch_adapter_input,
            sketch_adaptor_weight=sketch_adaptor_weight,
            region_sketch_adaptor_weight=region_sketch_adaptor_weight,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            **extra_kargs).images
    else:
        raise NotImplementedError
    return images


def merge_lora_into_weight(original_state_dict, lora_state_dict, modification_layer_names, model_type, alpha, device):

    def get_lora_down_name(original_layer_name):
        if model_type == 'text_encoder':
            lora_down_name = original_layer_name.replace('q_proj.weight', 'q_proj_lora.down.weight') \
                .replace('k_proj.weight', 'k_proj_lora.down.weight') \
                .replace('v_proj.weight', 'v_proj_lora.down.weight') \
                .replace('out_proj.weight', 'out_proj_lora.down.weight') \
                .replace('fc1.weight', 'fc1_lora.down.weight') \
                .replace('fc2.weight', 'fc2_lora.down.weight')
        else:
            lora_down_name = k.replace('to_q.weight', 'to_q_lora.down.weight') \
                .replace('to_k.weight', 'to_k_lora.down.weight') \
                .replace('to_v.weight', 'to_v_lora.down.weight') \
                .replace('to_out.0.weight', 'to_out.0_lora.down.weight') \
                .replace('ff.net.0.proj.weight', 'ff.net.0.proj_lora.down.weight') \
                .replace('ff.net.2.weight', 'ff.net.2_lora.down.weight') \
                .replace('proj_out.weight', 'proj_out_lora.down.weight') \
                .replace('proj_in.weight', 'proj_in_lora.down.weight')

        return lora_down_name

    assert model_type in ['unet', 'text_encoder']
    new_state_dict = original_state_dict

    load_cnt = 0
    for k in modification_layer_names:
        lora_down_name = get_lora_down_name(k)
        lora_up_name = lora_down_name.replace('lora.down', 'lora.up')

        if lora_up_name in lora_state_dict:
            load_cnt += 1
            original_params = new_state_dict[k]
            # lora_down_params = lora_state_dict[lora_down_name].to(device).half()
            # lora_up_params = lora_state_dict[lora_up_name].to(device).half()
            lora_down_params = lora_state_dict[lora_down_name].to(device)
            lora_up_params = lora_state_dict[lora_up_name].to(device)
            if len(original_params.shape) == 4:
                lora_param = lora_up_params.squeeze() @ lora_down_params.squeeze()
                lora_param = lora_param.unsqueeze(-1).unsqueeze(-1)
            else:
                lora_param = lora_up_params @ lora_down_params
            merge_params = original_params + alpha * lora_param
            new_state_dict[k] = merge_params

    return new_state_dict


def merge_lora_weight(lora_state_dict, ori_state_dict, merge_part, device):
    assert merge_part in ['unet', 'text_encoder'], "merge_part must be unet or text_encoder"
    # merge lora into unet or text encoder
    LoRA_keys = []
    LoRA_keys += list(lora_state_dict.keys())
    LoRA_keys = set([key.replace('_lora.down', '').replace('_lora.up', '') for key in LoRA_keys])
    text_encoder_layer_names = LoRA_keys

    merged_state_dict = merge_lora_into_weight(
        ori_state_dict,
        lora_state_dict,
        text_encoder_layer_names,
        model_type=merge_part,
        alpha=1.0,
        device=device)
    return merged_state_dict

def add_new_concept(tokenizer, text_encoder, embedding, start_idx):
        new_token_names = [f'<new{start_idx + layer_id}>' for layer_id in range(NUM_CROSS_ATTENTION_LAYERS)]
        num_added_tokens = tokenizer.add_tokens(new_token_names)
        assert num_added_tokens == NUM_CROSS_ATTENTION_LAYERS
        new_token_ids = [tokenizer.convert_tokens_to_ids(token_name) for token_name in new_token_names]

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data

        token_embeds[new_token_ids] = token_embeds[new_token_ids].copy_(embedding)
        return start_idx + NUM_CROSS_ATTENTION_LAYERS, new_token_ids, new_token_names
def merge_pplus2sd_(pipe, lora_weight_path):
    # step 1: list pipe module
    pipe_lora = copy.deepcopy(pipe) 
    tokenizer, text_encoder = pipe_lora.tokenizer, pipe_lora.text_encoder
    lora_weight = torch.load(lora_weight_path, map_location=device)['params']
    print("load ckpt --> {}".format(lora_weight_path))
    new_concept_cfg = {}

    # step 2: load embedding into tokenizer/text_encoder:
    if 'new_concept_embedding' in lora_weight and len(lora_weight['new_concept_embedding']) != 0:
        start_idx = 0
        for concept_name, embedding in lora_weight['new_concept_embedding'].items():
            start_idx, new_token_ids, new_token_names = add_new_concept(tokenizer, text_encoder, embedding, start_idx)
            new_concept_cfg.update(
                {concept_name: {
                    'concept_token_ids': new_token_ids,
                    'concept_token_names': new_token_names
                }})

    # step 3: merge text_encoder_weight:
    if 'text_encoder' in lora_weight and len(lora_weight['text_encoder']) != 0:
        new_text_encoder_weights = merge_lora_weight(lora_weight['text_encoder'], text_encoder.state_dict(), 'text_encoder', pipe.device)
        
        sd_textenc_state_dict = copy.deepcopy(text_encoder.state_dict())

        for k in new_text_encoder_weights.keys():
            sd_textenc_state_dict[k] = new_text_encoder_weights[k]
        text_encoder.load_state_dict(sd_textenc_state_dict)

    # step 4: load unet_weight:
    if 'unet' in lora_weight and len(lora_weight['unet']) != 0:
        unet_lora_weight = lora_weight['unet']
        for k, v in unet_lora_weight.items():
            unet_lora_weight[k] = v.half()                
    return new_concept_cfg, pipe_lora, unet_lora_weight



def build_model(args, device):
    def build_adapter(pipe, device):
        pipe.keypose_adapter = Adapter(
                cin=int(3 * 64), channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False).half()
        pipe.keypose_adapter.load_state_dict(torch.load(keypose_adaptor_model))
        pipe.keypose_adapter = pipe.keypose_adapter.to(device)

        pipe.sketch_adapter = Adapter(
            cin=int(64), channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False).half()
        pipe.sketch_adapter.load_state_dict(torch.load(sketch_adaptor_model))
        pipe.sketch_adapter = pipe.sketch_adapter.to(device)
        return pipe
        
    pretrained_model = args.pretrained_model
    combined_model_root = args.combined_model_root
    sketch_adaptor_model = args.sketch_adaptor_model
    keypose_adaptor_model = args.keypose_adaptor_model
    pipeline_type = args.pipeline_type
    
    if pipeline_type == 'adaptor_pplus' or pipeline_type == 'adaptor_sd':
        new_concept_cfg_list = []
        tokenizer_list = []
        text_encoder_list = []
        unet_lora_weight_list = []
        print("start load base model")
        start = time.time()
        pipe_base = StableDiffusionAdapterPipeline.from_pretrained(
            pretrained_model, torch_dtype=torch.float16, safety_checker=None).to(device)
        print("load base model cost {}s".format(time.time()-start))
        scheduler = DPMSolverMultistepScheduler.from_config(pipe_base.scheduler.config)
        pipe_base.scheduler = scheduler

        combined_model_list = glob.glob(os.path.join(combined_model_root,"*.pth"))
        print("start load lora model")
        
        # find background lora
        bg_lora_flag = False
        prompts = [args.prompt]
        prompts_rewrite = [''] if args.no_region else [args.prompt_rewrite]
        pattern = r'<(.*?)>'
        matches = re.findall(pattern, prompts_rewrite[0])
        # find background lora
        match_scene = re.findall(pattern, prompts[0])
        if len(match_scene) != 0:
            match_pattern = '<' + match_scene[0] + '>'
            bg_lora_flag = True
        
        for conbined_model in combined_model_list:
            if pipeline_type == 'adaptor_pplus':
                start = time.time()
                new_concept_cfg, pipe_lora, unet_lora_weight = merge_pplus2sd_(pipe_base, conbined_model)
                new_concept_cfg_list.append(new_concept_cfg)
                tokenizer_list.append(pipe_lora.tokenizer)
                text_encoder_list.append(pipe_lora.text_encoder)
                unet_lora_weight_list.append(unet_lora_weight)
                # background lora merge in the unet of basemodel
                if bg_lora_flag: 
                    if match_pattern in new_concept_cfg.keys():
                        unet_state_dict = pipe_lora.unet.state_dict()
                        sd_unet_state_dict = copy.deepcopy(unet_state_dict)
                        new_unet_weights = merge_lora_weight(unet_lora_weight, unet_state_dict, 'unet', device)
                        for k in new_unet_weights.keys():
                            sd_unet_state_dict[k] = new_unet_weights[k]
                        pipe_lora.unet.load_state_dict(sd_unet_state_dict)
                        
                        bg_new_concept_cfg = new_concept_cfg
                        new_concept_cfg_list.pop()
                        tokenizer_list.pop()
                        text_encoder_list.pop()
                        unet_lora_weight_list.pop()
                        pipe_bg = copy.deepcopy(pipe_lora)
                else:
                    pipe_bg = copy.deepcopy(pipe_base)
                    bg_new_concept_cfg = new_concept_cfg  # When there is no background lora, you can define it casually
                
                base_unet = copy.deepcopy(pipe_base.unet)
                del pipe_lora
                torch.cuda.empty_cache()
                print("load lora model cost {}s".format(time.time()-start))
            else:
                raise NotImplementedError
    else:
        raise NotImplementedError
    
    assert len(matches) == len(new_concept_cfg_list) * 2, f"Check the matching relationship between{matches} and the number of LoRAs in the {args.combined_model_root} directory"

    pipe_bg = build_adapter(pipe_bg, device)
    
    for unet_lora in unet_lora_weight_list:
        keys_to_delete = [key for key in unet_lora if "to_q_lora" in key]
        for k in keys_to_delete:
            del unet_lora[k]
        keys_to_delete = [key for key in unet_lora if "attn1" in key]
        for k in keys_to_delete:
            del unet_lora[k]
    
    del pipe_base
    torch.cuda.empty_cache()
    return new_concept_cfg_list, bg_new_concept_cfg, tokenizer_list, text_encoder_list, unet_lora_weight_list, base_unet, pipe_bg


def prepare_text(prompt, region_prompts, height, width):
    '''
    Args:
        prompt_entity: [subject1]-*-[attribute1]-*-[Location1]|[subject2]-*-[attribute2]-*-[Location2]|[global text]
    Returns:
        full_prompt: subject1, attribute1 and subject2, attribute2, global text
        context_prompt: subject1 and subject2, global text
        entity_collection: [(subject1, attribute1), Location1]
    '''
    region_collection = []

    regions = region_prompts.split('|')

    for region in regions:
        if region == '':
            break
        prompt_region, neg_prompt_region, pos = region.split('-*-')
        prompt_region = prompt_region.replace('[', '').replace(']', '')
        neg_prompt_region = neg_prompt_region.replace('[', '').replace(']', '')
        pos = eval(pos)
        if len(pos) == 0:
            pos = [0, 0, 1, 1]
        else:
            pos[0], pos[2] = pos[0] / height, pos[2] / height
            pos[1], pos[3] = pos[1] / width, pos[3] / width

        region_collection.append((prompt_region, neg_prompt_region, pos))
    return (prompt, region_collection)


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--pretrained_model', default='experiments/pretrained_models/anything-v4.0', type=str)
    parser.add_argument(
        '--combined_model_root',
        default='experiments/composed_edlora/anythingv4',
        type=str)
    parser.add_argument('--sketch_adaptor_model', default=None, type=str)
    parser.add_argument('--sketch_condition', default=None, type=str)
    parser.add_argument('--sketch_adaptor_weight', default=1.0, type=float)
    parser.add_argument('--region_sketch_adaptor_weight', default='', type=str)
    parser.add_argument('--keypose_adaptor_model', default=None, type=str)
    parser.add_argument('--keypose_condition', default=None, type=str)
    parser.add_argument('--keypose_adaptor_weight', default=1.0, type=float)
    parser.add_argument('--region_keypose_adaptor_weight', default='', type=str)
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--pipeline_type', default='sd', type=str)
    parser.add_argument('--prompt', default='photo of a toy', type=str)
    parser.add_argument('--negative_prompt', default='', type=str)
    parser.add_argument('--prompt_rewrite', default='', type=str)
    parser.add_argument('--seed', default=16141, type=int)
    parser.add_argument('--image_guidance', default=False, type=bool)
    parser.add_argument('--suffix', default='', type=str)
    parser.add_argument('--no_region', action='store_true')
    parser.add_argument('--gen_num', default=50, type=int)
    parser.add_argument('--mode', default='', type=str)
    parser.add_argument('--exp_name', default='defult', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    new_concept_cfg_list, bg_new_concept_cfg, tokenizer_list, text_encoder_list, unet_lora_weight_list, base_unet, pipe_bg = build_model(args, device)  # pipe_lora中的顺序和new_concept_cfg_list是对应的
        
    if args.pipeline_type == 'adaptor_pplus':  # condition
        if args.sketch_condition is not None and os.path.exists(args.sketch_condition):
            sketch_condition = Image.open(args.sketch_condition).convert('L')
            width_sketch, height_sketch = sketch_condition.size
            print('use sketch condition')
        else:
            sketch_condition, width_sketch, height_sketch = None, 0, 0
            print('skip sketch condition')

        if args.keypose_condition is not None and os.path.exists(args.keypose_condition):
            keypose_condition = Image.open(args.keypose_condition).convert('RGB')
            width_pose, height_pose = keypose_condition.size
            print('use pose condition')
        else:
            keypose_condition, width_pose, height_pose = None, 0, 0
            print('skip pose condition')

        if width_sketch != 0 and width_pose != 0:
            assert width_sketch == width_pose and height_sketch == height_pose, 'conditions should be same size'
        # width, height = max(width_pose, width_sketch), max(height_pose, height_sketch)

        if args.mode == 'anime':
            print("anime task")
            width, height = 1024, 512
            attn_resolution = [16, 32]
        else:
            print("real task")
            width, height = 1024, 768
            attn_resolution = [24,32]

        kwargs = {
            'sketch_condition': sketch_condition,
            'keypose_condition': keypose_condition,
            'new_concept_cfg_list': new_concept_cfg_list,
            'bg_new_concept_cfg': bg_new_concept_cfg,
            'height': height,
            'width': width,
            'tokenizer_list': tokenizer_list,
            'text_encoder_list': text_encoder_list,
            'unet_lora_weight_list': unet_lora_weight_list,
            'base_unet': base_unet,
            'image_guidance': args.image_guidance
        }
        prompts = [args.prompt]
        prompts_rewrite = [''] if args.no_region else [args.prompt_rewrite]
        input_prompt = [prepare_text(p, p_w, height, width) for p, p_w in zip(prompts, prompts_rewrite)]
        save_prompt = input_prompt[0][0]
    elif args.pipeline_type == 'sd':
        kwargs = {
            'new_concept_cfg_list': new_concept_cfg_list,
            'height': 512,
            'width': 512,
        }
        input_prompt = [args.prompt]
        save_prompt = input_prompt[0]
    else:
        raise NotImplementedError
    
    attention_store = AttentionStore('all', resolution=attn_resolution)
    image = inference_image(
        pipe_bg,
        input_prompt=input_prompt,
        input_neg_prompt=[args.negative_prompt] * len(input_prompt),
        generator=torch.Generator(device).manual_seed(args.seed),
        sketch_adaptor_weight=args.sketch_adaptor_weight,
        region_sketch_adaptor_weight=args.region_sketch_adaptor_weight,
        keypose_adaptor_weight=args.keypose_adaptor_weight,
        region_keypose_adaptor_weight=args.region_keypose_adaptor_weight,
        pipeline_type=args.pipeline_type,
        attention_store=attention_store,
        self_attn_mask=True,
        vis_region_rewrite_decoder = True,
        **kwargs)

    print(f'save to: {args.save_dir}')
    save_name = f'{args.seed}.png'
    save_dir = os.path.join(args.save_dir, args.exp_name, f'seed-{args.seed}')
    os.makedirs(save_dir, exist_ok=True)
    image[0].save(os.path.join(save_dir, save_name))