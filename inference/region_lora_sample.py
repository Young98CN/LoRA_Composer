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


def merge_pplus2sd_(pipe, lora_weight_path):

    def add_new_concept(embedding):
        new_token_names = [f'<new{start_idx + layer_id}>' for layer_id in range(NUM_CROSS_ATTENTION_LAYERS)]
        num_added_tokens = tokenizer.add_tokens(new_token_names)
        assert num_added_tokens == NUM_CROSS_ATTENTION_LAYERS
        new_token_ids = [tokenizer.convert_tokens_to_ids(token_name) for token_name in new_token_names]

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data

        token_embeds[new_token_ids] = token_embeds[new_token_ids].copy_(embedding)
        return start_idx + NUM_CROSS_ATTENTION_LAYERS, new_token_ids, new_token_names

    # step 1: list pipe module
    pipe_lora = copy.deepcopy(pipe) 
    tokenizer, unet, text_encoder = pipe_lora.tokenizer, pipe_lora.unet, pipe_lora.text_encoder
    lora_weight = torch.load(lora_weight_path, map_location='cpu')['params']
    print("load ckpt --> {}".format(lora_weight_path))
    new_concept_cfg = {}

    # step 2: load embedding into tokenizer/text_encoder:
    if 'new_concept_embedding' in lora_weight and len(lora_weight['new_concept_embedding']) != 0:
        start_idx = 0
        NUM_CROSS_ATTENTION_LAYERS = 16
        for idx, (concept_name, embedding) in enumerate(lora_weight['new_concept_embedding'].items()):
            start_idx, new_token_ids, new_token_names = add_new_concept(embedding)
            new_concept_cfg.update(
                {concept_name: {
                    'concept_token_ids': new_token_ids,
                    'concept_token_names': new_token_names
                }})

    # step 3: load text_encoder_weight:
    if 'text_encoder' in lora_weight and len(lora_weight['text_encoder']) != 0:
        sd_textenc_state_dict = copy.deepcopy(text_encoder.state_dict())

        for k in lora_weight['text_encoder'].keys():
            sd_textenc_state_dict[k] = lora_weight['text_encoder'][k]
        text_encoder.load_state_dict(sd_textenc_state_dict)

    if 'unet' in lora_weight and len(lora_weight['unet']) != 0:
        sd_unet_state_dict = copy.deepcopy(unet.state_dict())

        for k in lora_weight['unet'].keys():
            sd_unet_state_dict[k] = lora_weight['unet'][k]

        unet.load_state_dict(sd_unet_state_dict)
    return new_concept_cfg, pipe_lora


def merge2sd_(pipe, lora_weight_path):

    # step 1: list pipe module
    tokenizer, unet, text_encoder = pipe.tokenizer, pipe.unet, pipe.text_encoder
    lora_weight = torch.load(lora_weight_path, map_location='cpu')['params']

    # step 2: load embedding into tokenizer/text_encoder:
    if 'new_concept_embedding' in lora_weight and len(lora_weight['new_concept_embedding']) != 0:
        new_concept_embedding = list(lora_weight['new_concept_embedding'].keys())

        for new_token in new_concept_embedding:
            # Add the placeholder token in tokenizer
            _ = tokenizer.add_tokens(new_token)
            new_token_id = tokenizer.convert_tokens_to_ids(new_token)
            text_encoder.resize_token_embeddings(len(tokenizer))
            token_embeds = text_encoder.get_input_embeddings().weight.data
            token_embeds[new_token_id] = lora_weight['new_concept_embedding'][new_token]

    # step 3: load text_encoder_weight:
    if 'text_encoder' in lora_weight and len(lora_weight['text_encoder']) != 0:
        sd_textenc_state_dict = copy.deepcopy(text_encoder.state_dict())

        for k in lora_weight['text_encoder'].keys():
            sd_textenc_state_dict[k] = lora_weight['text_encoder'][k]
        text_encoder.load_state_dict(sd_textenc_state_dict)

    if 'unet' in lora_weight and len(lora_weight['unet']) != 0:
        sd_unet_state_dict = copy.deepcopy(unet.state_dict())

        for k in lora_weight['unet'].keys():
            sd_unet_state_dict[k] = lora_weight['unet'][k]

        unet.load_state_dict(sd_unet_state_dict)


def build_model(pretrained_model, combined_model_root, sketch_adaptor_model, keypose_adaptor_model, pipeline_type, device):
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
        
        
    pipe_lora_list = []
    if pipeline_type == 'adaptor_pplus' or pipeline_type == 'adaptor_sd':
        new_concept_cfg_list = []
        print("start load base model")
        start = time.time()
        pipe_base = StableDiffusionAdapterPipeline.from_pretrained(
            pretrained_model, torch_dtype=torch.float16, safety_checker=None).to(device)
        print("load base model cost {}s".format(time.time()-start))
        scheduler = DPMSolverMultistepScheduler.from_config(pipe_base.scheduler.config)
        pipe_base.scheduler = scheduler

        combined_model_list = glob.glob(os.path.join(combined_model_root,"**/combined_model_.pth"))
        print("start load lora model")
        for conbined_model in combined_model_list:
            if pipeline_type == 'adaptor_pplus':
                start = time.time()
                new_concept_cfg, pipe_lora = merge_pplus2sd_(pipe_base, lora_weight_path=conbined_model)
                print("load lora model cost {}s".format(time.time()-start))
                new_concept_cfg_list.append(new_concept_cfg)
                pipe_lora = build_adapter(pipe_lora, device)
                pipe_lora_list.append(pipe_lora)
            else:
                raise NotImplementedError
    else:
        raise NotImplementedError
    
    pipe_base = build_adapter(pipe_base, device)
    return pipe_lora_list, new_concept_cfg_list, pipe_base


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
        pos = eval(pos)  # 将字符串转换为list
        if len(pos) == 0:
            pos = [0, 0, 1, 1]
        else:
            pos[0], pos[2] = pos[0] / height, pos[2] / height  # 缩放到生成图像的比例
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

    pipe_lora_list, new_concept_cfg_list, pipe_base = build_model(args.pretrained_model, args.combined_model_root, args.sketch_adaptor_model,
                                        args.keypose_adaptor_model, args.pipeline_type, device)  # pipe_lora中的顺序和new_concept_cfg_list是对应的
    
    prompts = [args.prompt]
    prompts_rewrite = [''] if args.no_region else [args.prompt_rewrite]
    pattern = r'<(.*?)>'
    matches = re.findall(pattern, prompts_rewrite[0])
    # find background lora
    match_scene = re.findall(pattern, prompts[0])
    if len(match_scene) != 0:
        match_pattern = '<' + match_scene[0] + '>'
        for i, item in enumerate(new_concept_cfg_list):
            if match_pattern in item.keys(): 
                pipe_base = pipe_lora_list[i]  # replace basemodel with background lora
                pipe_lora_list.pop(i)
                bg_new_concept_cfg = new_concept_cfg_list[i]
                new_concept_cfg_list.pop(i)
                break
    else:
        bg_new_concept_cfg = new_concept_cfg_list[0]  # When there is no background lora, you can define it casually
    print(len(matches), len(pipe_lora_list) * 2)
    assert len(matches) == len(pipe_lora_list) * 2, f"Check the matching relationship between{matches} and the number of LoRAs in the {args.combined_model_root} directory"
        
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
            'pipe_lora_list': pipe_lora_list,
            'image_guidance': args.image_guidance
        }
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
    vis = Visualizer(pipe_base, pipe_lora_list,attention_store,draw_region=False)
    image = inference_image(
        pipe_base,
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