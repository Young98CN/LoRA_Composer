import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='', required=True)
    parser.add_argument('--save_path', type=str, default='experiments/MixofShow_Results/EDLoRA_Models')
    return parser.parse_args()

if __name__ == '__main__':
    arg = parse_args()
    text_encoder_list = []
    unet_list = []

    model = torch.load(arg.ckpt_path)['params']

    new_state_dict_text_encoder = {}
    new_state_dict_unet = {}

    for k,v in model['text_encoder'].items():
        if 'down' in k:
            k_old = k.replace('.lora_down', '_lora.down')
        else:
            k_old = k.replace('.lora_up', '_lora.up')
        
        new_state_dict_text_encoder.update({k_old:v})
            
    for k,v in model['unet'].items():
        key = k.split('.', 1)[-1]
        if 'down' in key:
            k_old = k.replace('.lora_down', '_lora.down')
        else:
            k_old = k.replace('.lora_up', '_lora.up')
        
        new_state_dict_unet.update({k_old:v})
        

    new_ckpt = {'params':{'new_concept_embedding':model['new_concept_embedding'],
                'text_encoder':new_state_dict_text_encoder,
                'unet':new_state_dict_unet}}

    torch.save(new_ckpt, arg.save_path)