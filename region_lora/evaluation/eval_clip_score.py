import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append('/data/yangyang/mixofshow/scripts/evaluation_scripts/clipscore_main')
import cv2
import json
from tqdm import tqdm
import torch, shutil
import clip
import warnings
from scripts.evaluation_scripts.clipscore_main.clipscore_image_alignment import extract_all_images, get_clip_score
from scripts.evaluation_scripts.clipscore_main.clipscore import get_clip_score as get_text_clip_score
import numpy as np


class_mapping = {
    'Hina_Amano': 'girl',
    'Tezuka_Kunimitsu': 'man',
    'Mitsuha_Miyamizu': 'girl',
    'Son_Goku': 'man',
    'Miyazono_Kaori': 'girl',
    'Haibara_Ai': 'girl',
    'Geoffrey_Hinton': 'man',
    'Harry_Potter': 'man',
    'Yann_LeCun': 'man',
    'Yoshua_bengio': 'man',
    'Hermione_Granger': 'woman',
    'catA': 'cat',
    'dogA': 'dog',
    'dogB': 'dog',
    'pyramid': 'pyramid',
    'rock': 'rock'
}

data_path_mapping = {
    'Hina_Amano': 'datasets/data/characters/anime/Hina_Amano',
    'Tezuka_Kunimitsu': 'datasets/data/characters/anime/Tezuka_Kunimitsu',
    'Mitsuha_Miyamizu': 'datasets/data/characters/anime/Mitsuha_Miyamizu',
    'Son_Goku': 'datasets/data/characters/anime/Son_Goku',
    'Miyazono_Kaori': 'datasets/data/characters/anime/Miyazono_Kaori',
    'Haibara_Ai': 'datasets/data/characters/anime/Haibara_Ai',
    'Geoffrey_Hinton': 'datasets/data/characters/real/Geoffrey_Hinton',
    'Harry_Potter': 'datasets/data/characters/real/Harry_Potter',
    'Yann_LeCun': 'datasets/data/characters/real/Yann_LeCun',
    'Yoshua_bengio': 'datasets/data/characters/real/Yoshua_bengio',
    'Hermione_Granger': 'datasets/data/characters/real/Hermione_Granger',
    'catA': 'datasets/data/objects/real/cat/catA',
    'dogA': 'datasets/data/objects/real/dog/dogA',
    'dogB': 'datasets/data/objects/real/dog/dogB',
    'pyramid': 'datasets/data/scenes/real/Pyramid',
    'rock': 'datasets/data/scenes/real/Wululu'
}

def compute_img_score(reference_image_dir, image_dir):
    image_paths = [
        os.path.join(image_dir, path) for path in os.listdir(image_dir)
        if path.endswith(('.png', '.jpg', '.jpeg', '.tiff'))
    ]
    image_ids = [os.path.basename(path) for path in image_paths]

    ref_image_paths = [
        os.path.join(reference_image_dir, path) for path in os.listdir(reference_image_dir)
        if path.endswith(('.png', '.jpg', '.jpeg', '.tiff'))
    ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        warnings.warn('CLIP runs in full float32 on CPU. Results in paper were computed on GPU, which uses float16. '
                      'If you\'re reporting results on CPU, please note this when you report.')
    model, transform = clip.load('ViT-B/32', device=device, jit=False)
    model.eval()

    image_feats = extract_all_images(image_paths, model, device, batch_size=64, num_workers=0)
    ref_image_feats = extract_all_images(ref_image_paths, model, device, batch_size=64, num_workers=0)
    
    print('img_shape-->{}'.format(image_feats.shape))
    print('ref_shape-->{}'.format(ref_image_feats.shape))

    # get image-text clipscore
    _, per_instance_image_text, candidate_feats = get_clip_score(model, image_feats, ref_image_feats, device)

    scores = {
        image_id: {
            'CLIPScore': float(clipscore)
        }
        for image_id, clipscore in zip(image_ids, per_instance_image_text)
    }
    return np.mean([s['CLIPScore'] for s in scores.values()])

def compute_text_score(image_dir, candidates_json):
    image_paths = [
        os.path.join(image_dir, path) for path in os.listdir(image_dir)
        if path.endswith(('.png', '.jpg', '.jpeg', '.tiff'))
    ]
    image_ids = [os.path.basename(path) for path in image_paths]
    
    print(len(image_ids))

    with open(candidates_json) as f:
        candidates = json.load(f)

    candidates = [candidates[cid] for cid in image_ids]
    
    print(len(candidates))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        warnings.warn('CLIP runs in full float32 on CPU. Results in paper were computed on GPU, which uses float16. '
                      'If you\'re reporting results on CPU, please note this when you report.')
    model, transform = clip.load('ViT-B/32', device=device, jit=False)
    model.eval()

    image_feats = extract_all_images(image_paths, model, device, batch_size=64, num_workers=0)

    # get image-text clipscore
    _, per_instance_image_text, candidate_feats = get_text_clip_score(model, image_feats, candidates, device)

    scores = {
        image_id: {
            'CLIPScore': float(clipscore)
        }
        for image_id, clipscore in zip(image_ids, per_instance_image_text)
    }
    return np.mean([s['CLIPScore'] for s in scores.values()])
    

def eval_one(cfg_file_path, output_json_root):
    mean_factor = 0
    
    output_json_path = os.path.join(output_json_root, os.path.basename(cfg_file_path))
    if os.path.exists(output_json_path):
        print ('skip:{}'.format(output_json_path))
        return 
    output_json = {}
    image_score_all = 0
    text_score_all = 0
    with open(cfg_file_path, 'r') as file:
        data = json.load(file)

    for i in tqdm(data):
        # step1: crop image
        img_root = i["res_path"]
        save_root = i["crop_img_save_path"]

        if os.path.exists(save_root):
            shutil.rmtree(save_root)
        os.makedirs(save_root)

        crop_ranges = i["crop_box"]
        role_name = i["role_name"]
        json_prompt = [{} for _ in range(len(role_name))]

        for role in role_name:
            if os.path.exists(os.path.join(save_root, role)):
                pass
            else:
                os.makedirs(os.path.join(save_root, role))

        for root, folder, files in os.walk(img_root):
            for file in files:
                if file.endswith('.png'):
                    img_path = os.path.join(root, file)

                    # 加载图片
                    img = cv2.imread(img_path)
                    
                    for idx in range(len(crop_ranges)):
                        x1, y1, x2, y2 = crop_ranges[idx]
                        role = role_name[idx]
                        cropped_img = img[x1:x2, y1:y2]
                        cv2.imwrite(os.path.join(save_root, role, file), cropped_img)
                        # 获取caption
                        caption = i["region_prompt"].format(class_mapping[role])
                        json_prompt[idx].update({file: caption})
        # step2: eval clip score
        img_score_store = 0
        text_score_store = 0
        for idx, role in enumerate(role_name):
            mean_factor += 1
            crop_image_path = os.path.join(save_root, role)
            # save caption json
            caption_json_path = os.path.join(save_root,"{}_caption.json".format(role))
            with open(caption_json_path, 'w') as f:
                json.dump(json_prompt[idx], f)
            
            # eval image clip score
            ref_image_path = os.path.join(data_path_mapping[role], "image")
            
            img_score = compute_img_score(ref_image_path, crop_image_path)
            output_json.update({crop_image_path + "(Image-score)" : '{:.4f}'.format(img_score)})
            img_score_store += img_score
            image_score_all += img_score
            
            # eval text clip score
            text_score = compute_text_score(crop_image_path, caption_json_path)
            output_json.update({crop_image_path + "(Text-score)" : '{:.4f}'.format(text_score)})
            text_score_store += text_score
            text_score_all += text_score
        
        mean_img_score = img_score_store / len(role_name)
        output_json.update({save_root + "(Image-score)" : '{:.4f}'.format(mean_img_score)})
        
        mean_text_score = text_score_store / len(role_name)
        output_json.update({save_root + "(Text-score)" : '{:.4f}'.format(mean_text_score)})


    mean_image_score_all = image_score_all / mean_factor
    output_json.update({"Image-Alignment": '{:.4f}'.format(mean_image_score_all)})  
            
    mean_text_score_all = text_score_all / mean_factor
    output_json.update({"Text-Alignment": '{:.4f}'.format(mean_text_score_all)})     
        
    with open(output_json_path, 'w') as f:
        json.dump(output_json, f)

if __name__ == '__main__':
    cfg_file_root = "/data/yangyang/mixofshow/eval_cfg/real/4real.json"
    # cfg_file_root = "/data/yangyang/mixofshow/eval_cfg/ablition/no_all"
    output_json_root = "/data/yangyang/mixofshow/results/clipscore"
    
    if os.path.isdir(cfg_file_root):
        print("eval dir")
        for cfg_file_path in os.listdir(cfg_file_root):
            cfg_file_path = os.path.join(cfg_file_root, cfg_file_path)
            print(cfg_file_path)
            eval_one(cfg_file_path, output_json_root)
    else:
        print("eval_one")
        eval_one(cfg_file_root, output_json_root)
        
   