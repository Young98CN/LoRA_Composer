import os
import json


data_root = "/data/yangyang/mixofshow/results/clipscore/ablation"
save_path = os.path.join(data_root, "matrics.json")
exp_dir = os.listdir(data_root)
res_out = {}

for exp in exp_dir:
    mean_image_score = 0
    mean_text_score = 0
    num_exp = 0
    for file in os.listdir(os.path.join(data_root, exp)):
        if file.endswith(".json"):
            with open(os.path.join(data_root, exp, file), "r") as f:
                data = json.load(f)
                mean_image_score += float(data["Image-Alignment"])
                mean_text_score += float(data["Text-Alignment"])
                num_exp += 1

    
    mean_image_score = mean_image_score / num_exp
    mean_text_score = mean_text_score / num_exp
    
    res_out.update({
        exp:{
            "mean_image_score": "{:.4f}".format(mean_image_score),
            "mean_text_score": "{:.4f}".format(mean_text_score)
            }
    })

with open(save_path, "w") as f:
   json.dump(res_out, f)
