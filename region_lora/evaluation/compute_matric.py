import os
import json


data_root = "/data/yangyang/mixofshow/results/clipscore/uncondition"
save_path = os.path.join(data_root, "matrics.json")
exp_dir = os.listdir(data_root)
res_out = {}

mean_image_score_all_baseline = 0
mean_text_score_all_baseline = 0
mean_image_score_all = 0
mean_text_score_all = 0

for exp in exp_dir:
    mean_image_score = 0
    mean_text_score = 0
    mean_image_score_baseline = 0
    mean_text_score_baseline = 0
    num_exp_baseline = 0
    num_exp = 0
    for file in os.listdir(os.path.join(data_root, exp)):
        if file.endswith(".json"):
            with open(os.path.join(data_root, exp, file), "r") as f:
                data = json.load(f)
                if "_mixofshow" in file:
                    mean_image_score_baseline += float(data["Image-Alignment"])
                    mean_text_score_baseline += float(data["Text-Alignment"])
                    num_exp_baseline += 1
                else:
                    mean_image_score += float(data["Image-Alignment"])
                    mean_text_score += float(data["Text-Alignment"])
                    num_exp += 1
    if num_exp_baseline != 0:
        mean_image_score_baseline = mean_image_score_baseline / num_exp_baseline
        mean_text_score_baseline = mean_text_score_baseline / num_exp_baseline
    if num_exp != 0:
        mean_image_score = mean_image_score / num_exp
        mean_text_score = mean_text_score / num_exp
    
    res_out.update({
        exp:{
            "mean_image_score_baseline": "{:.4f}".format(mean_image_score_baseline),
            "mean_text_score_baseline": "{:.4f}".format(mean_text_score_baseline),
            "mean_image_score": "{:.4f}".format(mean_image_score),
            "mean_text_score": "{:.4f}".format(mean_text_score)
            }
    })

    mean_image_score_all_baseline += mean_image_score_baseline
    mean_text_score_all_baseline += mean_text_score_baseline
    mean_image_score_all += mean_image_score
    mean_text_score_all += mean_text_score
    
mean_image_score_all_baseline = mean_image_score_all_baseline / len(exp_dir)
mean_text_score_all_baseline = mean_text_score_all_baseline / len(exp_dir)
mean_text_score_all = mean_text_score_all / len(exp_dir)
mean_image_score_all = mean_image_score_all / len(exp_dir)
res_out.update({
    "mean_image_score_all_baseline": "{:.4f}".format(mean_image_score_all_baseline),
    "mean_text_score_all_baseline": "{:.4f}".format(mean_text_score_all_baseline),
    "mean_image_score_all": "{:.4f}".format(mean_image_score_all),
    "mean_text_score_all": "{:.4f}".format(mean_text_score_all)
})

with open(save_path, "w") as f:
   json.dump(res_out, f)
