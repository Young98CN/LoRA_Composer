export config_file="dogA"

python scripts/lora_composer_scripts/Weight_Fusion_EDLoRA.py \
    --concept_cfg="datasets/data_cfgs/MixofShow/region_lora/real/dogA.json" \
    --save_path="experiments/composed_edlora/chilloutmix/${config_file}" \
    --pretrained_models="experiments/pretrained_models/chilloutmix"