export config_file="mitsuha_anythingv4"

python scripts/lora_composer_scripts/Weight_Fusion_EDLoRA.py \
    --concept_cfg="datasets/data_cfgs/MixofShow/region_lora/anime/${config_file}.json" \
    --save_path="experiments/composed_edlora/anythingv4/${config_file}" \
    --pretrained_models="experiments/pretrained_models/anything-v4.0"