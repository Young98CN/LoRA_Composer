combined_model_root="experiments/link_folder"
image_guidance=''
task='anime'
#---------------------------------------------five_anime_sparse_box_new-------------------------------------------

five_anime_2=1
expdir="5anime_2"

if [ ${five_anime_2} -eq 1 ]
then
  echo $expdir
  context_prompt='three girls, a boy and a man are walking near a lake'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[a <hina1> <hina2>, near a lake]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[61, 0, 512, 160]'

  region2_prompt='[a <tezuka1> <tezuka2>, near a lake]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[20, 208, 512, 378]'

  region3_prompt='[a <ai1> <ai2>, near a lake]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[72, 426, 512, 586]'

  region4_prompt='[a <son1> <son2>, near a lake]'
  region4_neg_prompt="[${context_neg_prompt}]"
  region4='[9, 634, 512, 814]'

  region5_prompt='[a <kaori1> <kaori2>, near a lake]'
  region5_neg_prompt="[${context_neg_prompt}]"
  region5='[71, 862, 512, 1023]'
  
  keypose_condition='datasets/validation_spatial_condition/multi-characters/anime_pose/5anime+ai.png'
  keypose_adaptor_weight=1.0
  sketch_condition=''
  sketch_adaptor_weight=1.0
  region_keypose_adaptor_weight="${region1}-1.0|${region2}-1.0|${region3}-1.0|${region4}-1.0|${region5}-1.0"

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}|${region4_prompt}-*-${region4_neg_prompt}-*-${region4}|${region5_prompt}-*-${region5_neg_prompt}-*-${region5}"

  python inference/region_lora_sample.py \
    --pretrained_model="experiments/pretrained_models/anything-v4.0" \
    --combined_model_root="${combined_model_root}/${expdir}" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --region_keypose_adaptor_weight="${region_keypose_adaptor_weight}" \
    --keypose_condition=${keypose_condition} \
    --save_dir="results/test/${expdir}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=3251 \
    --image_guidance=${image_guidance} \
    --mode=${task}
fi

#---------------------------------------------five_anime_sparse_box_new-------------------------------------------

five_anime_3=1
expdir="5anime_3"

if [ ${five_anime_3} -eq 1 ]
then
  echo $expdir
  context_prompt='four girls, a man are walking near a lake'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[a <hina1> <hina2>, near a lake]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[61, 0, 512, 160]'

  region2_prompt='[a <mitsuha1> <mitsuha2>, near a lake]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[20, 208, 512, 378]'

  region3_prompt='[a <ai1> <ai2>, near a lake]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[72, 426, 512, 586]'

  region4_prompt='[a <son1> <son2>, near a lake]'
  region4_neg_prompt="[${context_neg_prompt}]"
  region4='[9, 634, 512, 814]'

  region5_prompt='[a <kaori1> <kaori2>, near a lake]'
  region5_neg_prompt="[${context_neg_prompt}]"
  region5='[71, 862, 512, 1023]'

  keypose_condition='datasets/validation_spatial_condition/multi-characters/anime_pose/5anime+ai2.png'
  keypose_adaptor_weight=0.0
  sketch_condition=''
  sketch_adaptor_weight=1.0
  # region_keypose_adaptor_weight="${region1}-1.0|${region2}-1.0|${region3}-1.0|${region4}-1.0|${region5}-1.0"

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}|${region4_prompt}-*-${region4_neg_prompt}-*-${region4}|${region5_prompt}-*-${region5_neg_prompt}-*-${region5}"

  python inference/region_lora_sample.py \
    --pretrained_model="experiments/pretrained_models/anything-v4.0" \
    --combined_model_root="${combined_model_root}/${expdir}" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --region_keypose_adaptor_weight="${region_keypose_adaptor_weight}" \
    --keypose_condition=${keypose_condition} \
    --save_dir="results/test/${expdir}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=11441 \
    --image_guidance=${image_guidance} \
    --mode=${task}
fi

#---------------------------------------------4_anime_sparse_box-------------------------------------------

four_anime=1
expdir="4anime"

if [ ${four_anime} -eq 1 ]
then
  echo $expdir
  context_prompt='four girls, walking near a garden'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[a <hina1> <hina2>, near a garden]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[61, 0, 512, 160]'

  region2_prompt='[a <mitsuha1> <mitsuha2>, near a garden]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[61, 208, 512, 378]'

  region3_prompt='[a <ai1> <ai2>, near a garden]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[72, 426, 512, 586]'

  region4_prompt='[a <kaori1> <kaori2>, near a garden]'
  region4_neg_prompt="[${context_neg_prompt}]"
  region4='[61, 634, 512, 814]'

  keypose_condition='datasets/validation_spatial_condition/multi-characters/anime_pose/4anime.png'
  keypose_adaptor_weight=0.0
  sketch_condition=''
  sketch_adaptor_weight=1.0
  region_keypose_adaptor_weight="${region1}-1.0|${region2}-1.0|${region3}-1.0|${region4}-1.0"

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}|${region4_prompt}-*-${region4_neg_prompt}-*-${region4}"

  python inference/region_lora_sample.py \
    --pretrained_model="experiments/pretrained_models/anything-v4.0" \
    --combined_model_root="${combined_model_root}/${expdir}" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --region_keypose_adaptor_weight="${region_keypose_adaptor_weight}" \
    --keypose_condition=${keypose_condition} \
    --save_dir="results/test/${expdir}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=16849 \
    --image_guidance="1" \
    --mode=${task}
fi

#---------------------------------------------4_anime_sparse_box_2-------------------------------------------

four_anime_2=1
expdir="4anime_2"

if [ ${four_anime_2} -eq 1 ]
then
  echo $expdir
  context_prompt='two girls, a man and a boy, walking near a mountain'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[a <tezuka1> <tezuka2>, near a mountain]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[20, 30, 512, 210]'

  region2_prompt='[a <ai1> <ai2>, near a mountain]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[61, 300, 512, 460]'

  region3_prompt='[a <kaori1> <kaori2>, near a mountain]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[72, 500, 512, 660]'

  region4_prompt='[a <son1> <son2>, near a mountain]'
  region4_neg_prompt="[${context_neg_prompt}]"
  region4='[9, 740, 512, 920]'

  keypose_condition='datasets/validation_spatial_condition/multi-characters/anime_pose/4anime_2.png'
  keypose_adaptor_weight=0.0
  sketch_condition=''
  sketch_adaptor_weight=1.0
  # region_keypose_adaptor_weight="${region1}-1.0|${region2}-1.0|${region3}-1.0|${region4}-1.0"

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}|${region4_prompt}-*-${region4_neg_prompt}-*-${region4}"

  python inference/region_lora_sample.py \
    --pretrained_model="experiments/pretrained_models/anything-v4.0" \
    --combined_model_root="${combined_model_root}/${expdir}" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --region_keypose_adaptor_weight="${region_keypose_adaptor_weight}" \
    --keypose_condition=${keypose_condition} \
    --save_dir="results/test/${expdir}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=17308 \
    --image_guidance=${image_guidance} \
    --mode=${task}
fi

#---------------------------------------------4_anime_sparse_box_3-------------------------------------------

four_anime_3=1
expdir="4anime_3"

if [ ${four_anime_3} -eq 1 ]
then
  echo $expdir
  context_prompt='three girls, one boy, walking near a forest'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[a <kaori1> <kaori2>, near a forest]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[61, 10, 512, 170]'

  region2_prompt='[a <ai1> <ai2>, near a forest]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[72, 300, 512, 460]'

  region3_prompt='[a <hina1> <hina2>, near a forest]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[61, 520, 512, 680]'

  region4_prompt='[a <son1> <son2>, near a forest]'
  region4_neg_prompt="[${context_neg_prompt}]"
  region4='[9, 740, 512, 920]'

  keypose_condition='datasets/validation_spatial_condition/multi-characters/anime_pose/4anime_3.png'
  keypose_adaptor_weight=0.0
  sketch_condition=''
  sketch_adaptor_weight=1.0
  # region_keypose_adaptor_weight="${region1}-1.0|${region2}-1.0|${region3}-1.0|${region4}-1.0"

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}|${region4_prompt}-*-${region4_neg_prompt}-*-${region4}"

  python inference/region_lora_sample.py \
    --pretrained_model="experiments/pretrained_models/anything-v4.0" \
    --combined_model_root="${combined_model_root}/${expdir}" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --region_keypose_adaptor_weight="${region_keypose_adaptor_weight}" \
    --keypose_condition=${keypose_condition} \
    --save_dir="results/test/${expdir}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=3225 \
    --image_guidance=${image_guidance} \
    --mode=${task}
fi

#---------------------------------------------three_anime_attr_new-------------------------------------------

three_anime_attr_new=1
expdir_3anime="3anime"
if [ ${three_anime_attr_new} -eq 1 ]
then
  echo "three_anime_attr_new"
  context_prompt='two girls and a boy are walking near a lake'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[a <tezuka1> <tezuka2>, in white suit, near a lake]'
  region1_neg_prompt="[${context_neg_prompt}]"
  # region1='[61, 115, 512, 273]'
  region1='[10, 10, 512, 240]'

  region2_prompt='[a <ai1> <ai2>, holding teddy bear, near a lake]'
  region2_neg_prompt="[${context_neg_prompt}]"
  # region2='[19, 292, 512, 512]'
  region2='[60, 290, 512, 472]'

  region3_prompt='[a <kaori1> <kaori2>, wearing a hat, near a lake]'
  region3_neg_prompt="[${context_neg_prompt}]"
  # region3='[48, 519, 512, 706]'
  region3='[30, 519, 512, 720]'

  keypose_condition='datasets/validation_spatial_condition/multi-characters/anime_pose/3anime.png'
  keypose_adaptor_weight=0.0
  sketch_condition=''
  sketch_adaptor_weight=1.0
  # region_keypose_adaptor_weight="${region1}-1.0|${region2}-1.0|${region3}-1.0"

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}"


  python inference/region_lora_sample.py \
    --pretrained_model="experiments/pretrained_models/anything-v4.0" \
    --combined_model_root="${combined_model_root}/${expdir_3anime}" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --region_keypose_adaptor_weight="${region_keypose_adaptor_weight}" \
    --keypose_condition=${keypose_condition} \
    --save_dir="results/test/${expdir_3anime}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=1977 \
    --image_guidance=${image_guidance} \
    --mode=${task}
fi

#---------------------------------------------three_anime_girl-------------------------------------------

three_anime_girl=1
expdir_3girls="3girl"
if [ ${three_anime_girl} -eq 1 ]
then
  echo "three_anime_girl"
  context_prompt='three girls are walking near a lake'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[a <hina1> <hina2>, near a lake]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[61, 115, 512, 273]'

  region2_prompt='[a <mitsuha1> <mitsuha2>, near a lake]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[49, 323, 512, 500]'

  region3_prompt='[a <kaori1> <kaori2>, near a lake]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[53, 519, 512, 715]'

  keypose_condition='datasets/validation_spatial_condition/multi-characters/anime_pose/3girl.png'
  keypose_adaptor_weight=0.0
  sketch_condition=''
  sketch_adaptor_weight=1.0
  region_keypose_adaptor_weight="${region1}-1.0|${region2}-1.0|${region3}-1.0"

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}"

  python inference/region_lora_sample.py \
    --pretrained_model="experiments/pretrained_models/anything-v4.0" \
    --combined_model_root="${combined_model_root}/${expdir_3girls}" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --region_keypose_adaptor_weight="${region_keypose_adaptor_weight}" \
    --keypose_condition=${keypose_condition} \
    --save_dir="results/test/${expdir_3girls}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=15025 \
    --image_guidance=${image_guidance} \
    --mode=${task}
fi

#---------------------------------------------three_anime_girl-------------------------------------------
three_anime_girl_2=1
expdir_3girls_2="3girl2"
if [ ${three_anime_girl_2} -eq 1 ]
then
  echo $expdir_3girls_2
  context_prompt='three girls are walking near a lake'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[a <hina1> <hina2>, near a lake]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[61, 115, 512, 273]'

  # region2_prompt='[a <mitsuha1> <mitsuha2>, near a lake]'
  region2_prompt='[a <ai1> <ai2>, near a lake]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[49, 323, 512, 500]'

  region3_prompt='[a <kaori1> <kaori2>, near a lake]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[53, 519, 512, 715]'

  keypose_condition='datasets/validation_spatial_condition/multi-characters/anime_pose/3girl2.png'
  keypose_adaptor_weight=0.0
  sketch_condition=''
  sketch_adaptor_weight=1.0
  # region_keypose_adaptor_weight="${region1}-1.0|${region2}-1.0|${region3}-1.0"

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}"

  python inference/region_lora_sample.py \
    --pretrained_model="experiments/pretrained_models/anything-v4.0" \
    --combined_model_root="${combined_model_root}/${expdir_3girls_2}"
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --region_keypose_adaptor_weight="${region_keypose_adaptor_weight}" \
    --keypose_condition=${keypose_condition} \
    --save_dir="results/test/${expdir_3girls_2}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=3295 \
    --image_guidance=${image_guidance} \
    --mode=${task}
fi

#---------------------------------------------2_anime_boy-------------------------------------------
two_anime_boy=1
expdir_2boys="2boys"
if [ ${two_anime_boy} -eq 1 ]
then
  echo $expdir_2boys
  context_prompt='two boys are walking near a lake'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[a <tezuka1> <tezuka2>, near a lake]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[20, 300, 512, 500]'

  region2_prompt='[a <son1> <son2>, near a lake]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[9, 550, 512, 750]'

  keypose_condition='datasets/validation_spatial_condition/multi-characters/anime_pose/2boys.png'
  keypose_adaptor_weight=0.0
  sketch_condition=''
  sketch_adaptor_weight=1.0
  region_keypose_adaptor_weight="${region1}-1.0|${region2}-1.0"

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}"

  python inference/region_lora_sample.py \
    --pretrained_model="experiments/pretrained_models/anything-v4.0" \
    --combined_model_root="${combined_model_root}/${expdir_2boys}" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --region_keypose_adaptor_weight="${region_keypose_adaptor_weight}" \
    --keypose_condition=${keypose_condition} \
    --save_dir="results/test/${expdir_2boys}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=6738 \
    --image_guidance="1" \
    --mode=${task}
fi

