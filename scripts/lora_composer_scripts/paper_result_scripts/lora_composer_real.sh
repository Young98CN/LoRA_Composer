combined_model_root="experiments/link_folder"
image_guidance=''

#---------------------------------------------samoke potter_rock---------------------------------------------
potter_rock_new=1
potter_rock_expdir_new='potter+hermione+rock_background'
if [ ${potter_rock_new} -eq 1 ]
then
  echo $potter_rock_expdir_new
  context_prompt='a man and a woman, at<rock1> <rock2>, 4K, high quality, high resolution, best quality'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[<potter1> <potter2>, in hogwarts school uniform, 4K, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[10, 0, 767, 350]'

  region2_prompt='[<hermione1> <hermione2>, in hogwarts school uniform, 4K, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[20, 410, 767, 760]'

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}"
  keypose_condition='datasets/validation_spatial_condition/characters-objects/harry_heminone_scene_pose.png'
  keypose_adaptor_weight=1.0
  region_keypose_adaptor_weight=""

  sketch_condition='datasets/validation_spatial_condition/characters-objects/harry_heminone_scene_sketch.png'
  sketch_adaptor_weight=0.5
  region_sketch_adaptor_weight="${region3}-0.8|${region4}-0.8"

  python inference/region_lora_sample.py \
    --pretrained_model="experiments/pretrained_models/chilloutmix" \
    --combined_model_root="${combined_model_root}/${potter_rock_expdir_new}" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --keypose_condition=${keypose_condition} \
    --save_dir="results/test/${potter_rock_expdir_new}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=1174 \
    --image_guidance=${image_guidance}
fi

#--------------------------------------------potter_pyramid shake hands---------------------------------------------
potter_pyramid_new=0
potter_pyramid_expdir_new='potter+hermione+pyramid_background+shake_hands'
if [ ${potter_pyramid_new} -eq 1 ]
then
  echo $potter_pyramid_expdir_new
  context_prompt='a man and a woman shaking hands, at <pyramid1> <pyramid2>, 4K, high quality, high resolution, best quality'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[<potter1> <potter2> shaking hand, looking at viewer, detail face, cowboy shot, 4K, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[10, 0, 767, 350]'

  region2_prompt='[<hermione1> <hermione2> shaking hand, looking at viewer, detail face, cowboy shot, 4K, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[20, 370, 767, 670]'

  keypose_condition='/data/yangyang/mixofshow/datasets/validation_spatial_condition/multi-characters/real_pose/shake_hand.png'
  keypose_adaptor_weight=0.0
  region_keypose_adaptor_weight="${region1}-1.0|${region2}-1.0"

  sketch_condition=''
  sketch_adaptor_weight=0.0
  region_sketch_adaptor_weight=""
  
  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}"

  python inference/region_lora_sample.py \
    --pretrained_model="experiments/pretrained_models/chilloutmix" \
    --combined_model_root="${combined_model_root}/${potter_pyramid_expdir_new}" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --region_keypose_adaptor_weight="${region_keypose_adaptor_weight}" \
    --keypose_condition=${keypose_condition} \
    --save_dir="results/test/${potter_pyramid_expdir_new}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=6433 \
    --image_guidance=${image_guidance}
fi

#---------------------------------------------2real------------------------------------------
real2=1
real2_expdir='2real'

if [ ${real2} -eq 1 ]
then
  echo $real2_expdir
  context_prompt='a woman and a man, near the castle, 4K, high quality, high resolution, best quality'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[a <potter1> <potter2>, in hogwarts school uniform, near the castle, detail face, cowboy shot, 4K, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[10, 10, 767, 310]'

  region2_prompt='[a <hermione1> <hermione2>, in hogwarts school uniform, near the castle, detail face, cowboy shot, 4K, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[70, 350, 767, 650]'

  keypose_condition='datasets/validation_spatial_condition/multi-characters/real_pose/2real.png'
  keypose_adaptor_weight=0.0
  sketch_condition=''
  sketch_adaptor_weight=0.0
  region_keypose_adaptor_weight="${region1}-1.0|${region2}-1.0"

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}"

  python inference/region_lora_sample.py \
    --pretrained_model="experiments/pretrained_models/chilloutmix" \
    --combined_model_root="${combined_model_root}/${real2_expdir}" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --region_keypose_adaptor_weight="${region_keypose_adaptor_weight}" \
    --keypose_condition=${keypose_condition} \
    --save_dir="results/test/${real2_expdir}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=18067 \
    --image_guidance=${image_guidance}
fi

#---------------------------------------------2real_2------------------------------------------
real2_2=1
real2_2_expdir='2real2'

if [ ${real2_2} -eq 1 ]
then
  echo $real2_2_expdir
  context_prompt='two men, in front of eiffel tower, 4K, high quality, high resolution, best quality'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[a <hinton1> <hinton2>, in front of eiffel tower, detail face, bust, 4K, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[10, 10, 767, 310]'

  region2_prompt='[a <lecun1> <lecun2>, in front of eiffel tower, detail face, bust, 4K, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[70, 600, 767, 900]'

  keypose_condition='datasets/validation_spatial_condition/multi-characters/real_pose/2real_2.png'
  keypose_adaptor_weight=0.0
  sketch_condition=''
  sketch_adaptor_weight=0.0
  region_keypose_adaptor_weight="${region1}-1.0|${region2}-1.0"

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}"

  python inference/region_lora_sample.py \
    --pretrained_model="experiments/pretrained_models/chilloutmix" \
    --combined_model_root="${combined_model_root}/${real2_2_expdir}" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --keypose_condition=${keypose_condition} \
    --save_dir="results/test/${real2_2_expdir}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=9023 \
    --image_guidance="1"
fi

#---------------------------------------------2real_3------------------------------------------
real2_3=1
real2_3_expdir='2real3'

if [ ${real2_3} -eq 1 ]
then
  echo $real2_3_expdir
  context_prompt='two men, simple background, 4K, high quality, high resolution, best quality'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[a <bengio1> <bengio2>, simple background, detail face, bust, 4K, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[10, 10, 767, 310]'

  region2_prompt='[a <potter1> <potter2>, simple background, detail face, bust, 4K, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[70, 600, 767, 900]'

  keypose_condition='datasets/validation_spatial_condition/multi-characters/real_pose/2real_3.png'
  keypose_adaptor_weight=0.0
  sketch_condition=''
  sketch_adaptor_weight=0.0
  region_keypose_adaptor_weight="${region1}-1.0|${region2}-1.0"

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}"

  python inference/region_lora_sample.py \
    --pretrained_model="experiments/pretrained_models/chilloutmix" \
    --combined_model_root="${combined_model_root}/${real2_3_expdir}" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --keypose_condition=${keypose_condition} \
    --save_dir="results/test/${real2_3_expdir}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=9023 \
    --image_guidance=${image_guidance}
fi

#---------------------------------------------sample dog-------------------------------------------
sample_dog=0
sample_dog_expdir='2dog+1cat'

if [ ${sample_dog} -eq 1 ]
then
  echo $sample_dog_expdir
  context_prompt='two dogs and a cat, in the forest, autumn leaves, trees, animal photography, 4K, high quality, high resolution, best quality'
  context_neg_prompt='dark, low quality, low resolution'

  region1_prompt='[a <dogB1> <dogB2>, in the forest, animal photography, 4K, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[150, 0, 700, 350]'

  region2_prompt='[a <catA1> <catA2>, in the forest, animal photography, 4K, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[250, 410, 700, 660]'

  region3_prompt='[a <dogA1> <dogA2>, in the forest, animal photography, 4K, high quality, high resolution, best quality]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[150, 723, 700, 1023]'

  keypose_condition=''
  keypose_adaptor_weight=0.0

  sketch_condition='datasets/validation_spatial_condition/multi-characters/real_pose/2dog1cat.jpg'
  sketch_adaptor_weight=0.0
  region_sketch_adaptor_weight="${region1}-0.8|${region2}-0.8|${region3}-0.8"

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}"

  python inference/region_lora_sample.py \
    --pretrained_model="experiments/pretrained_models/chilloutmix" \
    --combined_model_root="${combined_model_root}/${sample_dog_expdir}" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --region_sketch_adaptor_weight="${region_sketch_adaptor_weight}"\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --keypose_condition=${keypose_condition} \
    --save_dir="results/test/${sample_dog_expdir}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=2343 \
    --image_guidance=${image_guidance}
fi


#---------------------------------------------5real---------------------------------------------
real5=0
real5_expdir='5real'
if [ ${real5} -eq 1 ]
then
  echo $real5_expdir
  context_prompt='a man and a woman, a cat and a dog, at <rock1> <rock2>, 4K, high quality, high resolution, best quality'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[<potter1> <potter2>, looking at viewer, detail face, bust, 4K, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[10, 0, 767, 250]'

  region2_prompt='[<dogA1> <dogA2>, animal photography, 4K, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[375, 290, 767, 490]'

  region3_prompt='[<catA1> <catA2>, animal photography, 4K, high quality, high resolution, best quality]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[375, 530, 765, 730]'

  region4_prompt='[<hermione1> <hermione2>, looking at viewer, detail face, bust, 4K, high quality, high resolution, best quality]'
  region4_neg_prompt="[${context_neg_prompt}]"
  region4='[20, 770, 767, 1020]'

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}|${region4_prompt}-*-${region4_neg_prompt}-*-${region4}"
  keypose_condition='datasets/validation_spatial_condition/multi-characters/real_pose/5real.png'
  keypose_adaptor_weight=0.0
  region_keypose_adaptor_weight="${region1}-1.0|${region4}-1.0"

  sketch_condition='datasets/validation_spatial_condition/multi-characters/real_pose/5real_edge.jpg'
  sketch_adaptor_weight=0.0
  region_sketch_adaptor_weight="${region2}-0.8|${region3}-0.8"
  
  python inference/region_lora_sample.py \
    --pretrained_model="experiments/pretrained_models/chilloutmix" \
    --combined_model_root="${combined_model_root}/${real5_expdir}" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --region_sketch_adaptor_weight="${region_sketch_adaptor_weight}" \
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --region_keypose_adaptor_weight="${region_keypose_adaptor_weight}" \
    --keypose_condition=${keypose_condition} \
    --save_dir="results/test/${real5_expdir}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=15958 \
    --image_guidance="1"
fi

#---------------------------------------------5real2---------------------------------------------
real5_2=0
real5_2_expdir='5real2'
if [ ${real5_2} -eq 1 ]
then
  echo $real5_2_expdir
  context_prompt='two men, a cat and a dog, at <rock1> <rock2>, 4K, high quality, high resolution, best quality'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[<bengio1> <bengio2>, looking at viewer, detail face, bust, 4K, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[10, 0, 767, 250]'

  region2_prompt='[<dogA1> <dogA2>, animal photography, 4K, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[375, 290, 767, 490]'

  region3_prompt='[<catA1> <catA2>, animal photography, 4K, high quality, high resolution, best quality]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[375, 530, 765, 730]'

  region4_prompt='[<hinton1> <hinton2>, looking at viewer, detail face, bust, 4K, high quality, high resolution, best quality]'
  region4_neg_prompt="[${context_neg_prompt}]"
  region4='[20, 770, 767, 1020]'

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}|${region4_prompt}-*-${region4_neg_prompt}-*-${region4}"
  keypose_condition='datasets/validation_spatial_condition/multi-characters/real_pose/5real_2.png'
  keypose_adaptor_weight=0.0
  region_keypose_adaptor_weight="${region1}-1.0|${region4}-1.0"

  sketch_condition='datasets/validation_spatial_condition/multi-characters/real_pose/5real_edge.jpg'
  sketch_adaptor_weight=0.0
  region_sketch_adaptor_weight="${region2}-0.8|${region3}-0.8"

  python inference/region_lora_sample.py \
    --pretrained_model="experiments/pretrained_models/chilloutmix" \
    --combined_model_root="${combined_model_root}/${real5_2_expdir}" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight} \
    --region_sketch_adaptor_weight="${region_sketch_adaptor_weight}" \
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --region_keypose_adaptor_weight="${region_keypose_adaptor_weight}" \
    --keypose_condition=${keypose_condition} \
    --save_dir="results/test/${real5_2_expdir}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=10966 \
    --image_guidance=${image_guidance}
fi

#---------------------------------------------5real3---------------------------------------------
real5_3=0
real5_3_expdir='5real3'
if [ ${real5_3} -eq 1 ]
then
  echo $real5_3_expdir
  context_prompt='two men, a cat and a dog, at <rock1> <rock2>, 4K, high quality, high resolution, best quality'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[<hinton1> <hinton2>, looking at viewer, detail face, bust, 4K, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[10, 0, 767, 250]'

  region2_prompt='[<dogA1> <dogA2>, animal photography, 4K, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[375, 290, 767, 490]'

  region3_prompt='[<catA1> <catA2>, animal photography, 4K, high quality, high resolution, best quality]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[375, 530, 765, 730]'

  region4_prompt='[<potter1> <potter2>, looking at viewer, detail face, bust, 4K, high quality, high resolution, best quality]'
  region4_neg_prompt="[${context_neg_prompt}]"
  region4='[20, 770, 767, 1020]'

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}|${region4_prompt}-*-${region4_neg_prompt}-*-${region4}"
  keypose_condition='datasets/validation_spatial_condition/multi-characters/real_pose/5real_3.png'
  keypose_adaptor_weight=0.0
  region_keypose_adaptor_weight="${region1}-1.0|${region4}-1.0"

  sketch_condition='datasets/validation_spatial_condition/multi-characters/real_pose/5real_edge.jpg'
  sketch_adaptor_weight=0.0
  region_sketch_adaptor_weight="${region2}-0.8|${region3}-0.8"
  echo $image_guidance
  python inference/region_lora_sample.py \
    --pretrained_model="experiments/pretrained_models/chilloutmix" \
    --combined_model_root="${combined_model_root}/${real5_3_expdir}" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --region_sketch_adaptor_weight="${region_sketch_adaptor_weight}" \
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --region_keypose_adaptor_weight="${region_keypose_adaptor_weight}" \
    --keypose_condition=${keypose_condition} \
    --save_dir="results/test/${real5_3_expdir}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=19569 \
    --image_guidance=${image_guidance}
fi

#---------------------------------------------sample lecun-------------------------------------------
sample_lecun=0
sample_lecun_expdir='3real'
if [ ${sample_lecun} -eq 1 ]
then
  echo $sample_lecun_expdir
  context_prompt='three men, standing near a lake, looking at viewer, detail face, 4K, high quality, high resolution, best quality'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[<bengio1> <bengio2>, standing near a lake, looking at viewer, detail face, 4K, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[6, 51, 767, 293]'

  region2_prompt='[a <lecun1> <lecun2>, standing near a lake, looking at viewer, detail face, 4K, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[1, 350, 767, 618]'

  region3_prompt='[a <hinton1> <hinton2>, standing near a lake, looking at viewer, detail face, 4K, high quality, high resolution, best quality]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[3, 657, 767, 923]'

  keypose_condition='datasets/validation_spatial_condition/multi-characters/real_pose/3real.png'
  keypose_adaptor_weight=0.0
  region_keypose_adaptor_weight="${region1}-1.0|${region2}-1.0|${region3}-1.0"

  sketch_condition=''
  sketch_adaptor_weight=0.0

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}"

  python inference/region_lora_sample.py \
    --pretrained_model="experiments/pretrained_models/chilloutmix" \
    --combined_model_root="${combined_model_root}/${sample_lecun_expdir}" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --region_keypose_adaptor_weight="${region_keypose_adaptor_weight}" \
    --keypose_condition=${keypose_condition} \
    --save_dir="results/test/${sample_lecun_expdir}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=3880 \
    --image_guidance=${image_guidance}
fi


#---------------------------------------------4real-------------------------------------------
real4=1
real4_expdir='4real'
if [ ${real4} -eq 1 ]
then
  echo $real4_expdir
  context_prompt='a dog, two men and a woman, near the lake, 4K, high quality, high resolution, best quality'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[<hermione1> <hermione2>, near the lake, looking at viewer, detail face, bust, 4K, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[10, 0, 767, 230]'

  region2_prompt='[<dogA1> <dogA2>, near the lake, looking at viewer, detail face, bust, 4K, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[400, 250, 767, 420]'

  region3_prompt='[<hinton1> <hinton2>, near the lake, looking at viewer, detail face, bust, 4K, high quality, high resolution, best quality]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[80, 460, 767, 710]'

  region4_prompt='[<bengio1> <bengio2>, near the lake, looking at viewer, detail face, bust, 4K, high quality, high resolution, best quality]'
  region4_neg_prompt="[${context_neg_prompt}]"
  region4='[10, 770, 767, 1023]'

  keypose_condition='datasets/validation_spatial_condition/multi-characters/real_pose/4real.png'
  keypose_adaptor_weight=0.0
  region_keypose_adaptor_weight="${region1}-1.0|${region3}-1.0|${region4}-1.0"

  sketch_condition='/data/yangyang/mixofshow/datasets/validation_spatial_condition/multi-characters/real_pose/4real_edge.jpg'
  sketch_adaptor_weight=0.0
  region_sketch_adaptor_weight="${region2}-0.8"

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}|${region4_prompt}-*-${region4_neg_prompt}-*-${region4}"

  python inference/region_lora_sample.py \
    --pretrained_model="experiments/pretrained_models/chilloutmix" \
    --combined_model_root="${combined_model_root}/${real4_expdir}" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --region_sketch_adaptor_weight="${region_sketch_adaptor_weight}"\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --region_keypose_adaptor_weight="${region_keypose_adaptor_weight}" \
    --keypose_condition=${keypose_condition} \
    --save_dir="results/test/${real4_expdir}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=15946 \
    --image_guidance=${image_guidance}
fi

#---------------------------------------------4real_2-------------------------------------------
real4_2=0
real4_2_expdir='4real2'
if [ ${real4_2} -eq 1 ]
then
  echo $real4_2_expdir
  context_prompt='a dog and three men, near the lake, 4K, high quality, high resolution, best quality'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[<potter1> <potter2>, near the lake, looking at viewer, detail face, bust, 4K, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[10, 0, 767, 230]'

  region2_prompt='[<dogA1> <dogA2>, near the lake, looking at viewer, detail face, bust, 4K, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[400, 250, 767, 420]'

  region3_prompt='[<hinton1> <hinton2>, near the lake, looking at viewer, detail face, bust, 4K, high quality, high resolution, best quality]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[80, 460, 767, 710]'

  region4_prompt='[<lecun1> <lecun2>, near the lake, looking at viewer, detail face, bust, 4K, high quality, high resolution, best quality]'
  region4_neg_prompt="[${context_neg_prompt}]"
  region4='[10, 770, 767, 1023]'

  keypose_condition='datasets/validation_spatial_condition/multi-characters/real_pose/4real_2.png'
  keypose_adaptor_weight=0.0
  region_keypose_adaptor_weight="${region1}-1.0|${region3}-1.0|${region4}-1.0"

  sketch_condition='/data/yangyang/mixofshow/datasets/validation_spatial_condition/multi-characters/real_pose/4real_2_edge.jpg'
  sketch_adaptor_weight=0.0
  region_sketch_adaptor_weight="${region2}-0.8"

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}|${region4_prompt}-*-${region4_neg_prompt}-*-${region4}"

  python inference/region_lora_sample.py \
    --pretrained_model="experiments/pretrained_models/chilloutmix" \
    --combined_model_root="${combined_model_root}/${real4_2_expdir}" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --region_sketch_adaptor_weight="${region_sketch_adaptor_weight}"\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --region_keypose_adaptor_weight="${region_keypose_adaptor_weight}" \
    --keypose_condition=${keypose_condition} \
    --save_dir="results/test/${real4_2_expdir}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=3501 \
    --image_guidance="1"
fi

#---------------------------------------------4real_3-------------------------------------------
real4_3=0
real4_3_expdir='4real3'
if [ ${real4_3} -eq 1 ]
then
  echo $real4_3_expdir
  context_prompt='a woman, two men and a cat, near the lake, 4K, high quality, high resolution, best quality'
  context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

  region1_prompt='[<hermione1> <hermione2>, near the lake, looking at viewer, detail face, bust, 4K, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[10, 0, 767, 230]'

  region2_prompt='[<catA1> <catA2>, near the lake, looking at viewer, detail face, bust, 4K, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[400, 250, 767, 420]'

  region3_prompt='[<hinton1> <hinton2>, near the lake, looking at viewer, detail face, bust, 4K, high quality, high resolution, best quality]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[80, 460, 767, 710]'

  region4_prompt='[<potter1> <potter2>, near the lake, looking at viewer, detail face, bust, 4K, high quality, high resolution, best quality]'
  region4_neg_prompt="[${context_neg_prompt}]"
  region4='[10, 770, 767, 1023]'

  keypose_condition='datasets/validation_spatial_condition/multi-characters/real_pose/4real_3.png'
  keypose_adaptor_weight=0.0
  region_keypose_adaptor_weight="${region1}-1.0|${region3}-1.0|${region4}-1.0"

  sketch_condition='/data/yangyang/mixofshow/datasets/validation_spatial_condition/multi-characters/real_pose/4real_3_edge.jpg'
  sketch_adaptor_weight=0.0
  region_sketch_adaptor_weight="${region2}-0.8"

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}|${region4_prompt}-*-${region4_neg_prompt}-*-${region4}"

  python inference/region_lora_sample.py \
    --pretrained_model="experiments/pretrained_models/chilloutmix" \
    --combined_model_root="${combined_model_root}/${real4_3_expdir}" \
    --sketch_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_sketch_sd14v1.pth" \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --region_sketch_adaptor_weight="${region_sketch_adaptor_weight}"\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_model="experiments/pretrained_models/t2i_adapter/t2iadapter_openpose_sd14v1.pth" \
    --keypose_adaptor_weight="${keypose_adaptor_weight}"\
    --region_keypose_adaptor_weight="${region_keypose_adaptor_weight}" \
    --keypose_condition=${keypose_condition} \
    --save_dir="results/test/${real4_3_expdir}" \
    --pipeline_type="adaptor_pplus" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=8875 \
    --image_guidance="1"
fi