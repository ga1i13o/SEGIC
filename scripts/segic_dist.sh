NGPUS=${1:-2}
encoder_model=${2:-'dinov2'}
exp_name=${3:-'OUTPUT/segic_lvis_0_noinstproj'}

echo $NGPUS $encoder_model $exp_name
#echo ${@:4}
# lvis
python -m torch.distributed.launch --master_port 12345 --nproc_per_node=$NGPUS train.py --output $exp_name \
    --batch_size_train 16 --input_keys sem_corr point --eval_keys sem_corr --noised_inst  --use_dual_aug --use_simm_prompt --open_ft --find_unused_params  \
     --diff_text_prompt_ratio 0.75  --use_inst_train --reverse_context --learning_rate 0.0001 --use_cross_inst_prompt \
    --encoder_model $encoder_model --sem_datasets lvis_0 --samples_per_epoch 20000 --eval_datasets lvis_0 #--auto_resume ${@:4}
# --use_inst_proj

# original command
# python -m torch.distributed.launch --master_port 12345 --nproc_per_node=$NGPUS train.py --output $exp_name \
#     --input_keys sem_corr point --eval_keys sem_corr --noised_inst  --use_dual_aug --use_simm_prompt --open_ft --find_unused_params --use_dift  \
#     --use_inst_proj --diff_text_prompt_ratio 0.75  --use_inst_train --reverse_context --learning_rate 0.0001 --use_cross_inst_prompt \
#     --encoder_model $encoder_model --inst_datasets coco lvis --sem_datasets coco ade20k --samples_per_epoch 80000 #--auto_resume ${@:4}

# --eval --restore-model /your/ckpt/path --eval_datasets fss

# coco
# python -m torch.distributed.launch --master_port 12345 --nproc_per_node=$NGPUS train.py --output $exp_name \
#     --input_keys sem_corr point --eval_keys sem_corr --noised_inst  --use_dual_aug --use_simm_prompt --open_ft --find_unused_params  \
#     --use_inst_proj --diff_text_prompt_ratio 0.75  --use_inst_train --reverse_context --learning_rate 0.0001 --use_cross_inst_prompt \
#     --encoder_model $encoder_model --sem_datasets coco_0 --samples_per_epoch 80000 --eval_datasets coco_0 #--auto_resume ${@:4}


# paco_part
# python -m torch.distributed.launch --master_port 12445 --nproc_per_node=$NGPUS train.py --output $exp_name \
#     --input_keys sem_corr point --eval_keys sem_corr --noised_inst  --use_dual_aug --use_simm_prompt --open_ft --find_unused_params  \
#     --use_inst_proj --diff_text_prompt_ratio 0.75  --use_inst_train --reverse_context --learning_rate 0.0001 --use_cross_inst_prompt \
#     --encoder_model $encoder_model --sem_datasets lvis_0 --samples_per_epoch 80000 --eval_datasets lvis_0 #--auto_resume ${@:4}

# paco_part
python -m torch.distributed.launch --master_port 12445 --nproc_per_node=$NGPUS train.py \
    --output $exp_name --batch_size_train 16 \
    --input_keys sem_corr point --eval_keys sem_corr --noised_inst  --use_dual_aug --use_simm_prompt --open_ft --find_unused_params  \
    --use_inst_proj --diff_text_prompt_ratio 0.75  --use_inst_train --reverse_context --learning_rate 0.0001 --use_cross_inst_prompt \
    --encoder_model $encoder_model --sem_datasets pascal_part_0 --samples_per_epoch 200 --eval_datasets pascal_part_0

#     --encoder_model $encoder_model --sem_datasets pascal_part_0 --samples_per_epoch 20 --eval_datasets pascal_part_0
