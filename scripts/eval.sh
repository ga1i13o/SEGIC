NGPUS=${1:-1}
exp_name=${2:-'OUTPUT/eval_lvis_0_ck2'}

echo $NGPUS $encoder_model $exp_name
#echo ${@:4}
# lvis
python eval.py --output $exp_name \
    --batch_size_train 16 --input_keys sem_corr point --eval_keys sem_corr --noised_inst  --use_dual_aug --use_simm_prompt --open_ft --find_unused_params  \
     --diff_text_prompt_ratio 0.75  --use_inst_train --reverse_context --use_cross_inst_prompt \
    --encoder_model dinov2  --shots 1 --dinov2_model l \
    --use_inst_proj --eval --restore-model segic_lvis_fold0/epoch_2.pth --eval_datasets lvis_0
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
#     --encoder_model $encoder_model --sem_datasets pascal_part_0 --samples_per_epoch 20 --eval_datasets pascal_part_0
