#!/bin/bash

source activate unicorrn

export MODEL_CONFIG_PATH="/your_project_path/configs/models/unicorrn_large_stage2.yml"
export TRAINER_CONFIG_PATH="/your_project_path/configs/trainers/trainer_large_scale_finetune_all_tasks_joint_stage_2.yml"
export CKPT_PATH="/your_output_directory/model_last.pth"
export OUTPUT_DIR="/your_output_directory/"

accelerate launch --num_processes 4 \
    --multi_gpu \
    --main_process_port 12345 \
    --num_machines 1 \
    /your_project_path/train.py \
    --model_config $MODEL_CONFIG_PATH \
    --trainer_config $TRAINER_CONFIG_PATH \
    --resume_ckpt_path $CKPT_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size 4 \
    --accum_iter 1 \
    --set_static_graph
