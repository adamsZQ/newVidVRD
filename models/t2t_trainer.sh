#!/usr/bin/env bash


t2t-trainer \
    --t2t_usr_dir=. \
    --data_dir=$DATA_DIR \
    --problem=$PROBLEM_NAME \
    --model=$MODEL_NAME \
    --hparams_set=$HPARAMS_SET \
    --output_dir=$OUTPUT_DIR