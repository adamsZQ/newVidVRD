#!/usr/bin/env bash

USER_DIR='.'
OUT_DIR='../data/VidVRD-t2t-data'
TEP_DIR='../data/tmp_data'
PROBLEM_NAME=$1
#PROBLEM_NAME='text_class'
#PROBLEM_NAME='frame_class'
#PROBLEM_NAME='video_class'
#PROBLEM_NAME='frame_text'

MODEL_NAME=$2
DATA_DIR='../data/t2t-data'

echo "You are going to train $PROBLEM_NAME problem using $MODEL_NAME right now!"
echo "The encoded data is $DATA_DIR"
echo "The data will be kept in $OUT_DIR"

t2t-trainer \
    --t2t_usr_dir=. \
    --data_dir=$DATA_DIR \
    --problem=$PROBLEM_NAME \
    --model=$MODEL_NAME \
    --hparams_set=$HPARAMS_SET \
    --output_dir=$OUTPUT_DIR