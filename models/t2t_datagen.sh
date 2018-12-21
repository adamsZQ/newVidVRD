#!/usr/bin/env bash

USER_DIR='.'
OUT_DIR='../data/VidVRD-t2t-data'
TEP_DIR='../data/tmp_data'
PROBLEM_NAME=$1
#PROBLEM_NAME='text_class'
#PROBLEM_NAME='frame_class'
#PROBLEM_NAME='video_class'
#PROBLEM_NAME='frame_text'

echo "You are going to generate $PROBLEM_NAME data right now!"
echo "The data will be kept in $OUT_DIR"

t2t-datagen \
    --t2t_usr_dir=. \
    --data_dir=$OUT_DIR \
    --tmp_dir=../tmp_data \
    --problem=$PROBLEM_NAME
