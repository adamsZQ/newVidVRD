#!/usr/bin/env bash

USER_DIR='.'
OUT_DIR='../data/VidVRD-t2t-data'
TEP_DIR='../data/tmp_data'
#PROBLEM_NAME='text_class'
#PROBLEM_NAME='frame_class'
PROBLEM_NAME='video_class'
#PROBLEM_NAME='frame_text'

t2t-datagen \
    --t2t_usr_dir=. \
    --data_dir=$OUT_DIR \
    --tmp_dir=../tmp_data \
    --problem=$PROBLEM_NAME
