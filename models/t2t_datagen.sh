#!/usr/bin/env bash

t2t-datagen \
    --t2t_usr_dir=. \
    --data_dir=$OUT_DIR \
    --tmp_dir=../tmp_data \
    --problem=$PROBLEM_NAME
