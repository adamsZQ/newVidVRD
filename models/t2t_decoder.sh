#!/usr/bin/env bash

DECODE_FROM='../decode_in/sum_pooling_in.txt'
DECODE_TO='../decode_out_truth/sum_pooling_out.txt'

DECODE_HPARAMS="beam_size=5,alpha=0.6"

t2t-decoder \
    --t2t_usr_dir=. \
    --problem=$PROBLEM_NAME \
    --data_dir=$DATA_DIR \
    --model=$MODEL_NAME \
    --hparams_set=$HPARAMS_SET \
    --output_dir=$OUTPUT_DIR \
    --decode_hparams=$DECODE_HPARAMS \
    --decode_from_file=$DECODE_FROM \
    --decode_to_file=$DECODE_TO
