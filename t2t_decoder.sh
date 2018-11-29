#!/usr/bin/env bash

DECODE_FROM = '../decode_in/sum_pooling_in.txt'
  DECODE_TO = '../decode_out_truth/sum_pooling_out.txt'
  t2t-decoder --t2t_usr_dir=./ --problem=$PROBLEM_NAME--data_dir=DATA_DIR --model=transformer --hparams_set=transformer_base --output_dir=$OUTPUT_DIR --decode_hparams=”beam_size=5,alpha=0.6” --decode_from_file=$DECODE_FROM --decode_to_file=$DECODE_TO