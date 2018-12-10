# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import image_utils
from tensor2tensor.utils import registry

from utils.get_relation_list import gen_relations
"""
image generation with transformer (attention).
encoder: [Self-Attention, Feed-forward] x n
decoder: [Self-Attention, Source-Target-Attention, Feed-forward] x n
"""

@registry.register_problem
class MyImgTrans(image_utils.Image2ClassProblem):

    @property
    def is_small(self):
        return False

    @property
    def num_classes(self):
        return



# PROBLEM_NAME='attention_gru_feature'
# DATA_DIR='../train_data_atte_feature'
# OUTPUT_DIR='../output_atte_feature'
# t2t-datagen --t2t_usr_dir=. --data_dir=$DATA_DIR --tmp_dir=../tmp_data --problem=$PROBLEM_NAME
# t2t-trainer --t2t_usr_dir=. --data_dir=$DATA_DIR --problem=$PROBLEM_NAME --model=transformer --hparams_set=transformer_base --output_dir=$OUTPUT_DIR
