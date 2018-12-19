from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import problem, image_utils
from tensor2tensor.utils import registry

import sys
sys.path.append("../utils")

import numpy as np
import get_data as vid_data
import get_relation_list as vid_relation


@registry.register_problem
class FrameText(image_utils.Image2TextProblem):
    @property
    def is_character_level(self):
        return True

    @property
    def target_space_id(self):
        return problem.SpaceID.EN_CHR

    @property
    def train_shards(self):
        return 100

    @property
    def dev_shards(self):
        return 10

    def preprocess_example(self, example, mode, _):
        pass

    def generator(self, data_dir, tmp_dir, is_training):
        return self.vidvrd_generator()

    def vidvrd_generator(self):
        feature_type = 'train'
        # for each_ins in vid_data.gen_vrd_instance(feature_type):
        #     yield {
        #         "image/encoded": np.array2string(each_ins.get_my_feature(feature_type).ravel()),
        #         "image/format": ["jpeg"],
        #         "image/class/label": [vid_relation.load_relation('first')[each_ins.predicate.split('_')[0]]],
        #         "image/height": [each_ins.height],
        #         "image/width": [each_ins.width]
        #     }

        ins_list = vid_data.gen_vrd_instance(feature_type)
        for i in range(3):
            yield {
                "image/encoded": np.array2string(ins_list[i].get_my_feature(feature_type).ravel()),
                "image/format": ["jpeg"],
                "image/class/label": [vid_relation.load_relation('first')[ins_list[i].predicate.split('_')[0]]],
                "image/height": [ins_list[i].height],
                "image/width": [ins_list[i].width]
            }
