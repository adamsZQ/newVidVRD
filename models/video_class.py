from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import problem, video_utils
from tensor2tensor.utils import registry

import sys
sys.path.append("../utils")

import numpy as np
import get_data as vid_data
import get_relation_list as vid_relation


@registry.register_problem
class VideoClass(video_utils.Video2ClassProblem):

    @property
    def image_size(self):
        pass

    # feature_path = '../data/VidVRD-features/vid_features'

    classes_num = len(vid_relation.load_relation('first'))

    @property
    def num_generate_tasks(self):
        return self.classes_num

    def prepare_to_generate(self, data_dir, tmp_dir):
        pass

    def generator(self, data_dir, tmp_dir, is_training):
        return self.vidvrd_generator()

    @property
    def num_channels(self):
        return 3

    @property
    def is_small(self):
        return False

    @property
    def num_classes(self):
        return self.classes_num

    @property
    def class_labels(self):
        return vid_relation.load_relation(self.relation_path)

    @property
    def train_shards(self):
        return self.classes_num

    def feature_encoders(self, data_dir):
        del data_dir
        feature_type = 'train'
        for each_ins in vid_data.gen_vrd_instance(feature_type):
            yield {
                "inputs": each_ins.get_my_feature(feature_type),
                "targets": vid_relation.load_relation('first')[each_ins.predicate.split('_')[0]]
            }

    def vidvrd_generator(self):
        feature_type = 'train'

        # for each_ins in vid_data.gen_vrd_instance(feature_type):
        #     features = {"image/encoded": np.array2string(each_ins.get_my_feature(feature_type).ravel()),
        #                 "image/format": ["png"],
        #                 "image/height": [each_ins.height],
        #                 "image/width": [each_ins.width]}
        #     yield features

        ins_list = vid_data.gen_vrd_instance(feature_type)
        for i in range(3):
            features = {"image/encoded": np.array2string(ins_list[i].get_my_feature(feature_type).ravel()),
                        "image/format": ["png"],
                        "image/height": [ins_list[i].height],
                        "image/width": [ins_list[i].width]}
            yield features
