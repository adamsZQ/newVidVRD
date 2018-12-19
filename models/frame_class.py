from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import problem, image_utils
from tensor2tensor.utils import registry

import sys
sys.path.append("../utils")

import get_data as vid_data
import get_relation_list as vid_relation


@registry.register_problem
class FrameClass(image_utils.Image2ClassProblem):

    first_relation_path = '../data/first_relation_dict.txt'
    second_relation_path = '../data/second_relation_dict.txt'
    relation_path = first_relation_path
    # feature_path = '../data/VidVRD-features/vid_features'

    classes_num = len(vid_relation.load_relation(relation_path))

    @property
    def num_generate_tasks(self):
        return self.classes_num

    def prepare_to_generate(self, data_dir, tmp_dir):
        pass

    def generator(self, data_dir, tmp_dir, is_training):
        pass

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
            return {
                "inputs": each_ins.get_my_feature(feature_type),
                "targets": vid_relation.load_relation('first')[each_ins.predicate.split('_')[0]]
            }


if __name__ == '__main__':
    feature_type = 'train'
    each_ins = vid_data.gen_vrd_instance(feature_type)[0]
    print(each_ins.get_my_feature(feature_type))
    print(vid_relation.load_relation('first')[each_ins.predicate.split('_')[0]])
