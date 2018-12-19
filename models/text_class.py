from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from tensor2tensor.data_generators import problem, text_problems
from tensor2tensor.utils import registry

sys.path.append("../utils")

import numpy as np

import get_data as vid_data
import get_relation_list as vid_relation


@registry.register_problem
class TextClass(text_problems.Text2ClassProblem):
    PROBLEM_NAME = 'TextClass'

    first_relation_path = '../data/first_relation_dict.txt'
    second_relation_path = '../data/second_relation_dict.txt'
    relation_path = first_relation_path

    # feature_path = '../data/VidVRD-features/vid_features'

    @property
    def is_generate_per_split(self):
        return True

    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 5,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    @property
    def approx_vocab_size(self):
        return 2 ** 10  # 8k vocab suffices for this small dataset.

    @property
    def num_classes(self):
        return len(vid_relation.load_relation(self.relation_path))

    @property
    def vocab_filename(self):
        return self.PROBLEM_NAME + ".vocab.%d" % self.approx_vocab_size

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        del dataset_split

        feature_type = 'train'
        # for each_ins in vid_data.gen_vrd_instance(feature_type):
        #     yield {
        #         "inputs": np.array2string(each_ins.get_my_feature(feature_type).ravel())[1:-1],
        #         "targets": vid_relation.load_relation('first')[each_ins.predicate.split('_')[0]]
        #     }

        ins_list = vid_data.gen_vrd_instance(feature_type)
        for i in range(3):
            yield {
                "inputs": np.array2string(ins_list[i].get_my_feature(feature_type).ravel())[1:-1],
                "label": vid_relation.load_relation('first')[ins_list[i].predicate.split('_')[0]]
            }


if __name__ == '__main__':
    feature_type = 'train'
    each_ins = vid_data.gen_vrd_instance(feature_type)[0]
    print(np.array2string(each_ins.get_my_feature(feature_type).ravel())[1:-1])
    print(vid_relation.load_relation('first')[each_ins.predicate.split('_')[0]])
