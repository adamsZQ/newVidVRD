from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.utils import registry

import utils.get_data as vid_data
import utils.get_relation_list as vid_relation


@registry.register_problem
class Frame2Class(image_utils.Image2ClassProblem):

    relation_path = '../data/first_relation_dict.txt'
    # feature_path = '../data/VidVRD-features/vid_features'

    @property
    def num_channels(self):
        return 3

    @property
    def is_small(self):
        return False

    @property
    def num_classes(self):
        return len(vid_relation.load_relation(self.relation_path))

    @property
    def class_labels(self):
        return vid_relation.load_relation(self.relation_path)

    @property
    def train_shards(self):
        return 10

    def feature_encoders(self, data_dir):
        del data_dir
        return {
            "inputs": vid_data.load_feature(vid_data.load_feature(self.feature_path)),
            "targets": text_encoder.ClassLabelEncoder(self.class_labels)
        }
    def preprocess_example(self, example, mode, unused_hparams):
        image = example["inputs"]
        image.set_shape([_MNIST_IMAGE_SIZE, _MNIST_IMAGE_SIZE, 1])
        if not self._was_reversed:
            image = tf.image.per_image_standardization(image)
        example["inputs"] = image
        return example

    def generator(self, data_dir, tmp_dir, is_training):
        if is_training:
            return mnist_generator(tmp_dir, True, 55000)
        else:
            return mnist_generator(tmp_dir, True, 5000, 55000)
