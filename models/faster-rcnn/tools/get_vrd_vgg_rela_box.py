from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from model.config import cfg
from model.test import test_net_vg
from nets.vgg16 import vgg16

base_path = '/home/david/PycharmProjects/newVidVRD/models/tf-faster-rcnn/output/vgg16/vrd/'

output_dir = base_path + 'vrd_dete_pred/dete_pred_vg.npz'
save_path = base_path + 'vrd_dete_pred/vrd_detected_box.npz'
model_path = base_path + 'default/res101_faster_rcnn_iter_80000.ckpt'

num_classes = 101
vg_roidb = np.load('vrd_roidb.npz')
roidb_temp = vg_roidb['roidb']
roidb = roidb_temp[()]
train_roidb = roidb['train_roidb']
test_roidb = roidb['test_roidb']
N_train = len(train_roidb)
N_temp = np.int32(N_train / 2)
train_roidb_temp = train_roidb[0:N_temp]
net = vgg16()
net.create_architecture("TEST", num_classes, tag='default',
                        anchor_scales=cfg.ANCHOR_SCALES,
                        anchor_ratios=cfg.ANCHOR_RATIOS)
saver = tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    saver.restore(sess, model_path)
    train_detected_box = test_net_vg(sess, net, train_roidb_temp, output_dir, num_classes, max_per_image=300,
                                     thresh=0.05)
    test_detected_box = test_net_vg(sess, net, test_roidb, output_dir, num_classes, max_per_image=300, thresh=0.05)
vrd_detected_box = {'train_detected_box': train_detected_box,
                    'test_detected_box': test_detected_box}
np.savez(save_path, vrd_detected_box=vrd_detected_box)
