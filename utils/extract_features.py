#!/usr/bin/env python3
"""
extract_features.py

Script to extract CNN features from video frames.
"""

from __future__ import print_function

import argparse
import os
import sys

from moviepy.editor import VideoFileClip
import numpy as np
import scipy.misc
from tqdm import tqdm


def crop_center(im):
    """
    Crops the center out of an image.

    Args:
        im (numpy.ndarray): Input image to crop.
    Returns:
        numpy.ndarray, the cropped image.
    """

    h, w = im.shape[0], im.shape[1]

    if h < w:
        return im[0:h, int((w - h) / 2):int((w - h) / 2) + h, :]
    else:
        return im[int((h - w) / 2):int((h - w) / 2) + w, 0:w, :]


def extract_split_features(input_vid, output_dir, begin_fid, end_fid, model_type='inceptionv3', batch_size=32):
    name, _ = os.path.splitext(input_vid)
    name = name.split('/')[-1]
    output_dir = os.path.join(output_dir, name)  # RGB features
    # motion_dir = os.path.join(output_dir, 'motion') # Spatiotemporal features
    # opflow_dir = os.path.join(output_dir, 'opflow') # Optical flow features

    for directory in [output_dir]:  # , motion_dir, opflow_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    if model_type.lower() == 'inceptionv3':
        from keras.applications import InceptionV3
        model = InceptionV3(include_top=True, weights='imagenet')
    elif model_type.lower() == 'xception':
        from keras.applications import Xception
        model = Xception(include_top=True, weights='imagenet')
    elif model_type.lower() == 'resnet50':
        from keras.applications import ResNet50
        model = ResNet50(include_top=True, weights='imagenet')
    elif model_type.lower() == 'vgg16':
        from keras.applications import VGG16
        model = VGG16(include_top=True, weights='imagenet')
    elif model_type.lower() == 'vgg19':
        from keras.applications import VGG19
        model = VGG19(include_top=True, weights='imagenet')
    else:
        sys.stderr.write("'%s' is not a valid ImageNet model.\n" % model_type)
        sys.exit(1)

    if model_type.lower() == 'inceptionv3' or model_type.lower() == 'xception':
        shape = (299, 299)

    # Get outputs of model from layer just before softmax predictions

    from keras.models import Model
    model = Model(model.inputs, output=model.layers[-2].output)

    clip = VideoFileClip(input_vid)

    frames = [scipy.misc.imresize(crop_center(x.astype(np.float32)), shape)
              for idx, x in enumerate(clip.iter_frames())]

    from keras.applications.imagenet_utils import preprocess_input

    n_frames = end_fid - begin_fid + 1

    frames_arr = np.empty((n_frames,) + shape + (3,), dtype=np.float32)

    fid = 0
    for idx, frame in enumerate(frames):
        if begin_fid <= idx <= end_fid:
            frames_arr[fid, :, :, :] = frame
            fid += 1

    # print(frames_arr)

    frames_arr = preprocess_input(frames_arr)

    features = model.predict(frames_arr, batch_size=batch_size)

    feat_filepath = os.path.join(str(output_dir) + '/'
                                 + str(name)
                                 + '_' + str(begin_fid)
                                 + '_' + str(end_fid)
                                 + '.npy')

    print("Save to: " + feat_filepath)

    with open(feat_filepath, 'wb') as f:
        np.save(f, features)
    return features


def extract_features(input_dir, output_dir, model_type='resnet50', batch_size=32):
    """
    Extracts features from a CNN trained on ImageNet classification from all
    videos in a directory.

    Args:
        input_dir (str): Input directory of videos to extract from.
        output_dir (str): Directory where features should be stored.
        model_type (str): Model type to use.
        batch_size (int): Batch size to use when processing.
    """

    input_dir = os.path.expanduser(input_dir)
    output_dir = os.path.expanduser(output_dir)

    if not os.path.isdir(input_dir):
        sys.stderr.write("Input directory '%s' does not exist!\n" % input_dir)
        sys.exit(1)

    if model_type.lower() == 'inceptionv3':
        from keras.applications import InceptionV3
        model = InceptionV3(include_top=True, weights='imagenet')
    elif model_type.lower() == 'xception':
        from keras.applications import Xception
        model = Xception(include_top=True, weights='imagenet')
    elif model_type.lower() == 'resnet50':
        from keras.applications import ResNet50
        model = ResNet50(include_top=True, weights='imagenet')
    elif model_type.lower() == 'vgg16':
        from keras.applications import VGG16
        model = VGG16(include_top=True, weights='imagenet')
    elif model_type.lower() == 'vgg19':
        from keras.applications import VGG19
        model = VGG19(include_top=True, weights='imagenet')
    else:
        sys.stderr.write("'%s' is not a valid ImageNet model.\n" % model_type)
        sys.exit(1)

    if model_type.lower() == 'inceptionv3' or model_type.lower() == 'xception':
        shape = (299, 299)

    # Get outputs of model from layer just before softmax predictions

    from keras.models import Model
    model = Model(model.inputs, output=model.layers[-2].output)

    # Create output directories

    visual_dir = os.path.join(output_dir, 'vid_full_features')  # RGB features
    # motion_dir = os.path.join(output_dir, 'motion') # Spatiotemporal features
    # opflow_dir = os.path.join(output_dir, 'opflow') # Optical flow features

    for directory in [visual_dir]:  # , motion_dir, opflow_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Find all videos that need to have features extracted

    def is_video(x):
        return x.endswith('.mp4') or x.endswith('.avi') or x.endswith('.mov')

    vis_existing = [x.split('.')[0] for x in os.listdir(visual_dir)]
    # mot_existing = [os.path.splitext(x)[0] for x in os.listdir(motion_dir)]
    # flo_existing = [os.path.splitext(x)[0] for x in os.listdir(opflow_dir)]

    video_filenames = [x for x in sorted(os.listdir(input_dir))
                       if is_video(x) and os.path.splitext(x)[0] not in vis_existing]

    # Go through each video and extract features

    from keras.applications.imagenet_utils import preprocess_input

    for video_filename in tqdm(video_filenames):

        # Open video clip for reading
        try:
            clip = VideoFileClip(os.path.join(input_dir, video_filename))
        except Exception as e:
            sys.stderr.write("Unable to read '%s'. Skipping...\n" % video_filename)
            sys.stderr.write("Exception: {}\n".format(e))
            continue

        # Sample frames at 1fps
        fps = int(np.round(clip.fps))
        frames = [scipy.misc.imresize(crop_center(x.astype(np.float32)), shape)
                  for idx, x in enumerate(clip.iter_frames()) if idx % fps == fps // 2]

        n_frames = len(frames)

        frames_arr = np.empty((n_frames,) + shape + (3,), dtype=np.float32)
        for idx, frame in enumerate(frames):
            frames_arr[idx, :, :, :] = frame

        frames_arr = preprocess_input(frames_arr)

        features = model.predict(frames_arr, batch_size=batch_size)

        name, _ = os.path.splitext(video_filename)
        feat_filepath = os.path.join(visual_dir, name + '.npy')

        with open(feat_filepath, 'wb') as f:
            np.save(f, features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Extract ImageNet features from videos.")

    parser.add_argument('-i', '--input', type=str, required=False,
                        help="Directory of videos to process.")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Directory where extracted features should be stored.")

    parser.add_argument('-m', '--model', default='inceptionv3', type=str,
                        help="ImageNet model to use.")
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help="Number of frames to be processed each batch.")

    parser.add_argument('-f', '--full', type=bool, default=False,
                        help="Extract full frames features from video.")

    parser.add_argument('-v', '--video', type=str, required=False,
                        default='../data/VidVRD-videos/ILSVRC2015_train_00005015.mp4',
                        help="The path of video")

    parser.add_argument('-bf', '--begin_fid', type=int, required=False,
                        default=0,
                        help="The begin id of frames")

    parser.add_argument('-ef', '--end_fid', type=int, required=False,
                        default=30,
                        help="The end id of frames")
    args = parser.parse_args()

    if args.full:
        extract_features(input_dir=args.input, output_dir=args.output,
                         model_type=args.model, batch_size=args.batch_size)
    else:
        extract_split_features(args.video, args.output, args.begin_fid, args.end_fid)
