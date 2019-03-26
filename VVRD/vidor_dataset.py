import json
import os
import os.path
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data_utl
from torchvision import transforms
import videotransforms


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        img = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + '.jpg'))[:, :, [2, 1, 0]]
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def load_flow_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        imgx = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'x.jpg'), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'y.jpg'), cv2.IMREAD_GRAYSCALE)

        w, h = imgx.shape
        if w < 224 or h < 224:
            d = 224. - min(w, h)
            sc = 1 + d / min(w, h)
            imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
            imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

        imgx = (imgx / 255.) * 2 - 1
        imgy = (imgy / 255.) * 2 - 1
        img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


class Charades(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None):

        self.data = make_dataset(split_file, split, root, mode)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, dur, nf = self.data[index]
        start_f = random.randint(1, nf - 65)

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, start_f, 64)
        else:
            imgs = load_flow_frames(self.root, vid, start_f, 64)
        label = label[:, start_f:start_f + 64]

        imgs = self.transforms(imgs)

        return video_to_tensor(imgs), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)


def make_dataset(split_files, split, root, mode, num_classes):
    """
    make vidor dataset
    :param split_files: annotation directory ('/home/daivd/PycharmProjects/vidor/annotation')
    :param split: ['training', 'validation']
    :param root: video frames directory / video directory
    :param mode: rgb, flow
    :param num_classes: object_num, action_num = 42, relation_num = 42 + 8 (space_relation = 8)
    :return:
    """
    dataset = []

    for root, dirs, files in os.walk(split_files):
        print(root)

    # i = 0
    # for vid in data.keys():
    #     if data[vid]['subset'] != split:
    #         continue
    #
    #     if not os.path.exists(os.path.join(root, vid)):
    #         continue
    #     num_frames = len(os.listdir(os.path.join(root, vid)))
    #     if mode == 'flow':
    #         num_frames = num_frames // 2
    #
    #     if num_frames < 66:
    #         continue
    #
    #     label = np.zeros((num_classes, num_frames), np.float32)
    #
    #     fps = num_frames / data[vid]['duration']
    #     for ann in data[vid]['actions']:
    #         for fr in range(0, num_frames, 1):
    #             if ann[1] < fr / fps < ann[2]:
    #                 label[ann[0], fr] = 1  # binary classification
    #     dataset.append((vid, label, data[vid]['duration'], num_frames))
    #     i += 1
    #
    # return dataset


if __name__ == '__main__':

    # batch_size = 8 * 5
    #
    # train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
    #                                        videotransforms.RandomHorizontalFlip()])
    #
    # test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    #
    # root_path = 'data/Charades_v1_rgb'
    #
    # dataset = Charades('data/charades.json', 'training', root_path,
    #                    'rgb', train_transforms)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=36,
    #                                          pin_memory=True)
    # val_dataset = Charades('data/charades.json', 'testing', root_path,
    #                        'rgb', test_transforms)
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=36,
    #                                              pin_memory=True)
    #
    # dataloaders = {'train': dataloader, 'val': val_dataloader}
    # datasets = {'train': dataset, 'val': val_dataset}
    #
    # for data in dataloaders['train']:
    #     inputs, labels = data
    #     print(inputs, labels)

    # ============================
    make_dataset(split_files='/home/daivd/PycharmProjects/vidor/annotation',
                 split='training',
                 root='/home/daivd/PycharmProjects/vidor/train_vids',
                 mode='rgb',
                 num_classes=42)
