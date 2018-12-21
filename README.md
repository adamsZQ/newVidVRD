# New Video Visual Relation Detection with Transformer
## T-VideoVRD

### Mainly Structure

![Main Structure of model](https://github.com/Daviddddl/newVidVRD/blob/master/imgs/main_structure.png)

### Environment

Tensorflow1.8 + CUDA9.0 + python3.6 + Tensor2tensor

Step 1: 
  
  download and install Anaconda 
  https://www.anaconda.com/download/#linux
  
  for more install details: 
  https://blog.csdn.net/Davidddl/article/details/81873606
  
```
sudo bash Anaconda3-5.2.0-Linux-x86_64.sh

conda --version

conda create -n tensorflow pip python=3.6

source activate tensorflow

pip install --upgrade pip

(tensorflow)$ pip install --ignore-installed --upgrade https://download.tensorflow.google.cn/linux/gpu/tensorflow_gpu-1.8.0-cp36-cp36m-linux_x86_64.whl
```

Step 2: install tensor2tensor
```
(tensorflow)$ pip install tensor2tensor
```

### Python Package

#### Download NUS-VidVRD Dataset

Dataset:
NUS-VidVRD: http://lms.comp.nus.edu.sg/research/VidVRD.html
```bash
sudo add-apt-repository ppa:djcj/hybrid
sudo apt-get update
sudo apt-get install ffmpeg
sudo apt-get install build-essential cmake
sudo apt-get install libgtk-3-dev
sudo apt-get install libboost-all-dev
```

#### Install all of requirements

```bash
pip install -r requirements.txt
```

### Download Pretrained Model

We adopted the Faster-RCNN model from: https://github.com/endernewton/tf-faster-rcnn

And the result of this step can be download from: https://drive.google.com/drive/folders/1I2LLIVNAcOe2DWZZ_bzovY1o2MgvYy4s?hl=zh-CN

https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN

### Download Extracted features or generate by yourself

```bash
cd newVidVRD/utils && 
python extract_features.py -i [Videos_dir] \
    -o [Output_dir] \
    -m [model_name] \
    -b [batch_size]
```
```
# The input_dir and output_dir are your own identified directory, former of which includes raw videos to be extracted and the latter is to store results.

# And we provide several CNN networks to be chosen:
    InceptionV3, Xception, Resnet50, VGG16, VGG19
# You can simply name one of these and run the python script "extract_features.py".

# The default size of Batch_Size is 32, but apparently you can set it yourself.
```

We extracted frames features from several types of neural networks:

| network | download_link |
| ------- | ------------- |
| inceptionv3 | uploading |
| xception | uploading |
| resnet50 | uploading |
| vgg16 | uploading |
| vgg19 | uploading |

### Generate encoding data for tensor2tensor

Modify the 'models/t2t_datagen.sh', declare the 'OUT_DIR' and 'TEP_DIR', then run the shell script as following.
```bash
cd models && bash t2t_datagen.sh # [frame_class|frame_text|text_class|video_class]
```

### Train the model

Similarly, modify the 'models/t2t_trainer.sh' and run the shell.
```bash
cd models && bash t2t_trainer.sh frame_class transformer
```

###  Decode the test data

Modify and run the 'models/t2t_decoder.sh' to test.
```bash
cd models && bash t2t_decoder.sh
```

### Results

|               |  predicate    | phrase        | relation    |
| ------------- |:-------------:| -------------:| -----------:|
| VRD (R100)        | Coming soon | Coming soon | Coming soon |
| VG (R100)         | Coming soon | Coming soon | Coming soon |
| VG (R100)         | Coming soon | Coming soon | Coming soon |
| Sth-Sth V1        | Coming soon | Coming soon | Coming soon |
| Sth-Sth V2        | Coming soon | Coming soon | Coming soon |
| NUS-VVRD-V1 (1k)  | Coming soon | Coming soon | Coming soon |
| NUS-VVRD-V2 (10k) | Coming soon | Coming soon | Coming soon |


### Mainly directory structure of this project

Because of the number of Video and Image Dataset have tons of files, the detail of data omit some less vital parts.
```
├── data
│   ├── first_relation_dict.txt
│   ├── second_relation_dict.txt
│   ├── test
│   ├── tmp_data
│   ├── train
│   ├── traj_cls
│   ├── traj_cls_gt
│   ├── VidVRD-features
│   │   ├── separate_features
│   │   ├── test_Instances
│   │   └── train_Instances
│   └── VidVRD-videos
├── __init__.py
├── models
│   ├── faster-rcnn
│   │   ├── lib
│   │   └── tools
│   ├── frame_class.py
│   ├── frame_text.py
│   ├── ImgVRD
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── ST-GAN
│   │   └── V-Trans-E
│   ├── __init__.py
│   ├── other-models
│   │   ├── TRN
│   │   ├── TSN
│   │   └── two-stream
│   ├── __pycache__
│   │   ├── frame_class.cpython-36.pyc
│   │   ├── frame_text.cpython-36.pyc
│   │   ├── __init__.cpython-36.pyc
│   │   ├── text_class.cpython-36.pyc
│   │   └── video_class.cpython-36.pyc
│   ├── README.md
│   ├── t2t_datagen.sh
│   ├── t2t_decoder.sh
│   ├── t2t_define_hyper.sh
│   ├── t2t_trainer.sh
│   ├── text_class.py
│   └── video_class.py
├── README.md
├── requirements.txt
├── sim_baseline
│   ├── baseline
│   │   ├── association.py
│   │   ├── feature.py
│   │   ├── features
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── models
│   │   ├── __pycache__
│   │   └── trajectory.py
│   ├── baseline.py
│   ├── dataset.py
│   ├── evaluation.py
│   ├── __init__.py
│   ├── __pycache__
│   │   └── dataset.cpython-36.pyc
│   └── README.md
├── treeee.txt
└── utils
    ├── ext_features
    ├── extract_features.py
    ├── frames.py
    ├── get_data.py
    ├── get_relation_list.py
    ├── __init__.py
    ├── __pycache__
    │   ├── extract_features.cpython-36.pyc
    │   ├── frames.cpython-36.pyc
    │   ├── get_data.cpython-36.pyc
    │   ├── get_relation_list.cpython-36.pyc
    │   └── VRDInstance.cpython-36.pyc
    ├── README.md
    └── VRDInstance.py
```

### All Details of baselines

#### Original Video VRD

'sim_baseline/*' is the directory of original Video VRD programmed by Xindi Shang:
```
@inproceedings{shang2017video,
    author={Shang, Xindi and Ren, Tongwei and Guo, Jingfan and Zhang, Hanwang and Chua, Tat-Seng},
    title={Video Visual Relation Detection},
    booktitle={ACM International Conference on Multimedia},
    address={Mountain View, CA USA},
    month={October},
    year={2017}
}
```
https://github.com/xdshang/VidVRD-helper

And we transformed it to Tensorflow1.8 + Python3, more independent project:
https://github.com/Daviddddl/originVidVRD

#### Image VRD
'models/ImgVRD/V-Trans-E/*' is the folder of V-Trans-E project, mainly based on two other projects:

https://github.com/zawlin/cvpr17_vtranse

https://github.com/yangxuntu/vrd

```bash
@inproceedings{Zhang_2017_CVPR,
  author    = {Hanwang Zhang, Zawlin Kyaw, Shih-Fu Chang, Tat-Seng Chua},
  title     = {Visual Translation Embedding Network for Visual Relation Detection},
  booktitle = {CVPR},
  year      = {2017},
}
```

more details about this baseline are introduced in 'models/ImgVRD/README.md':
https://github.com/Daviddddl/newVidVRD/tree/master/models/ImgVRD

#### Video Action Recognition

##### TRN

The work of TRN is built on Pytorch, we didn't transfer it to Tensorflow, just ran the dataset using their model on Pytorch following the README

https://github.com/metalbubble/TRN-pytorch

```
@article{zhou2017temporalrelation,
    title = {Temporal Relational Reasoning in Videos},
    author = {Zhou, Bolei and Andonian, Alex and Oliva, Aude and Torralba, Antonio},
    journal={European Conference on Computer Vision},
    year={2018}
}
```

##### TSN

This work of TSN is also running on Pytorch.

https://github.com/yjxiong/temporal-segment-networks

https://github.com/yjxiong/tsn-pytorch

```
@inproceedings{TSN2016ECCV,
  author    = {Limin Wang and
               Yuanjun Xiong and
               Zhe Wang and
               Yu Qiao and
               Dahua Lin and
               Xiaoou Tang and
               Luc {Val Gool}},
  title     = {Temporal Segment Networks: Towards Good Practices for Deep Action Recognition},
  booktitle   = {ECCV},
  year      = {2016},
}
```

##### Two-Stream

Convolutional Two-Stream Network Fusion for Video Action Recognition

https://github.com/feichtenhofer/twostreamfusion

```
@inproceedings{feichtenhofer2016convolutional,
      title={Convolutional Two-Stream Network Fusion for Video Action Recognition},
      author={Feichtenhofer, Christoph and Pinz, Axel and Zisserman, Andrew},
      booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2016}
    }
```

https://github.com/jeffreyhuang1/two-stream-action-recognition

