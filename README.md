# new Video Visual Relation Detection with Transformer

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
| inceptionv3 | --- |
| xception | --- |
| resnet50 | --- |
| vgg16 | --- |
| vgg19 | --- |



| script | encoder | decoder |
| ------ | ------- | ------- |
| my_img_trans | Self-Attention, Feed-forward | Self-Attention, Source-Target-Attention, Feed-forward |
| my_img_trans_2d | Self-Attention, Feed-forward | Self-Attention, Source-Target-Attention, Feed-forward |
| my_lstm | lstm | lstm |
| my_mtf_img_trans | Modulation Transfer Function | Modulation Transfer Function |
| 
