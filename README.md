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

```bash
pip install -r requirements.txt
```

### Download Pretrained Model

