#!/bin/sh
sudo apt install ntp nano htop

### Install dependencies for python3 "cv2"
sudo apt update
sudo apt install -y build-essential make cmake cmake-curses-gui \
                      git g++ pkg-config curl libfreetype6-dev \
                      libcanberra-gtk-module libcanberra-gtk3-module \
                      python3-dev python3-pip
sudo pip3 install -U pip==20.2.1 Cython testresources setuptools
cd ${HOME}/project/jetson_nano
./install_protobuf-3.8.0.sh
sudo pip3 install numpy==1.16.1 matplotlib==3.2.2

### Test tegra-cam.py (using a USB webcam)
cd ${HOME}/project
wget https://gist.githubusercontent.com/jkjung-avt/86b60a7723b97da19f7bfa3cb7d2690e/raw/3dd82662f6b4584c58ba81ecba93dd6f52c3366c/tegra-cam.py
# python3 tegra-cam.py --usb --vid 0

sudo apt install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev \
                      zip libjpeg8-dev liblapack-dev libblas-dev gfortran
sudo pip3 install -U numpy==1.16.1 future==0.18.2 mock==3.0.5 h5py==2.10.0 \
                       keras_preprocessing==1.1.1 keras_applications==1.0.8 \
                       gast==0.2.2 futures pybind11
sudo pip3 install --pre --extra-index-url \
                    https://developer.download.nvidia.com/compute/redist/jp/v44 \
                    tensorflow==1.15.2

### Clone the tensorrt_demos code
cd ${HOME}/project
git clone https://github.com/jkjung-avt/tensorrt_demos.git
###
### Build TensorRT engine for GoogLeNet
cd ${HOME}/project/tensorrt_demos/googlenet
make
./create_engine
###
### Build TensorRT engines for MTCNN
cd ${HOME}/project/tensorrt_demos/mtcnn
make
./create_engines
###
### Build the "pytrt" Cython module
cd ${HOME}/project/tensorrt_demos
sudo pip3 install Cython
make

### Install dependencies and build TensorRT engines
cd ${HOME}/project/tensorrt_demos/ssd
./install.sh
./build_engines.sh

### Install dependencies and build TensorRT engine
sudo pip3 install onnx==1.4.1
cd ${HOME}/project/tensorrt_demos/plugins
make
cd ${HOME}/project/tensorrt_demos/yolo
./download_yolo.sh

cd ${HOME}/project
git clone https://github.com/JetsonHacksNano/installVSCode.git
cd installVSCode
./installVSCode.sh

cd ~/project
git clone --recursive https://github.com/dusty-nv/jetson-inference
cd jetson-inference
mkdir build
cd build
cmake ../
make
sudo make install
sudo ldconfig

# Test
cd ~/project
mkdir -p my-recognition-python
cd my-recognition-python
touch my-recognition.py
chmod +x my-recognition.py
wget https://github.com/dusty-nv/jetson-inference/raw/master/data/images/black_bear.jpg 
wget https://github.com/dusty-nv/jetson-inference/raw/master/data/images/brown_bear.jpg
wget https://github.com/dusty-nv/jetson-inference/raw/master/data/images/polar_bear.jpg

printf """
#!/usr/bin/python3

import jetson.inference
import jetson.utils

import time
import argparse

# parse the command line
parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, help='filename of the image to process')
parser.add_argument('--network', type=str, default='googlenet', help='model to use, can be:  googlenet, resnet-18, ect.')
args = parser.parse_args()

# load an image (into shared CPU/GPU memory)
img = jetson.utils.loadImage(args.filename)

load_net_stime = time.perf_counter()

# load the recognition network
net = jetson.inference.imageNet(args.network)

load_net_etime = time.perf_counter()

print('Load network take time around {} seconds'.format(load_net_etime-load_net_stime))

pred_stime = time.perf_counter()

# classify the image
class_idx, confidence = net.Classify(img)

# find the object description
class_desc = net.GetClassDesc(class_idx)

pred_etime = time.perf_counter()

# print out the result
print('image is recognized as \'{:s}\' (class #{:d}) with {:f} percent confidence'.format(class_desc, class_idx, confidence * 100))
print('Predict take time around {} seconds'.format(pred_etime-pred_stime))
""" >> ~/project/my-recognition-python/my-recognition.py

python3 ~/project/my-recognition-python/my-recognition.py \
--network=resnet-50 ~/project/my-recognition-python/black_bear.jpg

export DOCKER_COMPOSE_VERSION=1.27.4
sudo apt-get install libhdf5-dev
sudo apt-get install libssl-dev
sudo pip3 install docker-compose=="${DOCKER_COMPOSE_VERSION}"
pip install docker-compose

sudo -H pip install -U jetson-stats

pip install redis