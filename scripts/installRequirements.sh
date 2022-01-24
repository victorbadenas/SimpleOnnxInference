#!/usr/bin/env bash

# Install opencv

sudo apt-get install build-essential unzip
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev

wget https://github.com/opencv/opencv/archive/refs/tags/3.4.16.zip
unzip 3.4.16.zip
mkdir -p opencv-3.4.16/build
cd opencv-3.4.16/build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j12
sudo make install

# OnnxInstallation

cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.10.0/onnxruntime-linux-x64-1.10.0.tgz
tar -xvf onnxruntime-linux-x64-*.tgz
mkdir -p ~/.local/include/onnxruntime
mv onnxruntime-linux-x64-*/* ~/.local/include/onnxruntime
cd -
