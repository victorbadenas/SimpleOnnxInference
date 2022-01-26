#!/usr/bin/env bash

scriptDir=$(realpath $(dirname "$0"))
OnnxDir=~/.local/include/onnxruntime
OpenCVDir=/usr/local/share/OpenCV

function installOnnx {
    echo Installing onnx in $OnnxDir ...

    cd /tmp
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.10.0/onnxruntime-linux-x64-1.10.0.tgz
    tar -xvf onnxruntime-linux-x64-*.tgz
    mkdir -p $OnnxDir
    mv onnxruntime-linux-x64-*/* $OnnxDir
    cd -
}

function installOpenCV {
    echo Installing opencv...

    cd /tmp
    sudo apt install -y build-essential unzip
    sudo apt install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
    sudo apt install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev

    wget https://github.com/opencv/opencv/archive/refs/tags/3.4.16.zip
    unzip 3.4.16.zip
    mkdir -p opencv-3.4.16/build
    cd opencv-3.4.16/build
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_LIST=core,improc,imgcodecs ..
    make -j8
    sudo make install
}

# Install opencv
[ -d $OpenCVDir ] && echo found OpenCV installation in ${OpenCVDir}, skipping... || installOpenCV

# OnnxInstallation
[ -d $OnnxDir ] && echo found OnnxRuntime installation in ${OnnxDir}, skipping... || installOnnx
cd ${scriptDir}
