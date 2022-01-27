#!/usr/bin/env bash

set -ex

scriptDir=$( cd "$(dirname "$0")" ; pwd -P )
OnnxDir=~/.local/include/onnxruntime
OpenCVDir=/usr/local/share/OpenCV

function installOnnx {
    echo Installing onnx in $OnnxDir ...

    brew install onnxruntime
    cd /tmp
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.10.0/onnxruntime-osx-x86_64-1.10.0.tgz
    tar -xvf onnxruntime-osx-*.tgz
    mkdir -p $OnnxDir
    mv onnxruntime-osx-*/* $OnnxDir
    cd -
}

function installOpenCV {
    echo Installing opencv...

    cd /tmp
    wget https://github.com/opencv/opencv/archive/refs/tags/3.4.16.zip
    unzip 3.4.16.zip
    mkdir -p opencv-3.4.16/build
    cd opencv-3.4.16/build
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_LIST=core,improc,imgcodecs,dnn ..
    make -j8
    sudo make install
}

# install dependancies
# brew install cmake gcc

# Install opencv
[ -d $OpenCVDir ] && echo found OpenCV installation in ${OpenCVDir}, skipping... || installOpenCV

# OnnxInstallation
[ -d $OnnxDir ] && echo found OnnxRuntime installation in ${OnnxDir}, skipping... || installOnnx
cd "${scriptDir}"