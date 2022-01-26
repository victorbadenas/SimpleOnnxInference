#!/usr/bin/env bash

set -ex

scriptDir=$(realpath "$(dirname "$0")")
OnnxDir=~/.local/include/onnxruntime
OpenCVDir=/usr/local/share/OpenCV

function installOnnx {
    echo Installing onnx in $OnnxDir ...

    cd /tmp
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.10.0/onnxruntime-osx-x86_64-1.10.0.tgz
    tar -xvf onnxruntime-linux-osx-*.tgz
    mkdir -p $OnnxDir
    mv onnxruntime-linux-osx-*/* $OnnxDir
    cd -
}

function installOpenCV {
    echo Installing opencv...

    cd /tmp
    wget https://github.com/opencv/opencv/archive/refs/tags/3.4.16.zip
    unzip 3.4.16.zip
    mkdir -p opencv-3.4.16/build
    cd opencv-3.4.16/build
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
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