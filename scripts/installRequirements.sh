#!/usr/bin/env bash

# Install libraries for opencv

sudo apt install -y libgtk2.0-dev libva-dev libvdpau-dev

# OnnxInstallation

cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.10.0/onnxruntime-linux-x64-1.10.0.tgz
tar -xvf onnxruntime-linux-x64-*.tgz
mkdir -p ~/.local/include/onnxruntime
mv onnxruntime-linux-x64-*/* ~/.local/include/onnxruntime
cd -
