#!/usr/bin/env bash
cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.10.0/onnxruntime-linux-x64-1.10.0.tgz
tar -xvf onnxruntime-linux-x64-*.tgz
mkdir -p ~/.local/include/
mv onnxruntime-linux-x64-* ~/.local/include/onnxruntime
cd -