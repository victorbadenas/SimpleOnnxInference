#!/usr/bin/env bash

rm -rf build/
conan install -if build/ .
cmake -DONNXRUNTIME_ROOTDIR="~/.local/include/onnxruntime" -DOpenCV_DIR="/usr/local/share/OpenCV" -S . -B build/
make -C build/