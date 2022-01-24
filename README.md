# TestOnnxRuntime

![ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=flat&logo=ubuntu&logoColor=black)
![c++](https://img.shields.io/static/v1?label=C%2B%2B&message=20&color=blue&style=flat&logo=c%2B%2B&logoColor=blue&labelColor=white)
![onnxruntime](https://img.shields.io/static/v1?label=onnxruntime&message=1.10&color=lightblue&style=flat&logo=onnx&logoColor=blue&labelColor=white)
![opencv](https://img.shields.io/static/v1?label=opencv&message=3.4.1&color=cyan&style=flat&logo=opencv&logoColor=blue&labelColor=white)

## install requirements

For this project we need opencv and onnx. Both of them can be installed with the following bash script on ubuntu.

```bash
./scripts/installRequirements.sh
```

## build

```bash
mkdir build
conan install -if build/ .
cmake -DONNXRUNTIME_ROOTDIR="~/.local/include/onnxruntime" -DOpenCV_DIR="/usr/local/share/OpenCV" -S . -B build/
make -C build/
```

## run

```bash
./build/bin/main -m ${modelPath}
```
