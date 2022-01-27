# SimpleOnnxInference

![CI](https://github.com/vbadenas/SimpleOnnxInference/actions/workflows/cmake.yml/badge.svg?style=flat)
![debian](	https://img.shields.io/badge/Debian-A81D33?style=flat&logo=debian&logoColor=white)
![macos](https://img.shields.io/badge/mac%20os-000000?style=flat&logo=apple&logoColor=white)

![c++](https://img.shields.io/static/v1?label=C%2B%2B&message=20&color=lightblue&style=flat&logo=c%2B%2B&logoColor=blue&labelColor=white)
![onnxruntime](https://img.shields.io/static/v1?label=onnxruntime&message=1.10&color=lightblue&style=flat&logo=onnx&logoColor=blue&labelColor=white)
![opencv](https://img.shields.io/static/v1?label=opencv&message=3.4.1&color=lightblue&style=flat&logo=opencv&logoColor=blue&labelColor=white)

## install requirements

For this project we need opencv and onnx. OpenCV needs to be built from source. For this project we will be using the v3.4.1 tag in their [repository](https://github.com/opencv/opencv/tree/3.4.16).

### Ubuntu

For ubuntu installation, follow or run the [installRequirements.sh](scripts/installRequirements.sh) bash script.

```bash
./scripts/installRequirements.sh
```

### MacOS

For macos installation, follow or run the [installRequirementsOsx.sh](scripts/installRequirementsOsx.sh) bash script.

```bash
./scripts/installRequirementsOsx.sh
```

## Loading a sample model from torchvision for imagenet classification

This repository provides a [getTorchvisionModel.py](scripts/getTorchvisionModel.py) python script to load and export a model present in the torchvision package to onnx format. For running the python script create an environment and install the requirements in [requirements.txt](requirements.txt). Then simply run the file as follows:

```bash
python3 scripts/getTorchvisionModel.py -m ${modelArch} -O 13 -o ${outFolder} -i 3 224 224
```

where `-i 3 224 224` corresponds to the size of the input to the network.

## Build

To build, install the conan packages (`cxxopts`):

```bash
conan install -if build/ .
```

Then configure the cmake project:

```bash
cmake -DONNXRUNTIME_ROOTDIR="~/.local/include/onnxruntime" -DOpenCV_DIR="/usr/local/share/OpenCV" -S . -B build/
```

And build:

```bash
cmake --build build/
```

## Run

The binary is stored in the `./build/bin/` folder. It requires the modelPath (onnx format), the imagepath (jpeg format). the label file is optional but highly encouraged as it provides the output label in human-readable format. 

```bash
./build/bin/main -m ${modelPath} -i ${imagePath} -l ${labelPath} -d
```

The label file must be a plain text file where each line corresponds to the label of the line index where is located. for instance, for a 3 class classification, the label file would be as follows:

```text
class1
class2
class3
```
