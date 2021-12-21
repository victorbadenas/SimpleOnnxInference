# TestOnnxRuntime

![ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=flat&logo=ubuntu&logoColor=black) ![c++](https://img.shields.io/static/v1?label=C%2B%2B&message=20&color=blue&style=flat&logo=c%2B%2B&logoColor=blue&labelColor=white) ![onnxruntime](https://img.shields.io/static/v1?label=onnxruntime&message=1.10&color=blue&style=flat&logo=onnx&logoColor=blue&labelColor=white)

## install onnxruntime

```bash
./installonnxruntime.sh
```

## build

```bash
mkdir build
cd build
cmake -DONNXRUNTIME_ROOTDIR="~/.local/include/onnxruntime" -v ..
make
```

## run

```bash
./main
```
