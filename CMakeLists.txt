cmake_minimum_required(VERSION 3.13)

# Project and compilation flags
project(main VERSION 0.1.0 LANGUAGES C CXX)
if (WIN32)
    string(APPEND CMAKE_CXX_FLAGS " /W4")
else()
    string(APPEND CMAKE_CXX_FLAGS " -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra")
    string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
endif()

set (CMAKE_CXX_STANDARD 20)
set(BUILD_SHARED_LIBS ON)

# basic conan config
message(STATUS ${CMAKE_BINARY_DIR})
message(STATUS ${CMAKE_SOURCE_DIR})
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# add external cmake files
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_BINARY_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# find packages
find_package(cxxopts REQUIRED)
find_package(glog REQUIRED)
find_package(OpenCV REQUIRED)

# opencv variable logging
message(STATUS "OpenCV_FOUND: ${OpenCV_FOUND}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")

# onnxruntime providers
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
option(onnxruntime_USE_OPENVINO "Build with OpenVINO support" OFF)
option(onnxruntime_USE_NNAPI_BUILTIN "Build with builtin NNAPI lib for Android NNAPI support" OFF)
option(onnxruntime_USE_DNNL "Build with DNNL support" OFF)
option(onnxruntime_USE_NUPHAR "Build with Nuphar" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")

if(NOT ONNXRUNTIME_ROOTDIR)
    message(FATAL_ERROR "please specify the installation path for ONNXRUNTIME_ROOTDIR to cmake")
endif()

# onnxruntime debug log
message(STATUS "OnnRuntime directory: ${ONNXRUNTIME_ROOTDIR}")

if(onnxruntime_USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()
if(onnxruntime_USE_OPENVINO)
    add_definitions(-DUSE_OPENVINO)
endif()
if(onnxruntime_USE_NNAPI_BUILTIN)
    add_definitions(-DUSE_NNAPI)
endif()
if(onnxruntime_USE_DNNL)
    add_definitions(-DUSE_DNNL)
endif()
if(onnxruntime_USE_NUPHAR)
    add_definitions(-DUSE_NUPHAR)
endif()
if(onnxruntime_USE_TENSORRT)
    add_definitions(-DUSE_TENSORRT)
endif()
if(onnxruntime_USE_DML)
    message("Enabling DML")
    add_definitions(-DUSE_DML)
endif()

# # project library bindings
include_directories("${ONNXRUNTIME_ROOTDIR}/include" "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session" ${OpenCV_INCLUDE_DIRS} )
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

add_library(utils STATIC
    src/FileSystem.h
    src/FileSystem.cpp
    src/vectorOperations.h
    src/imageOperations.h
    src/imageOperations.cpp
    src/exceptions.h
    src/OnnxInferenceRunner.h
    src/OnnxInferenceRunner.cpp
)

target_link_libraries(utils PUBLIC ${OpenCV_LIBS} glog)

add_executable(main src/main.cpp)
target_link_libraries(main PUBLIC utils onnxruntime cxxopts::cxxopts glog)
