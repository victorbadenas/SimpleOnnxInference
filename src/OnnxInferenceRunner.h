#ifndef TESTONNXRUNTIME_ONNXINFERENCERUNNER_H
#define TESTONNXRUNTIME_ONNXINFERENCERUNNER_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/dnn/dnn.hpp>
#include "FileSystem.h"
#include "exceptions.h"
#include "vectorOperations.h"
#include "imageOperations.h"
#include <iostream>
#include <map>

std::string onnxDataTypeToString(ONNXTensorElementDataType dataType);

class OnnxInferenceRunner {
public:
    OnnxInferenceRunner();
    std::string toString();
    void loadModel(fs::path modelPath);

    std::vector<float> run(fs::path imagePath);
    std::vector<float> run(cv::Mat imageData);
private:
    // Image preprocessing
    cv::Mat preprocessImage(const cv::Mat& imageData);

    // Parameters
    OrtLoggingLevel m_loggingLevel;
    int m_intraNumThreads;
    GraphOptimizationLevel m_graphOptLevel;
    std::string m_logId;

    // Onnx Variables
    Ort::Env m_environment = Ort::Env(nullptr);
    Ort::SessionOptions m_sessionOptions = Ort::SessionOptions(nullptr);
    std::shared_ptr<Ort::Session> m_session;
    Ort::AllocatorWithDefaultOptions m_allocator;

    // Getter Graph parameters
    size_t GetSessionInputCount();
    size_t GetSessionOutputCount();
    std::vector<int64_t> GetSessionInputNodeDims(size_t index);
    std::vector<int64_t> GetSessionOutputNodeDims(size_t index);
    const char *GetSessionInputName(size_t index);
    const char *GetSessionOutputName(size_t index);
    ONNXTensorElementDataType GetSessionInputNodeType(size_t index);
    ONNXTensorElementDataType GetSessionOutputType(size_t index);

    // Setup methods
    void CreateEnv();
    void CreateSessionOptions();
};

#endif //TESTONNXRUNTIME_ONNXINFERENCERUNNER_H
