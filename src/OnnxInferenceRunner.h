#ifndef TESTONNXRUNTIME_ONNXINFERENCERUNNER_H
#define TESTONNXRUNTIME_ONNXINFERENCERUNNER_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/core/mat.hpp>
#include "FileSystem.h"
#include "exceptions.h"
#include "vectorOperations.h"
#include <iostream>

struct NodeParameters {
    size_t nodes;
    char* name;
    ONNXTensorElementDataType type;
    std::vector<int64_t> shape;
};

struct SessionParameters {
    NodeParameters input;
    NodeParameters output;

    void toString(std::ostream &os) const;
    [[nodiscard]] std::string toString() const;
};

class OnnxInferenceRunner {
public:
    OnnxInferenceRunner(fs::path& modelPath, Ort::SessionOptions& sessionOptions);
    std::string toString();

    std::vector<float> run(fs::path imagePath);
    std::vector<float> run();
private:
    Ort::Session loadModel(fs::path modelPath);
    void loadParameters();
    cv::Mat preprocessImage(cv::Mat imageData);

    Ort::Env m_environment;
    Ort::SessionOptions m_sessionOptions;
    SessionParameters m_sessionParameters;
    Ort::Session m_session;
};

#endif //TESTONNXRUNTIME_ONNXINFERENCERUNNER_H
