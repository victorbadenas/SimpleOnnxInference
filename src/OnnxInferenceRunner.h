#ifndef TESTONNXRUNTIME_ONNXINFERENCERUNNER_H
#define TESTONNXRUNTIME_ONNXINFERENCERUNNER_H

#include <onnxruntime_cxx_api.h>
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
};

class OnnxInferenceRunner {
public:
    OnnxInferenceRunner();
    void loadModel(fs::path modelPath);
    const std::string toString();
private:
    void loadParameters();

    fs::path m_modelPath = "";
    Ort::Env m_environment;
    Ort::SessionOptions m_sessionOptions;
    SessionParameters m_sessionParameters;
    std::unique_ptr<Ort::Session> m_session = NULL;
};

#endif //TESTONNXRUNTIME_ONNXINFERENCERUNNER_H
