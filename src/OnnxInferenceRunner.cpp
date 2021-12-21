#include "OnnxInferenceRunner.h"


OnnxInferenceRunner::OnnxInferenceRunner() : 
    m_environment(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "test"){
    m_sessionOptions.SetIntraOpNumThreads(1);
    m_sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
}

void OnnxInferenceRunner::loadModel(fs::path modelPath) {
    if (!FileSystem::fileExists(modelPath))
    {
        throw FileDoesNotExist(modelPath);
    }
    else if (!FileSystem::checkFileExtension(modelPath, ".onnx"))
    {
        throw WrongFileExtension(modelPath, ".onnx");
    }
    m_modelPath = modelPath;
    m_session = std::make_unique<Ort::Session>(m_environment, modelPath.c_str(), m_sessionOptions);
    loadParameters();
}

void OnnxInferenceRunner::loadParameters() {
    Ort::AllocatorWithDefaultOptions allocator;
    auto inputTensorInfo = m_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    auto outputTensorInfo = m_session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();

    m_sessionParameters.input.nodes = m_session->GetInputCount();
    m_sessionParameters.input.name = m_session->GetInputName(0, allocator);
    m_sessionParameters.input.type = inputTensorInfo.GetElementType();
    m_sessionParameters.input.shape = inputTensorInfo.GetShape();

    m_sessionParameters.output.nodes = m_session->GetOutputCount();
    m_sessionParameters.output.name = m_session->GetOutputName(0, allocator);
    m_sessionParameters.output.type = outputTensorInfo.GetElementType();
    m_sessionParameters.output.shape = outputTensorInfo.GetShape();
}

const std::string OnnxInferenceRunner::toString() {
    if (!m_session) {
        return "OnnxInferenceRunner not initialised.";
    }

    std::ostringstream ss;
    ss << "OnnxInferenceRunner:" << std::endl;
    ss << "Number of Input Nodes: " << m_sessionParameters.input.nodes << std::endl;
    ss << "Input Name: " << m_sessionParameters.input.name << std::endl;
    ss << "Input Type: " << m_sessionParameters.input.type << std::endl;
    // ss << "Input Dimensions: " << m_sessionParameters.input.shape << std::endl;

    ss << "Number of Output Nodes: " << m_sessionParameters.output.nodes << std::endl;
    ss << "Output Name: " << m_sessionParameters.output.name << std::endl;
    ss << "Output Type: " << m_sessionParameters.output.type << std::endl;
    // ss << "Output Dimensions: " << m_sessionParameters.output.shape << std::endl;
    return ss.str();
}

