#include "OnnxInferenceRunner.h"

/* initialization member functions */

// constructor and definition of member variables
OnnxInferenceRunner::OnnxInferenceRunner(
):
    m_loggingLevel(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING),
    m_intraNumThreads(1),
    m_graphOptLevel(GraphOptimizationLevel::ORT_DISABLE_ALL)
{
    CreateEnv();
    CreateSessionOptions();
}

// load onnx file
void OnnxInferenceRunner::loadModel(fs::path modelPath) {
    if (!FileSystem::fileExists(modelPath)) {
        throw FileDoesNotExist(modelPath);
    }
    else if (!FileSystem::checkFileExtension(modelPath, ".onnx")) {
        throw WrongFileExtension(modelPath, ".onnx");
    }
    else if (m_sessionOptions == nullptr) {
        throw std::runtime_error("Ort::SessionOptions not initialized.");
    }
    else if (m_environment == nullptr) {
        throw std::runtime_error("Ort::Env not initialized.");
    }
    m_session = std::make_shared<Ort::Session>(m_environment, modelPath.c_str(), m_sessionOptions);
}

// create onnx environment
void OnnxInferenceRunner::CreateEnv() {
    m_environment = Ort::Env(m_loggingLevel, m_logId.c_str());
}

// create onnx session options
void OnnxInferenceRunner::CreateSessionOptions() {
    m_sessionOptions = Ort::SessionOptions();
    m_sessionOptions.SetIntraOpNumThreads(m_intraNumThreads);
    m_sessionOptions.SetGraphOptimizationLevel(m_graphOptLevel);
}

/* Input preprocessing functions */

// preprocess image for network
cv::Mat OnnxInferenceRunner::preprocessImage(const cv::Mat& imageData) {
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, normalizedImage, preprocessedImage;
    std::vector<int64_t> inputDims = GetSessionInputNodeDims(0);

    cv::resize(imageData, resizedImageBGR, cv::Size(inputDims.at(2), inputDims.at(3)), cv::InterpolationFlags::INTER_CUBIC);
    cv::cvtColor(resizedImageBGR, resizedImageRGB, cv::ColorConversionCodes::COLOR_BGR2RGB);
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);
    normalizedImage = normalizePerChannel(resizedImage, std::vector<float>{0.485, 0.456, 0.406}, std::vector<float>{0.229, 0.224, 0.225});
    cv::dnn::blobFromImage(normalizedImage, preprocessedImage);
    return preprocessedImage;
}

/* inference member functions */

// run inference for image from path
std::vector<float> OnnxInferenceRunner::run(fs::path imagePath) {
    cv::Mat image = FileSystem::loadImage(imagePath);
    image = preprocessImage(image);
    return run(image);
}

// run inference for cv::Mat structure
std::vector<float> OnnxInferenceRunner::run(cv::Mat imageData) {
    if (m_session == nullptr)
        throw std::runtime_error("Session not yet initialized (model not loaded)");

    std::vector<int64_t> inputDims = GetSessionInputNodeDims(0);
    std::vector<int64_t> outputDims = GetSessionOutputNodeDims(0);

    size_t inputTensorSize = vectorProduct(inputDims);
    std::vector<float> inputTensorValues(inputTensorSize);
    inputTensorValues.assign(imageData.begin<float>(), imageData.end<float>());

    size_t outputTensorSize = vectorProduct(outputDims);
    std::vector<float> outputTensorValues(outputTensorSize);

    std::vector<const char*> inputNames{GetSessionInputName(0)}, outputNames{GetSessionOutputName(0)};
    std::vector<Ort::Value> inputTensors, outputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(), inputDims.size()));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValues.data(), outputTensorSize, outputDims.data(), outputDims.size()));
    m_session->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), 1, outputNames.data(), outputTensors.data(), 1);
    softmax(outputTensorValues);
    return outputTensorValues;
}

/* onnx session parameters getters */

size_t OnnxInferenceRunner::GetSessionInputCount() {
    if (nullptr == m_session) {
        throw std::runtime_error("Session not yet initialized (model not loaded)");
    }
    return m_session->GetInputCount();
}

size_t OnnxInferenceRunner::GetSessionOutputCount() {
    if (nullptr == m_session) {
        throw std::runtime_error("Session not yet initialized (model not loaded)");
    }
    return m_session->GetOutputCount();
}

std::vector<int64_t> OnnxInferenceRunner::GetSessionInputNodeDims(size_t index) {
    if (nullptr == m_session) {
        throw std::runtime_error("Session not yet initialized (model not loaded)");
    }
    return m_session->GetInputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape();
}

std::vector<int64_t> OnnxInferenceRunner::GetSessionOutputNodeDims(size_t index) {
    if (nullptr == m_session) {
        throw std::runtime_error("Session not yet initialized (model not loaded)");
    }
    return m_session->GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape();
}

ONNXTensorElementDataType OnnxInferenceRunner::GetSessionInputNodeType(size_t index) {
    if (nullptr == m_session) {
        throw std::runtime_error("Session not yet initialized (model not loaded)");
    }
    return m_session->GetInputTypeInfo(index).GetTensorTypeAndShapeInfo().GetElementType();
}

ONNXTensorElementDataType OnnxInferenceRunner::GetSessionOutputType(size_t index) {
    if (nullptr == m_session) {
        throw std::runtime_error("Session not yet initialized (model not loaded)");
    }
    return m_session->GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo().GetElementType();
}

const char *OnnxInferenceRunner::GetSessionInputName(size_t index) {
    if (nullptr == m_session) {
        throw std::runtime_error("Session not yet initialized (model not loaded)");
    }
//    Ort::AllocatorWithDefaultOptions allocator;
    return m_session->GetInputName(index, m_allocator);
}

const char *OnnxInferenceRunner::GetSessionOutputName(size_t index) {
    if (nullptr == m_session) {
        throw std::runtime_error("Session not yet initialized (model not loaded)");
    }
//    Ort::AllocatorWithDefaultOptions allocator;
    return m_session->GetOutputName(index, m_allocator);
}

/* toString methods for printing onnx sessions' parameters */

std::string OnnxInferenceRunner::toString() {
    if (m_session == nullptr) {
        throw std::runtime_error("Session not yet initialized (model not loaded)");
    }

    std::ostringstream os;
    os << "OnnxInferenceRunner with parameters:" << std::endl;
    os << "Number of Input Nodes: " << GetSessionInputCount() << std::endl;
    for (size_t i = 0; i<GetSessionInputCount(); ++i) {
        os << "\tInput " << i << ": ";
        os << "Name=" << GetSessionInputName(i) << ", ";
        os << "Type=" << onnxDataTypeToString(GetSessionInputNodeType(i)) << ", ";
        os << "Shape=" << GetSessionInputNodeDims(i) << std::endl;
    }
    os << "Number of Output Nodes: " << GetSessionOutputCount() << std::endl;
    for (size_t i = 0; i<GetSessionOutputCount(); ++i) {
        os << "\tOutput " << i << ": ";
        os << "Name=" << GetSessionOutputName(i) << ", ";
        os << "Type=" << onnxDataTypeToString(GetSessionOutputType(i)) << ", ";
        os << "Shape=" << GetSessionOutputNodeDims(i) << std::endl;
    }
    return os.str();
}

OnnxInferenceRunner::Result OnnxInferenceRunner::getResults(OnnxInferenceRunner::Logits logits) {
    size_t maxElementIndex = std::max_element(logits.begin(),logits.end()) - logits.begin();
    float maxElement = *std::max_element(logits.begin(), logits.end());
    float minElement = *std::min_element(logits.begin(), logits.end());

    std::string label;
    if (m_labels.size() == logits.size() && !m_labels.empty()) {
        label = m_labels.at(maxElementIndex);
    } else {
        std::cout << "labels not loaded or number of labels does not match output array size. Not reporting human readable label." << std::endl;
    }
    return {maxElementIndex, maxElement - minElement, label};
}

void OnnxInferenceRunner::loadLabels(fs::path labelPath) {
    try {
        m_labels = FileSystem::readLines(labelPath);
    } catch (std::exception& exception) {
        std::cout << "Something happened when parsing label file. Defaulting to behaviour with no labelFile." << std::endl;
        std::cout << "Original Exception: " << exception.what() << std::endl;
    }
}

std::string onnxDataTypeToString(ONNXTensorElementDataType dataType) {
    std::map<ONNXTensorElementDataType, std::string> typeToStringMap({
        {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, "UNDEFINED"},
        {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "FLOAT"},
        {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, "UINT8"},
        {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, "INT8"},
        {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, "UINT16"},
        {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, "INT16"},
        {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, "INT32"},
        {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, "INT64"},
        {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, "STRING"},
        {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, "BOOL"},
        {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, "FLOAT16"},
        {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, "DOUBLE"},
        {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, "UINT32"},
        {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, "UINT64"},
        {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64, "COMPLEX64"},
        {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128, "COMPLEX128"},
        {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, "BFLOAT16"},
    });
    return typeToStringMap.at(dataType);
}
