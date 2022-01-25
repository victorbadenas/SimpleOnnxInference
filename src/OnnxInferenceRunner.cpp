#include "OnnxInferenceRunner.h"


OnnxInferenceRunner::OnnxInferenceRunner(
):
    m_loggingLevel(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING),
    m_intraNumThreads(0),
    m_graphOptLevel(GraphOptimizationLevel::ORT_DISABLE_ALL),
    m_logId("")
{
}

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

void OnnxInferenceRunner::CreateEnv() {
    m_environment = Ort::Env(m_loggingLevel, m_logId.c_str());
}

void OnnxInferenceRunner::CreateSessionOptions() {
    m_sessionOptions = Ort::SessionOptions();
    m_sessionOptions.SetIntraOpNumThreads(m_intraNumThreads);
    m_sessionOptions.SetGraphOptimizationLevel(m_graphOptLevel);
}

std::vector<float> OnnxInferenceRunner::run(cv::Mat imageData) {
    return std::vector<float>{};
}

std::vector<float> OnnxInferenceRunner::run(fs::path imagePath) {
    cv::Mat image = FileSystem::loadImage(imagePath);
    image = preprocessImage(image);
    return run(image);
}

cv::Mat OnnxInferenceRunner::resizeImage(cv::Mat imageData) {
    std::vector<int64_t> inputDims = GetSessionInputNodeDims(0);
    cv::Mat resizedImage;
    cv::resize(imageData, resizedImage,
               cv::Size(inputDims.at(2), inputDims.at(3)),
               cv::InterpolationFlags::INTER_CUBIC);
    return resizedImage;
}

cv::Mat OnnxInferenceRunner::preprocessImage(cv::Mat imageData) {
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, normalizedImage, preprocessedImage;

//    std::cout << "input image bgr: " << getImageInfo(imageData) << std::endl;
    resizedImageBGR = resizeImage(imageData);
//    std::cout << "resized image bgr: " << getImageInfo(resizedImageBGR) << std::endl;
    cv::cvtColor(resizedImageBGR, resizedImageRGB, cv::ColorConversionCodes::COLOR_BGR2RGB);
//    std::cout << "resized image rgb: " << getImageInfo(resizedImageBGR) << std::endl;
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);
//    std::cout << "resized image float[0,1] rgb: " << getImageInfo(resizedImage) << std::endl;
    normalizedImage = normalizePerChannel(resizedImageRGB, std::vector<float>{0.485, 0.456, 0.406}, std::vector<float>{0.229, 0.224, 0.225});
//    std::cout << "final preprocessed image: " << getImageInfo(normalizedImage) << std::endl;
    cv::dnn::blobFromImage(normalizedImage, preprocessedImage);
    return preprocessedImage;
}

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

const char *OnnxInferenceRunner::GetSessionInputName(size_t index, OrtAllocator *allocator) {
    if (nullptr == m_session) {
        throw std::runtime_error("Session not yet initialized (model not loaded)");
    }
    if (nullptr == allocator) {
        throw std::runtime_error("Received null allocator pointer");
    }

    return m_session->GetInputName(index, allocator);
}

const char *OnnxInferenceRunner::GetSessionOutputName(size_t index, OrtAllocator *allocator) {
    if (nullptr == m_session) {
        throw std::runtime_error("Session not yet initialized (model not loaded)");
    }
    if (nullptr == allocator) {
        throw std::runtime_error("Received null allocator pointer");
    }

    return m_session->GetOutputName(index, allocator);
}


std::string OnnxInferenceRunner::toString() {
    if (m_session == nullptr) {
        throw std::runtime_error("Session not yet initialized (model not loaded)");
    }
    Ort::AllocatorWithDefaultOptions inputAllocator;
    Ort::AllocatorWithDefaultOptions outputAllocator;

    std::ostringstream os;
    os << "OnnxInferenceRunner with parameters:" << std::endl;
    os << "Number of Input Nodes: " << GetSessionInputCount() << std::endl;
    for (size_t i = 0; i<GetSessionInputCount(); ++i) {
        os << "\tInput " << i << ": ";
        os << "Name=" << GetSessionInputName(i, inputAllocator) << ", ";
        os << "Type=" << onnxDataTypeToString(GetSessionInputNodeType(i)) << ", ";
        os << "Shape=" << GetSessionInputNodeDims(i) << std::endl;
    }
    os << std::endl;

    os << "Number of Output Nodes: " << GetSessionOutputCount() << std::endl;
    for (size_t i = 0; i<GetSessionOutputCount(); ++i) {
        os << "\tOutput " << i << ": ";
        os << "Name=" << GetSessionOutputName(i, outputAllocator) << ", ";
        os << "Type=" << onnxDataTypeToString(GetSessionOutputType(i)) << ", ";
        os << "Shape=" << GetSessionOutputNodeDims(i) << std::endl;
    }
    return os.str();
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
