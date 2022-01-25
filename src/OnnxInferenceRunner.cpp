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

// std::vector<float> OnnxInferenceRunner::run(cv::Mat imageData) {
std::vector<float> OnnxInferenceRunner::run() {
    return std::vector<float>{};
}

std::vector<float> OnnxInferenceRunner::run(fs::path imagePath) {
    cv::Mat imageBGR = FileSystem::loadImage(imagePath);
    preprocessImage(imageBGR);
    return run();
}

cv::Mat OnnxInferenceRunner::preprocessImage(cv::Mat imageData){
    return imageData;
    //TODO: implement preprocessing
    /*
    step 2: Resize the image.
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
    cv::resize(imageBGR, resizedImageBGR,
               cv::Size(inputDims.at(2), inputDims.at(3)),
               cv::InterpolationFlags::INTER_CUBIC);

    // step 3: Convert the image to HWC RGB UINT8 format.
    cv::cvtColor(resizedImageBGR, resizedImageRGB,
                 cv::ColorConversionCodes::COLOR_BGR2RGB);
    // step 4: Convert the image to HWC RGB float format by dividing each pixel by 255.
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

    // step 5: Split the RGB channels from the image.   
    cv::Mat channels[3];
    cv::split(resizedImage, channels);

    //step 6: Normalize each channel.
    // Normalization per channel
    // Normalization parameters obtained from
    // https://github.com/onnx/models/tree/master/vision/classification/squeezenet
    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;

    //step 7: Merge the RGB channels back to the image.
    cv::merge(channels, 3, resizedImage);

    // step 8: Convert the image to CHW RGB float format.
    // HWC to CHW
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);
    */
}

size_t OnnxInferenceRunner::GetSessionInputCount() {
    if (nullptr == m_session) {
        throw std::runtime_error("Received null session pointer");
    }
    return m_session->GetInputCount();
}

size_t OnnxInferenceRunner::GetSessionOutputCount() {
    if (nullptr == m_session) {
        throw std::runtime_error("Received null session pointer");
    }
    return m_session->GetOutputCount();
}

std::vector<int64_t> OnnxInferenceRunner::GetSessionInputNodeDims(size_t index) {
    if (nullptr == m_session) {
        throw std::runtime_error("Received null session pointer");
    }
    return m_session->GetInputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape();
}

std::vector<int64_t> OnnxInferenceRunner::GetSessionOutputNodeDims(size_t index) {
    if (nullptr == m_session) {
        throw std::runtime_error("Received null session pointer");
    }
    return m_session->GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape();
}

ONNXTensorElementDataType OnnxInferenceRunner::GetSessionInputNodeType(size_t index) {
    if (nullptr == m_session) {
        throw std::runtime_error("Received null session pointer");
    }
    return m_session->GetInputTypeInfo(index).GetTensorTypeAndShapeInfo().GetElementType();
}

ONNXTensorElementDataType OnnxInferenceRunner::GetSessionOutputType(size_t index) {
    if (nullptr == m_session) {
        throw std::runtime_error("Received null session pointer");
    }
    return m_session->GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo().GetElementType();
}

const char *OnnxInferenceRunner::GetSessionInputName(size_t index, OrtAllocator *allocator) {
    if (nullptr == m_session) {
        throw std::runtime_error("Received null session pointer");
    }
    if (nullptr == allocator) {
        throw std::runtime_error("Received null session pointer");
    }

    return m_session->GetInputName(index, allocator);
}

const char *OnnxInferenceRunner::GetSessionOutputName(size_t index, OrtAllocator *allocator) {
    if (nullptr == m_session) {
        throw std::runtime_error("Received null session pointer");
    }
    if (nullptr == allocator) {
        throw std::runtime_error("Received null session pointer");
    }

    return m_session->GetOutputName(index, allocator);
}


std::string OnnxInferenceRunner::toString() {
    Ort::AllocatorWithDefaultOptions inputAllocator;
    Ort::AllocatorWithDefaultOptions outputAllocator;

    std::ostringstream os;
    os << "OnnxInferenceRunner:" << std::endl;
    os << "Number of Input Nodes: " << GetSessionInputCount() << std::endl;
    for (size_t i = 0; i<GetSessionInputCount(); ++i) {
        os << "Input " << i << ": ";
        os << "Name=" << GetSessionInputName(i, inputAllocator) << ", ";
        os << "Type=" << onnxDataTypeToString(GetSessionInputNodeType(i)) << ", ";
        os << "Shape=" << GetSessionInputNodeDims(i) << std::endl;
    }

    os << "Number of Output Nodes: " << GetSessionOutputCount() << std::endl;
    for (size_t i = 0; i<GetSessionOutputCount(); ++i) {
        os << "Output " << i << ": ";
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
