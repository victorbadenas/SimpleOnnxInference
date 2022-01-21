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
    ss << "Input Dimensions: " << m_sessionParameters.input.shape << std::endl;

    ss << "Number of Output Nodes: " << m_sessionParameters.output.nodes << std::endl;
    ss << "Output Name: " << m_sessionParameters.output.name << std::endl;
    ss << "Output Type: " << m_sessionParameters.output.type << std::endl;
    ss << "Output Dimensions: " << m_sessionParameters.output.shape << std::endl;
    return ss.str();
}

std::vector<float> OnnxInferenceRunner::run(cv::Mat imageData) {

}

std::vector<float> OnnxInferenceRunner::run(fs::path imagePath) {
    cv::Mat imageBGR = FileSystem::loadImage(imagePath);
    preprocessImage(imageBGR);
    return run(imageBGR);
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