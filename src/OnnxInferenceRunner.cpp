#include "OnnxInferenceRunner.h"


void SessionParameters::toString(std::ostream& ss) const {
    ss << "Number of Input Nodes: " << input.nodes << std::endl;
    ss << "Input Name: " << input.name << std::endl;
    ss << "Input Type: " << input.type << std::endl;
    ss << "Input Dimensions: " << input.shape << std::endl;

    ss << "Number of Output Nodes: " << output.nodes << std::endl;
    ss << "Output Name: " << output.name << std::endl;
    ss << "Output Type: " << output.type << std::endl;
    ss << "Output Dimensions: " << output.shape << std::endl;
}

std::string SessionParameters::toString() const {
    std::ostringstream ss;
    toString(ss);
    return ss.str();
}

OnnxInferenceRunner::OnnxInferenceRunner(
    fs::path& modelPath,
    Ort::SessionOptions& sessionOptions
):
    m_environment(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "test"),
    m_sessionOptions(std::move(sessionOptions)),
    m_session(loadModel(modelPath))
{
    loadParameters();
}

Ort::Session OnnxInferenceRunner::loadModel(fs::path modelPath) {
    if (!FileSystem::fileExists(modelPath))
    {
        throw FileDoesNotExist(modelPath);
    }
    else if (!FileSystem::checkFileExtension(modelPath, ".onnx"))
    {
        throw WrongFileExtension(modelPath, ".onnx");
    }
    return Ort::Session(m_environment, modelPath.c_str(), m_sessionOptions);
}

void OnnxInferenceRunner::loadParameters() {
    Ort::AllocatorWithDefaultOptions allocator;
    auto inputTensorInfo = m_session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    auto outputTensorInfo = m_session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();

    m_sessionParameters.input.nodes = m_session.GetInputCount();
    m_sessionParameters.input.name = m_session.GetInputName(0, allocator);
    m_sessionParameters.input.type = inputTensorInfo.GetElementType();
    copyVector(inputTensorInfo.GetShape(), m_sessionParameters.input.shape);

    m_sessionParameters.output.nodes = m_session.GetOutputCount();
    m_sessionParameters.output.name = m_session.GetOutputName(0, allocator);
    m_sessionParameters.output.type = outputTensorInfo.GetElementType();
    copyVector(outputTensorInfo.GetShape(), m_sessionParameters.output.shape);
    std::cout << m_sessionParameters.toString() << std::endl;
}

std::string OnnxInferenceRunner::toString() {
    if (!m_session) {
        return "OnnxInferenceRunner not initialised.";
    }

    std::ostringstream os;
    os << "OnnxInferenceRunner:" << std::endl << "sessionParameters:" << std::endl;
    m_sessionParameters.toString(os);
    return os.str();
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