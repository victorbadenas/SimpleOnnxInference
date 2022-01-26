#include <iostream>
#include <cxxopts.hpp>
#include "OnnxInferenceRunner.h"

namespace fs = std::filesystem;

cxxopts::ParseResult parseArgumentsFromCommandLine(int argc, char** argv) {
    cxxopts::Options options(argv[0], "A short test of onnx runtime inference.");

    options.allow_unrecognised_options().add_options()
        ("m,modelPath", "Onnx model path", cxxopts::value<fs::path>())
        ("i,imagePath", "Input Image", cxxopts::value<fs::path>())
        ("l,labelPath", "Label File", cxxopts::value<fs::path>()->default_value(""))
        ("d,debug", "Enable debugging", cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print usage");

    cxxopts::ParseResult result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    return result;
}


int main(int argc, char** argv){
    cxxopts::ParseResult argStruct = parseArgumentsFromCommandLine(argc, argv);

    fs::path modelPath = argStruct["modelPath"].as<fs::path>();
    fs::path imagePath = argStruct["imagePath"].as<fs::path>();
    fs::path labelPath = argStruct["labelPath"].as<fs::path>();
    bool debug = argStruct["debug"].as<bool>();

    std::cout << "------------ script parameters ------------" << std::endl;
    std::cout << "Running " << argv[0] << " with args:" << std::endl;
    std::cout << "modelPath: " << modelPath << std::endl;
    std::cout << "imagePath: " << imagePath << std::endl;
    std::cout << "labelPath: " << labelPath << std::endl;
    std::cout << "debug: " << debug << std::endl;

    OnnxInferenceRunner onnxInferenceRunner;
    onnxInferenceRunner.loadModel(modelPath);
    if (labelPath != "")
        onnxInferenceRunner.loadLabels(labelPath);

    std::cout << "-------- onnx inference parameters --------" << std::endl;
    std::cout << onnxInferenceRunner.toString();

    std::cout << "----------------- logits ------------------" << std::endl;
    OnnxInferenceRunner::Logits logits= onnxInferenceRunner.run(imagePath);
    std::cout << onnxInferenceRunner.run(imagePath) << std::endl;

    std::cout << "------------------results------------------" << std::endl;
    OnnxInferenceRunner::Result result = onnxInferenceRunner.getResults(logits);
    std::cout << "Index: " << std::get<0>(result) << std::endl;
    std::cout << "Confidence: " << std::get<1>(result) << std::endl;
    std::cout << "Label: " << std::get<2>(result) << std::endl;
    return 0;
}
