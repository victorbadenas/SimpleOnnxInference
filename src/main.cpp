#include <iostream>
#include <cxxopts.hpp>
#include "OnnxInferenceRunner.h"

namespace fs = std::filesystem;

cxxopts::ParseResult parseArgumentsFromCommandLine(int argc, char** argv) {
    cxxopts::Options options(argv[0], "A short test of onnx runtime inference.");

    options.allow_unrecognised_options().add_options()
        ("m,modelPath", "Onnx model path", cxxopts::value<fs::path>())
        ("i,imagePath", "Input Image", cxxopts::value<fs::path>())
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
    bool debug = argStruct["debug"].as<bool>();

    std::cout << "Running " << argv[0] << " with args:" << std::endl;
    std::cout << "modelPath: " << modelPath << std::endl;
    std::cout << "debug: " << debug << std::endl;

    OnnxInferenceRunner onnxInferenceRunner;
    onnxInferenceRunner.CreateEnv();
    onnxInferenceRunner.CreateSessionOptions();
    onnxInferenceRunner.loadModel(modelPath);
    std::cout << onnxInferenceRunner.toString() << std::endl;
    std::cout << onnxInferenceRunner.run(imagePath) << std::endl;

    return 0;
}
