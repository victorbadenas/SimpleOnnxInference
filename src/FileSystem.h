#ifndef TESTONNXRUNTIME_FILESYSTEM_H
#define TESTONNXRUNTIME_FILESYSTEM_H

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

class FileSystem {
public:
    static bool fileExists(fs::path &pathToFile);
    static bool fileExists(std::string &pathToFile);
    static bool checkFileExtension(fs::path &pathToFile, std::string extension);
    static bool checkFileExtension(std::string &pathToFile, std::string extension);
    static cv::Mat loadImage(fs::path &pathToFile);
    static cv::Mat loadImage(std::string &stringPath);
};

#endif //TESTONNXRUNTIME_FILESYSTEM_H
