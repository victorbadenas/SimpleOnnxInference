#ifndef TESTONNXRUNTIME_FILESYSTEM_H
#define TESTONNXRUNTIME_FILESYSTEM_H

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <string>
#include "exceptions.h"

namespace fs = std::filesystem;

class FileSystem {
public:
    typedef std::vector<std::string> Lines;

    static bool fileExists(fs::path &pathToFile);
    static bool checkFileExtension(fs::path &pathToFile, std::string extension);
    static cv::Mat loadImage(fs::path &pathToFile);
    static cv::Mat loadImage(std::string &stringPath);
    static Lines readLines(fs::path &pathToFile);
};

#endif //TESTONNXRUNTIME_FILESYSTEM_H
