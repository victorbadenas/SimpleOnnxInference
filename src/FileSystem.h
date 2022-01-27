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

    static bool fileExists(const fs::path& pathToFile);
    static bool checkFileExtension(const fs::path& pathToFile, const std::string& extension);
    static cv::Mat loadImage(const fs::path& pathToFile);
    static cv::Mat loadImage(const std::string& stringPath);
    static Lines readLines(const fs::path& pathToFile);
};

#endif //TESTONNXRUNTIME_FILESYSTEM_H
