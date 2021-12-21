#ifndef TESTONNXRUNTIME_FILESYSTEM_H
#define TESTONNXRUNTIME_FILESYSTEM_H

#include <filesystem>
#include <string>

namespace fs = std::filesystem;

class FileSystem {
public:
    static bool fileExists(fs::path &pathToFile);
    static bool fileExists(std::string &pathToFile);
    static bool checkFileExtension(fs::path &pathToFile, std::string extension);
    static bool checkFileExtension(std::string &pathToFile, std::string extension);
};

#endif //TESTONNXRUNTIME_FILESYSTEM_H
