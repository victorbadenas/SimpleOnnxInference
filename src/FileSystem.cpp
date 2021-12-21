#include "FileSystem.h"

bool FileSystem::fileExists(std::string &stringPath) {
    fs::path pathToFile(stringPath);
    return exists(pathToFile);
}

bool FileSystem::fileExists(fs::path &pathToFile) {
    if (pathToFile.empty() || !pathToFile.has_filename()) return false;
    return fs::exists(pathToFile);
}

bool FileSystem::checkFileExtension(fs::path &pathToFile, std::string extension) {
    if (pathToFile.empty() || !pathToFile.has_extension()) return false;
    return pathToFile.extension() == extension;
};

bool FileSystem::checkFileExtension(std::string &stringPath, std::string extension) {
    fs::path pathToFile(stringPath);
    return checkFileExtension(pathToFile, extension);
};
