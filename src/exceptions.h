#include <string>
#include <stdexcept>

class FileDoesNotExist : public std::runtime_error {
public:
    FileDoesNotExist(std::string path) : std::runtime_error(path + " does not exist.") {};
};

class WrongFileExtension : public std::runtime_error {
public:
    WrongFileExtension(std::string path, std::string extension) : std::runtime_error(path + " does not have \"" + extension + "\" extension") {};
};