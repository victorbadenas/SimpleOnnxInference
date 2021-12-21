#include <onnxruntime_cxx_api.h>
#include "FileSystem.h"
#include "exceptions.h"

class OnnxInferenceRunner {
public:
    OnnxInferenceRunner(fs::path modelPath);
};
