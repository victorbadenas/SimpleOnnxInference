#ifndef TESTONNXRUNTIME_IMAGEOPERATIONS_H
#define TESTONNXRUNTIME_IMAGEOPERATIONS_H

#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <sstream>
#include "vectorOperations.h"

std::string getImageInfo(const cv::Mat& imageData);
std::string cvTypeToString(int type);
cv::Mat normalizePerChannel(const cv::Mat& imageData, std::vector<float> means, std::vector<float> std);

#endif //TESTONNXRUNTIME_IMAGEOPERATIONS_H