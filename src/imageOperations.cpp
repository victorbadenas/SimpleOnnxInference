#include "imageOperations.h"

std::string getImageInfo(const cv::Mat& imageData) {
    std::ostringstream os;
    os << "Shape=" << std::vector<int>{imageData.channels(), imageData.cols, imageData.rows} << ", ";
    os << "Type=" << cvTypeToString(imageData.type()) << ", ";
    return os.str();
}

std::string cvTypeToString(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

cv::Mat normalizePerChannel(const cv::Mat& imageData, std::vector<float> means, std::vector<float> std) {
    cv::Mat normalizedImage;
    cv::Mat channels[imageData.channels()];
    cv::split(imageData, channels);

    for (int i=0; i<imageData.channels(); ++i) {
        if (std[i] == 0.0f)
            throw std::runtime_error("division by 0");
        channels[i] = (channels[i] - means[i]) / std[i];
    }
    cv::merge(channels, imageData.channels(), normalizedImage);
    return normalizedImage;
}