#ifndef TESTONNXRUNTIME_VECTOROPERATIONS_H
#define TESTONNXRUNTIME_VECTOROPERATIONS_H

#include <vector>
#include <iostream>
#include <numeric>

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T>
void softmax(std::vector<T>& vector) {
    T expSum = 0.0;
    for (size_t i=0; i<vector.size(); ++i)
        expSum += std::exp(vector.at(i));
    for (auto& value : vector)
        value = std::exp(value) / expSum;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (size_t i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

#endif //TESTONNXRUNTIME_VECTOROPERATIONS_H