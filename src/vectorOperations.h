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
void copyVector(std::vector<T> source, std::vector<T>& target) {
    std::copy(source.begin(), source.end(), std::back_inserter(target));
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