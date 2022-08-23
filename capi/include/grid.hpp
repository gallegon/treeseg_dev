#pragma once

template <class T>
class Grid {
private:
    char* data;

public:
    const size_t width;
    const size_t height;
    const size_t stride_x = sizeof(T) * height;
    const size_t stride_y = sizeof(T);

    Grid(size_t width, size_t height, void* data) : width(width), height(height), data{static_cast<char*>(data)} {}

    T* at(int i, int j) {
        return static_cast<T*>(static_cast<void*>(&(data[i * stride_x + j * stride_y])));
    }
};
