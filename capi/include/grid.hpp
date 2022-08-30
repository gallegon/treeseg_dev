#pragma once

template <class T>
class Grid {
public:
    char* data;
    const size_t width;
    const size_t height;
    const size_t stride_x = sizeof(T) * height;
    const size_t stride_y = sizeof(T);

    Grid(size_t width, size_t height, void* data) : width(width), height(height), data{static_cast<char*>(data)} {}

    Grid(size_t width, size_t height) : width(width), height(height) {
        data = static_cast<char*>(static_cast<void*>((new T[width * height])));
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                set(i, j, 0);
            }
        }
    }

    inline T* at(int i, int j) {
        return static_cast<T*>(static_cast<void*>(&(data[i * stride_x + j * stride_y])));
    }

    inline T get(int i, int j) {
        return *at(i, j);
    }

    inline void set(int i, int j, T v) {
        *at(i, j) = v;
    }
};
