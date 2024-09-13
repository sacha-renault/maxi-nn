#pragma once
#include <xtensor/xview.hpp>

inline xt::xstrided_slice<std::ptrdiff_t> strides() {
    return xt::all();
};

inline xt::xstrided_slice<std::ptrdiff_t> strides(std::ptrdiff_t exact) {
    return xt::range(exact, exact + 1);
};

inline xt::xstrided_slice<std::ptrdiff_t> strides(std::ptrdiff_t start, std::ptrdiff_t end) {
    return xt::range(start, end);
};
