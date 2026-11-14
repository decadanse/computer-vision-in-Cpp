#pragma once
// Минимальный «нативный» тип изображения (8 бит, 1 канал) без зависимости от OpenCV
// в самих алгоритмах. OpenCV используется только для I/O и как эталон для сравнения.

#include <cstddef>
#include <cstdint>
#include <vector>
#include <opencv2/core.hpp>

struct GrayImage
{
    int w = 0, h = 0;
    std::vector<uint8_t> px; // построчно, без выравнивания строк (stride == w)

    GrayImage() = default;
    GrayImage(int w_, int h_) : w(w_), h(h_), px(static_cast<size_t>(w_) * h_) {}

    uint8_t*       row(int y)       { return px.data() + static_cast<size_t>(y) * w; }
    const uint8_t* row(int y) const { return px.data() + static_cast<size_t>(y) * w; }
};

// Способ достраивания изображения за границей (нужен окну, выходящему за край):
//   Replicate  : aaa|abcdefg|ggg   (так делает cv::adaptiveThreshold)
//   Reflect101 : gfe|abcdefg|fed   (дефолт cv::GaussianBlur)
enum class Border { Replicate, Reflect101 };

// Отображает произвольный индекс i в допустимый [0, n-1] согласно режиму границы.
inline int border_index(int i, int n, Border b)
{
    if (n == 1) return 0;
    if (b == Border::Replicate)
        return i < 0 ? 0 : (i >= n ? n - 1 : i);
    // Reflect101: -1 -> 1, -2 -> 2, n -> n-2, n+1 -> n-3 ... (крайний пиксель не дублируется)
    while (i < 0 || i >= n)
    {
        if (i < 0)  i = -i;
        if (i >= n) i = 2 * (n - 1) - i;
    }
    return i;
}

// Добавляет рамку шириной r со всех сторон: результат (w+2r) x (h+2r).
// После этого окно k x k (k = 2r+1) для ЛЮБОГО пикселя целиком лежит внутри буфера,
// и во внутренних циклах не нужны проверки границ (ветвления убивают SIMD).
inline GrayImage pad_image(const GrayImage& src, int r, Border b)
{
    GrayImage out(src.w + 2 * r, src.h + 2 * r);
    std::vector<int> xmap(out.w);
    for (int x = 0; x < out.w; ++x) xmap[x] = border_index(x - r, src.w, b);

    for (int y = 0; y < out.h; ++y)
    {
        const uint8_t* s = src.row(border_index(y - r, src.h, b));
        uint8_t* d = out.row(y);
        for (int x = 0; x < out.w; ++x) d[x] = s[xmap[x]];
    }
    return out;
}

inline GrayImage from_mat(const cv::Mat& m) // ожидается CV_8UC1
{
    GrayImage g(m.cols, m.rows);
    for (int y = 0; y < m.rows; ++y)
        std::copy(m.ptr<uint8_t>(y), m.ptr<uint8_t>(y) + m.cols, g.row(y));
    return g;
}

inline cv::Mat to_mat(const GrayImage& g)
{
    return cv::Mat(g.h, g.w, CV_8UC1, const_cast<uint8_t*>(g.px.data())).clone();
}
