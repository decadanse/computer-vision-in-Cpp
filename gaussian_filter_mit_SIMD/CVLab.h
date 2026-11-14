#pragma once
// Состояние лабораторной «Усредняющий фильтр Гаусса».
// Пишет фоновый поток (RunCVLab), читает поток рендера (DrawCVLab). Всё, кроме atomic, защищено mtx.

#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

struct BenchSeries
{
    std::string name;
    std::vector<double> x;  // размер ядра (px) либо ширина изображения (px)
    std::vector<double> ms; // медианное время одного вызова, мс
};

// Сравнение нативной реализации с cv::GaussianBlur при одних и тех же (ksize, sigmaX, sigmaY)
struct VerifyRow
{
    std::string name;
    int max_abs_diff = 0;      // max |native - opencv| по пикселям (0..255)
    double mismatch_pct = 0.0; // % пикселей, где значения не совпали
};

struct CVLabState
{
    std::mutex mtx;
    std::atomic<bool> loaded{false};
    std::atomic<bool> bench_done{false};
    std::atomic<bool> stop{false};

    cv::Mat src;             // исходное серое изображение (под mtx)
    std::string source_note; // откуда взято изображение

    std::vector<VerifyRow> verify;
    std::vector<BenchSeries> gauss_vs_k;    // время vs размер ядра (кадр 1280 px)
    std::vector<BenchSeries> gauss_vs_size; // время vs ширина кадра (ядро 15)
};

// Эталонная реализация на OpenCV (BORDER_REFLECT_101 — как у нативной по умолчанию)
cv::Mat gaussian_blur_opencv(const cv::Mat& gray, int ksize, double sigmaX, double sigmaY);

void RunCVLab(CVLabState& state); // вызывается в std::thread
