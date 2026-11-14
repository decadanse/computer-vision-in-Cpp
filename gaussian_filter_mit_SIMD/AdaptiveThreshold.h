//#pragma once
//#include "Gaussian.h"
//#include <opencv2/core.hpp>
//
//// Адаптивная бинаризация: порог считается ЛОКАЛЬНО для каждого пикселя.
////   T(x, y) = mean_{окно blockSize x blockSize}(src) - C          (метод mean)
////   T(x, y) = gauss_{окно blockSize x blockSize}(src) - C         (метод gaussian)
////   dst(x, y) = maxval, если src(x, y) > T(x, y), иначе 0         (THRESH_BINARY)
//// blockSize — нечётное >= 3. C — константа, вычитаемая из локального среднего.
//
//// mean: локальное среднее через ИНТЕГРАЛЬНОЕ изображение — O(1) на пиксель при любом blockSize.
//GrayImage adaptive_threshold_mean(const GrayImage& src, int blockSize, double C, uint8_t maxval = 255);
//
//// gaussian: локальное среднее = Гауссово размытие окном blockSize (sigma <= 0 -> авто, как в OpenCV).
//GrayImage adaptive_threshold_gauss(const GrayImage& src, int blockSize, double sigma, double C,
//                                   GaussMethod blurMethod, uint8_t maxval = 255);
//
//// ---- эталон на OpenCV (для проверки корректности и бенчмарка) ----
//enum class AdaptiveMethod { Mean, Gaussian };
//cv::Mat adaptive_threshold_opencv(const cv::Mat& gray, AdaptiveMethod m, int blockSize, double C);
//cv::Mat gaussian_blur_opencv(const cv::Mat& gray, int ksize, double sigmaX, double sigmaY);
