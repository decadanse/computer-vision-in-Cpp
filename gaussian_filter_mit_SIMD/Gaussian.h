#pragma once
#include <vector>
#include "Image.h"

//G(x, y) = 1 / (2 * pi * sigma^2) * e^(- (x^2 + y^2) / (2*sigma^2))


// Три нативные реализации одного и того же фильтра Гаусса:
//   Naive2D       — прямая 2D-свёртка,            O(k^2) операций на пиксель
//   Separable     — две 1D-свёртки (строки, затем столбцы), O(2k) на пиксель
//   SeparableSIMD — то же, но внутренний цикл на ARM NEON (8 пикселей за итерацию);
//                   на архитектурах без NEON откатывается на скалярный код
enum class GaussMethod { Naive2D, Separable, SeparableSIMD };

// sigma <= 0 -> автоподбор по размеру ядра (формула OpenCV):
//   sigma = 0.3 * ((ksize - 1) / 2 - 1) + 0.8
double auto_sigma(int ksize);

// Нормированное 1D-ядро Гаусса длины ksize (нечётной):
//   g(i) = exp(-(i - r)^2 / (2 sigma^2)),   r = ksize / 2
//   k(i) = g(i) / sum_j g(j)               (сумма весов = 1, яркость не меняется)
std::vector<float> gaussian_kernel_1d(int ksize, double sigma);

// sigmaY <= 0 -> sigmaY = sigmaX. Если и sigmaX <= 0 -> обе выбираются автоматически.
GrayImage gaussian_blur(const GrayImage& src, int ksize, double sigmaX, double sigmaY,
                        GaussMethod method, Border border = Border::Reflect101);
