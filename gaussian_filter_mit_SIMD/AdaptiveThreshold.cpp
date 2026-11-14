//#include "AdaptiveThreshold.h"
//
//#include <cassert>
//#include <cmath>
//#include <opencv2/imgproc.hpp>
//
//// Общий финальный шаг: dst = maxval, если src - T > -C, иначе 0.
//// T здесь — локальное среднее, уже округлённое до uint8 (как у OpenCV, у которого среднее 8-битное).
//// Условие  src > mean - C  <=>  src - mean > -C.
//// Для THRESH_BINARY OpenCV, насколько я помню, берёт delta = ceil(C) (целочисленная таблица);
//// если на тестах будут расхождения на границе порога — проверить именно это место.
//static GrayImage apply_threshold(const GrayImage& src, const GrayImage& mean, double C, uint8_t maxval)
//{
//    const int idelta = static_cast<int>(std::ceil(C));
//    GrayImage dst(src.w, src.h);
//    for (int y = 0; y < src.h; ++y)
//    {
//        const uint8_t* s = src.row(y);
//        const uint8_t* m = mean.row(y);
//        uint8_t* d = dst.row(y);
//        for (int x = 0; x < src.w; ++x)
//            d[x] = (static_cast<int>(s[x]) - static_cast<int>(m[x]) > -idelta) ? maxval : 0;
//    }
//    return dst;
//}
//
//// ---------------------------------------------------------------------------------------------
//// Локальное среднее через интегральное изображение (summed-area table).
////   S(x, y) = sum_{i < x, j < y} img(i, j)          (размер (W+1) x (H+1), нулевая 0-я строка/столбец)
////   Сумма по прямоугольнику [x0, x0+k) x [y0, y0+k):
////     sum = S(x0+k, y0+k) - S(x0, y0+k) - S(x0+k, y0) + S(x0, y0)
////   mean = sum / k^2   -> 4 обращения в память независимо от k.
//// Строим S один раз за O(W*H):  S(x+1, y+1) = rowsum(y, 0..x) + S(x+1, y).
//// uint64, чтобы не переполниться на больших кадрах (255 * W * H может превысить 2^32).
//// ---------------------------------------------------------------------------------------------
//GrayImage adaptive_threshold_mean(const GrayImage& src, int blockSize, double C, uint8_t maxval)
//{
//    assert(blockSize % 2 == 1 && blockSize >= 3);
//    const int r = blockSize / 2;
//    const int k = blockSize;
//
//    // Replicate — как в cv::adaptiveThreshold (BORDER_REPLICATE | BORDER_ISOLATED)
//    const GrayImage padded = pad_image(src, r, Border::Replicate);
//    const int pw = padded.w, ph = padded.h;
//    const size_t W = static_cast<size_t>(pw) + 1;
//
//    std::vector<uint64_t> S(W * (static_cast<size_t>(ph) + 1), 0);
//    for (int y = 0; y < ph; ++y)
//    {
//        const uint8_t* p = padded.row(y);
//        uint64_t rowsum = 0;
//        for (int x = 0; x < pw; ++x)
//        {
//            rowsum += p[x];
//            S[(y + 1) * W + (x + 1)] = rowsum + S[y * W + (x + 1)];
//        }
//    }
//
//    const uint64_t area = static_cast<uint64_t>(k) * k;
//    GrayImage mean(src.w, src.h);
//    for (int y = 0; y < src.h; ++y)
//    {
//        uint8_t* m = mean.row(y);
//        for (int x = 0; x < src.w; ++x)
//        {
//            // окно в padded-координатах: [x, x+k) x [y, y+k) — центр совпадает с пикселем (x, y) исходника
//            const uint64_t sum = S[(y + k) * W + (x + k)] - S[y * W + (x + k)]
//                               - S[(y + k) * W + x]       + S[y * W + x];
//            m[x] = static_cast<uint8_t>((sum + area / 2) / area); // округление к ближайшему
//        }
//    }
//    return apply_threshold(src, mean, C, maxval);
//}
//
//// Gaussian: T(x, y) = sum_{i,j} G(i, j) * src(x+i-r, y+j-r) - C, веса G — нормированные Гауссовы.
//// Пиксели, близкие к центру окна, весят больше -> порог меньше «ломается» на краях объектов, чем у mean.
//GrayImage adaptive_threshold_gauss(const GrayImage& src, int blockSize, double sigma, double C,
//                                   GaussMethod blurMethod, uint8_t maxval)
//{
//    assert(blockSize % 2 == 1 && blockSize >= 3);
//    const GrayImage blurred = gaussian_blur(src, blockSize, sigma, sigma, blurMethod, Border::Replicate);
//    return apply_threshold(src, blurred, C, maxval);
//}
//
//// ------------------------------ эталон OpenCV ------------------------------------------------
//cv::Mat adaptive_threshold_opencv(const cv::Mat& gray, AdaptiveMethod m, int blockSize, double C)
//{
//    cv::Mat dst;
//    cv::adaptiveThreshold(gray, dst, 255,
//                          m == AdaptiveMethod::Mean ? cv::ADAPTIVE_THRESH_MEAN_C : cv::ADAPTIVE_THRESH_GAUSSIAN_C,
//                          cv::THRESH_BINARY, blockSize, C);
//    return dst;
//}
//
//cv::Mat gaussian_blur_opencv(const cv::Mat& gray, int ksize, double sigmaX, double sigmaY)
//{
//    cv::Mat dst;
//    cv::GaussianBlur(gray, dst, cv::Size(ksize, ksize), sigmaX, sigmaY, cv::BORDER_REFLECT_101);
//    return dst;
//}
