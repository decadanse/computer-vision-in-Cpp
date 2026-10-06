#include "Gaussian.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

//G(x, y) = 1 / (2 * pi * sigma^2) * e^(- (x^2 + y^2) / (2*sigma^2))

double auto_sigma(int ksize)
{
    return 0.3 * ((ksize - 1) * 0.5 - 1.0) + 0.8;
}

std::vector<float> gaussian_kernel_1d(int ksize, double sigma)
{
    assert(ksize % 2 == 1 && ksize >= 1);
    if (sigma <= 0.0) sigma = auto_sigma(ksize);

    const int r = ksize / 2;
    const double inv = -1.0 / (2.0 * sigma * sigma);

    std::vector<double> g(ksize);
    double sum = 0.0;
    for (int i = 0; i < ksize; ++i)
    {
        const double d = i - r;
        g[i] = std::exp(d * d * inv); // g(i) = exp(-d^2 / (2 sigma^2))
        sum += g[i];
    }
    std::vector<float> k(ksize);
    for (int i = 0; i < ksize; ++i) k[i] = static_cast<float>(g[i] / sum); // нормировка: sum k = 1
    return k;
}

// Округление float -> uint8 с насыщением. Значения всегда >= 0 (веса и пиксели неотрицательны).
static inline uint8_t to_u8(float v)
{
    int i = static_cast<int>(v + 0.5f);
    return static_cast<uint8_t>(i > 255 ? 255 : i);
}

// ---------------------------------------------------------------------------------------------
// Проход по строкам (горизонтальная 1D-свёртка).
//   out[x] = sum_{i=0}^{k-1} kx[i] * p[x + i]
// p — начало строки в УЖЕ дополненном (padded) изображении, поэтому p[x + i] соответствует
// исходному пикселю x + i - r, т.е. окну, центрированному в x. x0 — с какого x начать (хвост).
// ---------------------------------------------------------------------------------------------
static void hpass_scalar(const uint8_t* p, float* out, int w, const float* kx, int k, int x0)
{
    for (int x = x0; x < w; ++x)
    {
        float a = 0.f;
        for (int i = 0; i < k; ++i) a += kx[i] * p[x + i];
        out[x] = a;
    }
}

// Проход по столбцам (вертикальная 1D-свёртка) для строки y результата:
//   out[x] = to_u8( sum_{j=0}^{k-1} ky[j] * tmp[(y + j) * w + x] )
static void vpass_scalar(const float* tmp, uint8_t* out, int w, int y, const float* ky, int k, int x0)
{
    for (int x = x0; x < w; ++x)
    {
        float a = 0.f;
        for (int j = 0; j < k; ++j) a += ky[j] * tmp[static_cast<size_t>(y + j) * w + x];
        out[x] = to_u8(a);
    }
}

#if defined(__ARM_NEON)
// NEON: обрабатываем 8 соседних x за итерацию. Для каждого весa ядра:
//   uint8x8 -> uint16x8 -> 2 x uint32x4 -> 2 x float32x4, затем FMA: acc += f * kx[i].
// Векторизация идёт ПО x (соседние выходные пиксели независимы), а не по ядру — поэтому
// размер ядра не влияет на эффективность и нет горизонтальных редукций.
static void hpass_simd(const uint8_t* p, float* out, int w, const float* kx, int k)
{
    int x = 0;
    for (; x + 8 <= w; x += 8)
    {
        float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
        for (int i = 0; i < k; ++i)
        {
            const uint16x8_t v16 = vmovl_u8(vld1_u8(p + x + i)); // 8 пикселей, невыровненная загрузка
            const float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v16)));
            const float32x4_t f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v16)));
            a0 = vfmaq_n_f32(a0, f0, kx[i]);
            a1 = vfmaq_n_f32(a1, f1, kx[i]);
        }
        vst1q_f32(out + x, a0);
        vst1q_f32(out + x + 4, a1);
    }
    hpass_scalar(p, out, w, kx, k, x); // хвост (< 8 пикселей)
}

static void vpass_simd(const float* tmp, uint8_t* out, int w, int y, const float* ky, int k)
{
    int x = 0;
    for (; x + 8 <= w; x += 8)
    {
        float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
        for (int j = 0; j < k; ++j)
        {
            const float* t = tmp + static_cast<size_t>(y + j) * w + x;
            a0 = vfmaq_n_f32(a0, vld1q_f32(t), ky[j]);
            a1 = vfmaq_n_f32(a1, vld1q_f32(t + 4), ky[j]);
        }
        // float -> int32 (с округлением к ближайшему) -> int16 (saturate) -> uint8 (saturate)
        const int16x8_t s16 = vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(a0)), vqmovn_s32(vcvtnq_s32_f32(a1)));
        vst1_u8(out + x, vqmovun_s16(s16));
    }
    vpass_scalar(tmp, out, w, y, ky, k, x);
}
#endif

GrayImage gaussian_blur(const GrayImage& src, int ksize, double sigmaX, double sigmaY,
                        GaussMethod method, Border border)
{
    assert(ksize % 2 == 1);
    if (sigmaY <= 0.0) sigmaY = sigmaX;

    const std::vector<float> kx = gaussian_kernel_1d(ksize, sigmaX);
    const std::vector<float> ky = gaussian_kernel_1d(ksize, sigmaY);

    const int r = ksize / 2;
    const GrayImage padded = pad_image(src, r, border);
    GrayImage out(src.w, src.h);

    if (method == GaussMethod::Naive2D)
    {
        // 2D-ядро — внешнее произведение 1D-ядер (Гаусс сепарабелен):
        //   G(i, j) = ky[j] * kx[i]
        // Результат: dst(x, y) = sum_j sum_i G(i, j) * src(x + i - r, y + j - r)
        std::vector<float> k2(static_cast<size_t>(ksize) * ksize);
        for (int j = 0; j < ksize; ++j)
            for (int i = 0; i < ksize; ++i) k2[j * ksize + i] = ky[j] * kx[i];

        for (int y = 0; y < src.h; ++y)
        {
            uint8_t* o = out.row(y);
            for (int x = 0; x < src.w; ++x)
            {
                float a = 0.f;
                for (int j = 0; j < ksize; ++j)
                {
                    const uint8_t* p = padded.row(y + j) + x;
                    const float* kr = &k2[j * ksize];
                    for (int i = 0; i < ksize; ++i) a += kr[i] * p[i];
                }
                o[x] = to_u8(a);
            }
        }
        return out;
    }

    // Сепарабельный вариант: G = ky * kx^T  =>  blur = V(H(src)).
    // Стоимость 2k вместо k^2 умножений на пиксель.
    // tmp хранит результат горизонтального прохода для ВСЕХ строк padded (h + 2r строк),
    // т.к. вертикальному проходу нужны строки y..y+k-1.
    std::vector<float> tmp(static_cast<size_t>(padded.h) * src.w);

    const bool simd =
#if defined(__ARM_NEON)
        (method == GaussMethod::SeparableSIMD);
#else
        false;
#endif

    for (int yy = 0; yy < padded.h; ++yy)
    {
        float* t = tmp.data() + static_cast<size_t>(yy) * src.w;
#if defined(__ARM_NEON)
        if (simd) { hpass_simd(padded.row(yy), t, src.w, kx.data(), ksize); continue; }
#endif
        hpass_scalar(padded.row(yy), t, src.w, kx.data(), ksize, 0);
    }
    for (int y = 0; y < src.h; ++y)
    {
#if defined(__ARM_NEON)
        if (simd) { vpass_simd(tmp.data(), out.row(y), src.w, y, ky.data(), ksize); continue; }
#endif
        vpass_scalar(tmp.data(), out.row(y), src.w, y, ky.data(), ksize, 0);
    }
    (void)simd;
    return out;
}
