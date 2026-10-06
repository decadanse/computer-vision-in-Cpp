#include "CVLab.h"

#include "Gaussian.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>

namespace {

// Путь к входному изображению. Если файла нет — строится синтетическое (градиент + текст + шум):
// острые края текста и шум хорошо показывают, как разные ksize и sigma сглаживают картинку.

//const char* kInputPath = "/input.jpg";
//const String kInputPath = "/input.jpg";
const std::filesystem::path kInputPath =
    std::filesystem::path(__FILE__).parent_path() / "input.jpg";

volatile uint8_t g_sink = 0; // чтобы компилятор не выкинул вычисления в бенчмарке

// Медиана времени (мс) по reps запусков + 1 прогревочный. Медиана устойчивее среднего к выбросам.
template <class F>
double time_ms(F&& f, int reps)
{
    using clock = std::chrono::steady_clock;
    f(); // прогрев (кэш, ленивые аллокации)
    std::vector<double> t(reps);
    for (double& v : t)
    {
        const auto a = clock::now();
        f();
        const auto b = clock::now();
        v = std::chrono::duration<double, std::milli>(b - a).count();
    }
    std::sort(t.begin(), t.end());
    return t[reps / 2];
}

cv::Mat make_synthetic()
{
    cv::Mat f(1080, 1920, CV_32F);
    for (int y = 0; y < f.rows; ++y)
        for (int x = 0; x < f.cols; ++x)
            f.at<float>(y, x) = 70.f + 150.f * x / f.cols; // плавный градиент яркости

    cv::Mat mask = cv::Mat::zeros(f.size(), CV_8U);
    cv::putText(mask, "Gaussian Blur", {80, 360}, cv::FONT_HERSHEY_SIMPLEX, 5.0, cv::Scalar(255), 12);
    cv::putText(mask, "sigma = 1.5", {80, 760}, cv::FONT_HERSHEY_SIMPLEX, 5.0, cv::Scalar(255), 12);
    cv::subtract(f, cv::Scalar(55), f, mask); // «чернила» темнее фона

    cv::Mat noise(f.size(), CV_32F);
    cv::randn(noise, 0.0, 8.0); // шум, который фильтр должен подавлять
    f += noise;

    cv::Mat out;
    f.convertTo(out, CV_8U); // с насыщением
    return out;
}

VerifyRow compare(const std::string& name, const GrayImage& native, const cv::Mat& ref)
{
    const cv::Mat a = to_mat(native);
    cv::Mat d;
    cv::absdiff(a, ref, d);
    double mx = 0;
    cv::minMaxLoc(d, nullptr, &mx);
    VerifyRow r;
    r.name = name;
    r.max_abs_diff = static_cast<int>(mx);
    r.mismatch_pct = 100.0 * cv::countNonZero(d) / static_cast<double>(d.total());
    return r;
}

void push(CVLabState& s, std::vector<BenchSeries>& v, size_t i, double x, double ms)
{
    std::lock_guard<std::mutex> lock(s.mtx);
    v[i].x.push_back(x);
    v[i].ms.push_back(ms);
}

void write_csv(const std::string& file, const std::vector<BenchSeries>& v)
{
    std::ofstream out(file);
    out << "method,x,ms\n";
    for (const auto& s : v)
        for (size_t i = 0; i < s.x.size(); ++i) out << s.name << "," << s.x[i] << "," << s.ms[i] << "\n";
}

std::vector<BenchSeries> make_series()
{
    std::vector<BenchSeries> v(4);
    v[0].name = "native 2D";
    v[1].name = "native separable";
    v[2].name = "native separable SIMD";
    v[3].name = "OpenCV";
    return v;
}

// Одна точка бенчмарка: время всех 4 реализаций на изображении m с ядром k (sigma — авто).
void bench_point(CVLabState& s, std::vector<BenchSeries>& v, double x, const cv::Mat& m, int k)
{
    const GrayImage g = from_mat(m);
    // Наивная 2D свёртка при k > 31 занимает секунды на кадр — пропускаем, чтобы бенчмарк не висел
    if (k <= 31)
        push(s, v, 0, x, time_ms([&] { g_sink = gaussian_blur(g, k, 0, 0, GaussMethod::Naive2D).px[0]; }, 3));
    push(s, v, 1, x, time_ms([&] { g_sink = gaussian_blur(g, k, 0, 0, GaussMethod::Separable).px[0]; }, 7));
    push(s, v, 2, x, time_ms([&] { g_sink = gaussian_blur(g, k, 0, 0, GaussMethod::SeparableSIMD).px[0]; }, 7));
    push(s, v, 3, x, time_ms([&] { g_sink = gaussian_blur_opencv(m, k, 0, 0).at<uint8_t>(0, 0); }, 7));
}

cv::Mat resize_to_width(const cv::Mat& img, int width)
{
    cv::Mat out;
    cv::resize(img, out, cv::Size(width, width * img.rows / img.cols), 0, 0, cv::INTER_AREA);
    return out;
}

} // namespace

cv::Mat gaussian_blur_opencv(const cv::Mat& gray, int ksize, double sigmaX, double sigmaY)
{
    cv::Mat dst;
    cv::GaussianBlur(gray, dst, cv::Size(ksize, ksize), sigmaX, sigmaY, cv::BORDER_REFLECT_101);
    return dst;
}

void RunCVLab(CVLabState& s)
{
    // 1. Загрузка
    cv::Mat img = cv::imread(kInputPath, cv::IMREAD_GRAYSCALE);
//    cv::Mat img = cv::imread(kInputPath.string(), cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
        std::cerr << "OpenCV failed to load: "
                  << kInputPath << '\n';
    }
    else
    {
        std::cout << "Loaded: "
                  << img.cols << "x" << img.rows << '\n';
    }
    const bool synthetic = img.empty();
    if (synthetic) img = make_synthetic();
    {
        std::lock_guard<std::mutex> lock(s.mtx);
        s.src = img.clone();
        s.source_note = synthetic ? "синтетическое изображение (файл не найден)" : kInputPath;
    }
    s.loaded.store(true);
    std::cerr << "cv: image " << img.cols << "x" << img.rows << (synthetic ? " (synthetic)" : "") << "\n";

    // 2. Проверка корректности на разных (ksize, sigmaX, sigmaY): нативная реализация против OpenCV.
    //    Кадр уменьшаем до 640 px, чтобы наивная свёртка не тормозила.
    {
        const cv::Mat small = resize_to_width(img, 640);
        const GrayImage g = from_mat(small);
        struct Case { int k; double sx, sy; };
        const Case cases[] = {{3, 0, 0}, {7, 1.5, 1.5}, {15, 0, 0}, {15, 3, 1}, {31, 5, 5}, {51, 0, 0}};
        std::vector<VerifyRow> rows;
        for (const Case& c : cases)
        {
            const cv::Mat ref = gaussian_blur_opencv(small, c.k, c.sx, c.sy);
            char tag[64];
            snprintf(tag, sizeof(tag), "k=%d sigma=%.1f/%.1f", c.k, c.sx, c.sy);
            if (c.k <= 31)
                rows.push_back(compare(std::string(tag) + " | 2D", gaussian_blur(g, c.k, c.sx, c.sy, GaussMethod::Naive2D), ref));
            rows.push_back(compare(std::string(tag) + " | separable", gaussian_blur(g, c.k, c.sx, c.sy, GaussMethod::Separable), ref));
            rows.push_back(compare(std::string(tag) + " | SIMD", gaussian_blur(g, c.k, c.sx, c.sy, GaussMethod::SeparableSIMD), ref));
        }
        std::lock_guard<std::mutex> lock(s.mtx);
        s.verify = rows;
    }

    // 3. Бенчмарки. Каждая точка сразу пушится под мьютексом -> графики растут вживую.
    {
        std::lock_guard<std::mutex> lock(s.mtx);
        s.gauss_vs_k = make_series();
        s.gauss_vs_size = make_series();
    }

    const cv::Mat base = resize_to_width(img, 1280); // фиксированный кадр для зависимости от размера ядра
    for (int k : {3, 5, 7, 9, 11, 15, 21, 31, 41, 51})
    {
        if (s.stop) return;
        bench_point(s, s.gauss_vs_k, k, base, k);
    }
    for (int wd : {320, 640, 960, 1280, 1920})
    {
        if (s.stop) return;
        bench_point(s, s.gauss_vs_size, wd, resize_to_width(img, wd), 15); // фиксированное ядро 15
    }

    {
        std::lock_guard<std::mutex> lock(s.mtx);
        write_csv("bench_gauss_vs_ksize.csv", s.gauss_vs_k);
        write_csv("bench_gauss_vs_size.csv", s.gauss_vs_size);
    }
    std::cerr << "cv: benchmark done\n";
    s.bench_done.store(true);
}
