#import "CVLabView.h"

#include "imgui.h"
#include "implot.h"

#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "Gaussian.h"

namespace {

struct GpuTex
{
    id<MTLTexture> tex = nil;
    int w = 0, h = 0;
};

// Серое изображение -> RGBA8-текстура Metal (ImGui::Image ожидает цветную текстуру).
void upload(id<MTLDevice> dev, GpuTex& t, const cv::Mat& gray)
{
    cv::Mat rgba;
    cv::cvtColor(gray, rgba, cv::COLOR_GRAY2RGBA);
    if (t.tex == nil || t.w != gray.cols || t.h != gray.rows)
    {
        MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                      width:gray.cols
                                                                                     height:gray.rows
                                                                                  mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead;
        t.tex = [dev newTextureWithDescriptor:d];
        t.w = gray.cols;
        t.h = gray.rows;
    }
    [t.tex replaceRegion:MTLRegionMake2D(0, 0, gray.cols, gray.rows)
             mipmapLevel:0
               withBytes:rgba.data
             bytesPerRow:rgba.step];
}

// Приведение к ImTextureID: ImGui 1.92+ — ImU64, в старых версиях — void*; каст через intptr_t подходит обоим.
ImTextureID tex_id(const GpuTex& t) { return (ImTextureID)(intptr_t)(__bridge void*)t.tex; }

void plot_series(const char* title, const char* xlabel, const std::vector<BenchSeries>& v, float width)
{
    if (ImPlot::BeginPlot(title, ImVec2(width, 300)))
    {
        ImPlot::SetupAxes(xlabel, "ms (log)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10); // времена различаются на порядки
        for (const auto& s : v)
            if (!s.x.empty()) ImPlot::PlotLine(s.name.c_str(), s.x.data(), s.ms.data(), (int)s.x.size());
        ImPlot::EndPlot();
    }
}

} // namespace

void DrawCVLab(CVLabState& s, id<MTLDevice> device)
{
    static cv::Mat src, out, diff_vis;
    static GrayImage src_g;
    static GpuTex t_src, t_out, t_diff;
    static int method = 2, ksize = 15, max_diff = 0;
    static float sigma_x = 0.f, sigma_y = 0.f;
    static bool dirty = true, skipped = false;
    static double ms_native = 0.0, ms_cv = 0.0;

    if (src.empty() && s.loaded.load())
    {
        std::lock_guard<std::mutex> lock(s.mtx);
        src = s.src.clone();
        src_g = from_mat(src);
        upload(device, t_src, src);
    }

    ImGui::Begin("Gaussian filter");
    if (src.empty()) { ImGui::Text("Загрузка изображения..."); ImGui::End(); return; }

    // ---- параметры. Порядок элементов Combo совпадает с enum GaussMethod ----
    dirty |= ImGui::Combo("Метод", &method, "2D naive\0Separable\0Separable SIMD (NEON)\0");
    dirty |= ImGui::SliderInt("Размер ядра k (нечётный)", &ksize, 3, 101);
    ksize |= 1;
    dirty |= ImGui::SliderFloat("sigmaX (0 = авто)", &sigma_x, 0.f, 30.f);
    dirty |= ImGui::SliderFloat("sigmaY (0 = как sigmaX)", &sigma_y, 0.f, 30.f);

    const double eff_sx = sigma_x > 0.f ? sigma_x : auto_sigma(ksize);
    const double eff_sy = sigma_y > 0.f ? sigma_y : eff_sx;
    ImGui::Text("Эффективные sigmaX = %.3f, sigmaY = %.3f", eff_sx, eff_sy);

    // Наивная 2D свёртка стоит ~ W*H*k^2: на больших k UI подвис бы на секунды
    const bool too_slow = method == 0 && static_cast<double>(src.total()) * ksize * ksize > 2e9;
    if (dirty)
    {
        skipped = too_slow;
        if (!too_slow)
        {
            using clk = std::chrono::steady_clock;
            auto t0 = clk::now();
            const GrayImage r = gaussian_blur(src_g, ksize, sigma_x, sigma_y, static_cast<GaussMethod>(method), Border::Reflect101);
            ms_native = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            out = to_mat(r);

            t0 = clk::now();
            const cv::Mat ref = gaussian_blur_opencv(src, ksize, sigma_x, sigma_y);
            ms_cv = std::chrono::duration<double, std::milli>(clk::now() - t0).count();

            cv::Mat d;
            cv::absdiff(out, ref, d);
            double mx = 0;
            cv::minMaxLoc(d, nullptr, &mx);
            max_diff = static_cast<int>(mx);
            d.convertTo(diff_vis, CV_8U, 40.0); // разность x40, иначе +-1 не видно

            upload(device, t_out, out);
            upload(device, t_diff, diff_vis);
        }
        dirty = false;
    }

    if (skipped)
        ImGui::TextColored(ImVec4(1, 0.6f, 0.2f, 1), "2D naive при таком k слишком долго - выберите Separable или уменьшите k");
    else
        ImGui::Text("Нативно: %.2f мс | OpenCV: %.2f мс | max|diff| = %d (одиночный замер)", ms_native, ms_cv, max_diff);
    if (ImGui::Button("Сохранить PNG") && !out.empty())
        cv::imwrite("gauss_k" + std::to_string(ksize) + "_m" + std::to_string(method) + ".png", out);

    // ---- три изображения: исходное / результат / |нативное - OpenCV| x40 ----
    const float third = (ImGui::GetContentRegionAvail().x - 16.f) / 3.f;
    const ImVec2 sz(third, third * src.rows / src.cols);
    ImGui::Image(tex_id(t_src), sz);
    if (t_out.tex != nil)
    {
        ImGui::SameLine(); ImGui::Image(tex_id(t_out), sz);
        ImGui::SameLine(); ImGui::Image(tex_id(t_diff), sz);
    }

    // ---- график 1D-ядра: веса k(i) = g(i) / sum g ----
    {
        const std::vector<float> kern = gaussian_kernel_1d(ksize, sigma_x);
        std::vector<double> xs(kern.size()), ys(kern.size());
        for (size_t i = 0; i < kern.size(); ++i) { xs[i] = double(i) - double(kern.size() / 2); ys[i] = kern[i]; }
        if (ImPlot::BeginPlot("1D-ядро Гаусса (sigmaX)", ImVec2(-1, 160)))
        {
            ImPlot::SetupAxes("смещение i - r", "вес", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotBars("k(i)", xs.data(), ys.data(), (int)xs.size(), 0.8);
            ImPlot::EndPlot();
        }
    }

    // ---- копия данных фонового потока под мьютексом ----
    std::vector<VerifyRow> verify;
    std::vector<BenchSeries> gk, gs;
    {
        std::lock_guard<std::mutex> lock(s.mtx);
        verify = s.verify; gk = s.gauss_vs_k; gs = s.gauss_vs_size;
    }

    ImGui::Separator();
    ImGui::Text("Проверка: нативная реализация vs cv::GaussianBlur (кадр 640 px)");
    for (const auto& v : verify)
        ImGui::Text("%-36s max|diff| = %d, несовпадение = %.3f%%", v.name.c_str(), v.max_abs_diff, v.mismatch_pct);

    ImGui::Separator();
    ImGui::Text(s.bench_done.load() ? "Бенчмарк завершён (CSV сохранены)" : "Бенчмарк идёт...");
    const float pw = (ImGui::GetContentRegionAvail().x - 8.f) * 0.5f;
    plot_series("Время vs размер ядра (кадр 1280 px)", "ksize", gk, pw);
    ImGui::SameLine();
    plot_series("Время vs ширина кадра (k = 15)", "width, px", gs, pw);

    ImGui::End();
}
