#pragma once
// Только для .mm (Objective-C++). Помощники отрисовки, вынесенные из CVLabView.mm (ЛР1), чтобы
// переиспользовать их в других лабораторных: загрузка cv::Mat в Metal-текстуру и график серий бенчмарка.

#import <Metal/Metal.h>

#include <vector>
#include <opencv2/imgproc.hpp>

#include "imgui.h"
#include "implot.h"

#include "../lab1_gaussian_filter/CVLab.h" // BenchSeries

namespace gpu {

struct Tex
{
    id<MTLTexture> tex = nil;
    int w = 0, h = 0;
};

// cv::Mat (1 канал серый или 3 канала BGR) -> RGBA8-текстура Metal. Текстура пересоздаётся при смене размера.
inline void upload(id<MTLDevice> dev, Tex& t, const cv::Mat& img)
{
    cv::Mat rgba;
    cv::cvtColor(img, rgba, img.channels() == 1 ? cv::COLOR_GRAY2RGBA : cv::COLOR_BGR2RGBA);
    if (t.tex == nil || t.w != img.cols || t.h != img.rows)
    {
        MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                      width:img.cols
                                                                                     height:img.rows
                                                                                  mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead;
        t.tex = [dev newTextureWithDescriptor:d];
        t.w = img.cols;
        t.h = img.rows;
    }
    [t.tex replaceRegion:MTLRegionMake2D(0, 0, img.cols, img.rows)
             mipmapLevel:0
               withBytes:rgba.data
             bytesPerRow:rgba.step];
}

// ImGui 1.92+: ImTextureID = ImU64, в старых версиях void*; каст через intptr_t подходит обоим.
inline ImTextureID tex_id(const Tex& t) { return (ImTextureID)(intptr_t)(__bridge void*)t.tex; }

// График нескольких серий (время в логарифмической шкале: значения различаются на порядки).
inline void plot_series(const char* title, const char* xlabel, const std::vector<BenchSeries>& v,
                        float width, float height = 300.f)
{
    if (ImPlot::BeginPlot(title, ImVec2(width, height)))
    {
        ImPlot::SetupAxes(xlabel, "ms (log)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
        for (const auto& s : v)
            if (!s.x.empty()) ImPlot::PlotLine(s.name.c_str(), s.x.data(), s.ms.data(), (int)s.x.size());
        ImPlot::EndPlot();
    }
}

} // namespace gpu
