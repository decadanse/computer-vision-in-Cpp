#import "Renderer.h"

#include <thread>

#include "imgui.h"
#include "backends/imgui_impl_osx.h"
#include "backends/imgui_impl_metal.h"
#include "implot.h"

#include "LabSelect.h"

#if defined USE_CV_GAUSS_PIPELINE
#include "./lab1_gaussian_filter/CVLab.h"
#include "./lab1_gaussian_filter/CVLabView.h"
#elif defined USE_CV_TRACKING_PIPELINE
// #include "...": состояние и воркер MLP
#elif defined USE_CNN_PIPELINE
// #include "...": состояние и воркер CNN
#endif

@interface Renderer ()
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@end

@implementation Renderer {
    // Состояние активной лабораторной (пишет фоновый поток, читает поток рендера)
#if defined USE_CV_GAUSS_PIPELINE
    CVLabState _cv;
#elif defined USE_CV_TRACKING_PIPELINE
#elif defined USE_CNN_PIPELINE
#endif
    std::thread _workerThread;
}

- (instancetype)initWithView:(MTKView *)view
{
    self = [super init];
    if (self)
    {
        _device = view.device;
        _commandQueue = [_device newCommandQueue];

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();

        ImGuiIO &io = ImGui::GetIO();
        ImFontConfig config;
        config.OversampleH = 2;
        io.Fonts->AddFontFromFileTTF(
            "/System/Library/Fonts/Supplemental/Arial.ttf", // любой .ttf с кириллицей, есть в системе
            16.0f, &config, io.Fonts->GetGlyphRangesCyrillic());

        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

        ImGui::StyleColorsDark();

        ImGui_ImplMetal_Init(_device);
        ImGui_ImplOSX_Init(view);

        // окно уже показано — фоновая работа стартует сразу, UI ничего не блокирует
#if defined USE_CV_GAUSS_PIPELINE
        _workerThread = std::thread(RunCVLab, std::ref(_cv));
#elif defined USE_CV_TRACKING_PIPELINE
#endif
    }
    return self;
}

- (void)dealloc
{
    // на выходе из приложения просим поток остановиться и дожидаемся его
#if defined USE_CV_GAUSS_PIPELINE
    _cv.stop.store(true);
#endif
    if (_workerThread.joinable())
        _workerThread.join();
}

- (void)drawInMTKView:(MTKView *)view
{
    ImGuiIO &io = ImGui::GetIO();
    io.DisplaySize = ImVec2(view.bounds.size.width, view.bounds.size.height);

    CGFloat framebufferScale = view.window.screen.backingScaleFactor ?: NSScreen.mainScreen.backingScaleFactor;
    io.DisplayFramebufferScale = ImVec2(framebufferScale, framebufferScale);

    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];

    MTLRenderPassDescriptor *renderPassDescriptor = view.currentRenderPassDescriptor;
    if (renderPassDescriptor == nil)
    {
        [commandBuffer commit];
        return;
    }

    id<MTLRenderCommandEncoder> renderEncoder =
        [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];

    ImGui_ImplMetal_NewFrame(renderPassDescriptor);
    ImGui_ImplOSX_NewFrame(view);
    ImGui::NewFrame();

    ImGui::Begin("TEST");
    ImGui::Text("Renderer is running!");
    ImGui::End();
    
    // --- UI активной лабораторной. Каждая ветка сама снимает копию данных под своим мьютексом ---
#if defined USE_CV_GAUSS_PIPELINE
    DrawCVLab(_cv, _device);
#elif defined USE_CV_TRACKING_PIPELINE
    // сюда: снимок состояния под мьютексом + ImGui/ImPlot для MLP
#else
    ImGui::Begin("Lab");
    ImGui::Text("Лабораторная не выбрана - раскомментируйте define в LabSelect.h");
    ImGui::End();
#endif

    ImGui::Render();
    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), commandBuffer, renderEncoder);

    [renderEncoder endEncoding];
    [commandBuffer presentDrawable:view.currentDrawable];
    [commandBuffer commit];
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {}

@end
