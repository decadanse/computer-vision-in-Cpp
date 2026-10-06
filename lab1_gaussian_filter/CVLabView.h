#pragma once
// Только для .mm (Objective-C++): использует Metal-типы для загрузки текстур в ImGui.
#import <Metal/Metal.h>
#include "CVLab.h"

// Рисует окно "Gaussian filter": интерактивный просмотр (метод, ksize, sigmaX, sigmaY),
// график ядра, таблицу проверки против OpenCV и графики бенчмарка.
// Вызывать между ImGui::NewFrame() и ImGui::Render().
void DrawCVLab(CVLabState& state, id<MTLDevice> device);
