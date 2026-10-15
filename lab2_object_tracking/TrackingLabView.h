#pragma once
// Только для .mm (Objective-C++).
#import <Metal/Metal.h>
#include "TrackingLab.h"

// Окно "Object tracking": видео с рамкой и подписью, управление воспроизведением, параметры трекера,
// статистика, графики по кадрам и бенчмарк. Вызывать между ImGui::NewFrame() и ImGui::Render().
void DrawTrackingLab(TrackingLabState& state, id<MTLDevice> device);
