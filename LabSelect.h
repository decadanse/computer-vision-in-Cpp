#pragma once
// Выбор активной лабораторной: раскомментируйте ровно ОДНУ строку.
// Новые лабораторные добавляются так: новый #define здесь + новая ветка #elif в Renderer.mm.

#define USE_CV_GAUSS_PIPELINE        // Усредняющий фильтр Гаусса (OpenCV vs нативно)
//#define USE_CV_TRACKING_PIPELINE
