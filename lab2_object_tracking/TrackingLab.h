#pragma once
// Состояние ЛР2 (трекинг по ключевым точкам). Пишет фоновый поток (RunTrackingLab), читает поток рендера.
// Схема та же, что в ЛР1: всё под mtx, флаги — atomic, UI копирует данные только при смене версии.

#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

#include "Tracker.h"
#include "../lab1_gaussian_filter/CVLab.h" // BenchSeries

struct TrackStats
{
    int frames = 0;
    int found = 0;
    int n_orb = 0, n_flow = 0, n_held = 0; // сколько кадров найдено поиском / потоком / удержано прошлой рамкой
    double mean_ms = 0.0;      // среднее время process() на кадр
    // только для синтетической последовательности (есть истинная рамка):
    bool has_gt = false;
    int visible = 0;           // кадров, где объект есть
    int found_visible = 0;     // из них найдено
    int false_positive = 0;    // найдено там, где объекта нет
    double mean_iou = 0.0;     // средний IoU по найденным видимым кадрам
};

struct TrackingLabState
{
    std::mutex mtx;
    std::atomic<bool> loaded{false};        // кадры загружены
    std::atomic<bool> tracking_done{false}; // результаты трекинга валидны
    std::atomic<bool> bench_done{false};
    std::atomic<bool> stop{false};
    std::atomic<bool> reload{false};        // UI просит загрузить другое видео
    std::atomic<bool> rerun{false};         // UI просит пересчитать трекинг с новыми параметрами
    std::atomic<bool> rebench{false};       // UI просит повторить бенчмарк
    std::atomic<int>  progress{0};          // обработано кадров в текущем прогоне
    std::atomic<int>  results_version{0};   // растёт при каждом новом наборе результатов

    // Входные параметры (пишет UI, читает воркер) — под mtx
//    std::string video_path = "/Users/h4s/Dev/0_masters_degree_datasets/cv/test-videos/mona-lisa-blur.m4v";
    std::string video_path = "/Users/h4s/Dev/0_masters_degree_datasets/cv/test-videos/mona-lisa-blur-extra-credit.m4v";
    TrackerParams params;

    // Данные (пишет воркер) — под mtx
    std::string source_note;
    std::vector<cv::Mat> frames;                         // BGR, уменьшены до kMaxWidth; кадр 0 = эталон
    std::vector<TrackResult> results;                    // по кадру
    std::vector<double> iou;                             // по кадру; -1 если нет истины/объект не виден
    int ref_keypoints = 0;
    TrackStats stats;

    // Бенчмарк: 5 серий = поиск ORB: detect / match / RANSAC / итого поиска + один шаг оптического потока
    std::vector<BenchSeries> stage_vs_features; // x = nfeatures
    std::vector<BenchSeries> stage_vs_width;    // x = ширина кадра, px
};

void RunTrackingLab(TrackingLabState& state); // вызывается в std::thread
