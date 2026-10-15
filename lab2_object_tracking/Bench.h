#pragma once
// Общие помощники бенчмарка (те же приёмы, что в ЛР1 / CVLab.cpp), вынесены в заголовок,
// чтобы их можно было переиспользовать без копирования. Всё inline, в namespace bench.
//
// Из ЛР1 берутся без изменений: BenchSeries (структура серии) и output_path (абсолютный путь вывода).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "../lab1_gaussian_filter/CVLab.h" // BenchSeries, output_path

namespace bench {

// Защита от «выкидывания» вычислений компилятором: результат записываем сюда.
inline volatile uint8_t sink = 0;

// Медиана устойчивее среднего к одиночным выбросам (прерывания ОС, троттлинг).
inline double median(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Медиана времени (мс) по reps запускам + 1 прогревочный (кэш, ленивые аллокации).
template <class F>
double time_ms(F&& f, int reps)
{
    using clock = std::chrono::steady_clock;
    f(); // прогрев
    std::vector<double> t(reps);
    for (double& v : t)
    {
        const auto a = clock::now();
        f();
        const auto b = clock::now();
        v = std::chrono::duration<double, std::milli>(b - a).count();
    }
    return median(std::move(t));
}

// Набор пустых серий с именами (порядок имён = индексы для push).
inline std::vector<BenchSeries> make_series(std::initializer_list<const char*> names)
{
    std::vector<BenchSeries> v;
    for (const char* n : names) { BenchSeries s; s.name = n; v.push_back(s); }
    return v;
}

// Добавляет точку (x, ms) в серию i под мьютексом состояния: график в UI растёт «вживую».
// State — любая структура с полем std::mutex mtx (CVLabState, TrackingLabState, ...).
template <class State>
void push(State& s, std::vector<BenchSeries>& v, size_t i, double x, double ms)
{
    std::lock_guard<std::mutex> lock(s.mtx);
    v[i].x.push_back(x);
    v[i].ms.push_back(ms);
}

// CSV формата method,x,ms в ~/Desktop/gaussian_lab/ (см. output_path); полный путь печатается в консоль.
inline void write_csv(const std::string& file, const std::vector<BenchSeries>& v)
{
    std::ofstream out(file);
    if (!out) { std::cerr << "bench: CANNOT WRITE " << file << "\n"; return; }
    std::cerr << "bench: writing " << file << "\n";
    out << "method,x,ms\n";
    for (const auto& s : v)
        for (size_t i = 0; i < s.x.size(); ++i) out << s.name << "," << s.x[i] << "," << s.ms[i] << "\n";
}

} // namespace bench
