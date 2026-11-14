#include "TrackingLab.h"

#include "Bench.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <thread>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace {

constexpr int kMaxFrames = 240; // ограничение памяти: <= ~370 МБ при 960x540 BGR
constexpr int kMaxWidth  = 960; // кадры шире уменьшаются (трекинг идёт на уменьшенных)

// ---------------------------------------------------------------------------------------------
// Синтетическая последовательность с ИЗВЕСТНОЙ истинной рамкой: если видео не найдено, пайплайн всё
// равно работает, а качество можно измерить (IoU). Кадр 0 — «крупное изображение объекта», далее
// объект движется, вращается и наклоняется на текстурированном фоне; кадры 90..99 — объекта нет.
// ---------------------------------------------------------------------------------------------
cv::Mat make_texture(cv::Size sz, uint64 seed, bool object)
{
    cv::RNG rng(seed);
    cv::Mat n(sz, CV_8UC3);
    rng.fill(n, cv::RNG::UNIFORM, 0, 255);

    cv::Mat fine, coarse;
    cv::GaussianBlur(n, fine, cv::Size(0, 0), 1.5);
    fine.convertTo(fine, CV_8U, 3.0, -256.0);                 // 3*(v-128)+128: вернуть контраст после размытия
    cv::GaussianBlur(n, coarse, cv::Size(0, 0), 7.0);
    cv::normalize(coarse, coarse, 0, 255, cv::NORM_MINMAX);
    cv::Mat tex;
    cv::addWeighted(fine, 0.6, coarse, 0.4, 0, tex);

    if (object)
    {
        for (int i = 0; i < 25; ++i) // фигуры и надписи: даёт много углов для FAST
        {
            const cv::Scalar c(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            const cv::Point p(rng.uniform(0, sz.width), rng.uniform(0, sz.height));
            if (i % 3 == 0) cv::rectangle(tex, p, p + cv::Point(rng.uniform(30, 150), rng.uniform(30, 120)), c, -1);
            else if (i % 3 == 1) cv::circle(tex, p, rng.uniform(15, 70), c, -1);
            else cv::putText(tex, "TRACK" + std::to_string(i), p, cv::FONT_HERSHEY_SIMPLEX, 1.6, c, 3);
        }
        cv::rectangle(tex, cv::Point(0, 0), cv::Point(sz.width - 1, sz.height - 1), cv::Scalar(20, 20, 20), 6);
    }
    return tex;
}

void make_synthetic(std::vector<cv::Mat>& frames, std::vector<std::array<cv::Point2f, 4>>& gt, std::vector<char>& visible)
{
    const cv::Size sz(960, 540);
    const int N = 120;
    const cv::Mat ref = make_texture(sz, 1234, true);
    const cv::Mat bg = make_texture(sz, 99, false);
    cv::RNG noise_rng(7);
    const std::vector<cv::Point2f> ref_c = {{0, 0}, {float(sz.width), 0}, {float(sz.width), float(sz.height)}, {0, float(sz.height)}};

    for (int t = 0; t < N; ++t)
    {
        std::array<cv::Point2f, 4> q{};
        char vis = 1;
        cv::Mat frame;
        if (t == 0)
        {
            frame = ref.clone(); // первый кадр: объект во весь кадр
            for (int i = 0; i < 4; ++i) q[i] = ref_c[i];
        }
        else
        {
            const double ph = 2.0 * CV_PI * t / N;
            const double s = 0.42 + 0.10 * std::cos(1.5 * ph);          // масштаб
            const double th = 0.5 * std::sin(ph);                        // поворот, рад (~28 градусов)
            const double cx = sz.width * 0.5 + 220.0 * std::sin(ph);
            const double cy = sz.height * 0.5 + 90.0 * std::sin(2.0 * ph);
            const double hw = 0.5 * sz.width * s, hh = 0.5 * sz.height * s;
            const double cs = std::cos(th), sn = std::sin(th);
            const double sx[4] = {-1, 1, 1, -1}, sy[4] = {-1, -1, 1, 1};
            for (int i = 0; i < 4; ++i)
            {
                // перспективное искажение: угол смещается синусоидально (имитирует наклон объекта)
                const double ox = 0.07 * hw * std::sin(ph * 2.0 + i), oy = 0.07 * hh * std::cos(ph * 2.0 + 1.7 * i);
                const double x = sx[i] * hw + ox, y = sy[i] * hh + oy;
                q[i] = cv::Point2f(float(cx + cs * x - sn * y), float(cy + sn * x + cs * y));
            }
            frame = bg.clone();
            vis = !(t >= 90 && t < 100); // «объект ушёл из кадра»
            if (vis)
            {
                const cv::Mat H = cv::getPerspectiveTransform(ref_c, std::vector<cv::Point2f>(q.begin(), q.end()));
                cv::Mat warped, mask(sz, CV_8U, cv::Scalar(255)), mask_w;
                cv::warpPerspective(ref, warped, H, sz, cv::INTER_LINEAR);
                cv::warpPerspective(mask, mask_w, H, sz, cv::INTER_NEAREST);
                warped.copyTo(frame, mask_w);
            }
            cv::GaussianBlur(frame, frame, cv::Size(0, 0), 0.8);          // небольшая расфокусировка
            if (t >= 50 && t < 54) cv::GaussianBlur(frame, frame, cv::Size(0, 0), 8.0); // «смаз»: 4 кадра с сильным размытием
            cv::Mat nz(sz, CV_8UC3);
            noise_rng.fill(nz, cv::RNG::NORMAL, 0, 4);                     // шум сенсора, sigma=4
            cv::Mat f16;
            frame.convertTo(f16, CV_16SC3);
            nz.convertTo(nz, CV_16SC3);
            f16 += nz;
            f16.convertTo(frame, CV_8UC3);                                 // с насыщением
        }
        frames.push_back(frame);
        gt.push_back(q);
        visible.push_back(vis);
    }
}

bool load_video(const std::string& path, std::vector<cv::Mat>& frames)
{
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) return false;
    const double total = cap.get(cv::CAP_PROP_FRAME_COUNT);
    const int stride = total > kMaxFrames ? static_cast<int>(std::ceil(total / kMaxFrames)) : 1; // покрыть всё видео
    cv::Mat f;
    for (int i = 0; static_cast<int>(frames.size()) < kMaxFrames; ++i)
    {
        if (!cap.grab()) break;
        if (i % stride != 0) continue;
        if (!cap.retrieve(f) || f.empty()) break;
        if (f.cols > kMaxWidth) cv::resize(f, f, cv::Size(kMaxWidth, f.rows * kMaxWidth / f.cols), 0, 0, cv::INTER_AREA);
        frames.push_back(f.clone());
    }
    return frames.size() >= 2;
}

std::vector<std::array<cv::Point2f, 4>> g_gt;   // истина синтетики (только воркер-поток)
std::vector<char> g_visible;

void load_source(TrackingLabState& s)
{
    std::string path;
    {
        std::lock_guard<std::mutex> lock(s.mtx);
        path = s.video_path;
    }
    std::vector<cv::Mat> frames;
    g_gt.clear();
    g_visible.clear();
    std::string note;
    if (load_video(path, frames))
        note = path + " (" + std::to_string(frames.size()) + " кадров)";
    else
    {
        frames.clear();
        make_synthetic(frames, g_gt, g_visible);
        note = "синтетическая последовательность (видео не открылось: " + path + ")";
    }
    std::cerr << "track: source = " << note << ", " << frames[0].cols << "x" << frames[0].rows << "\n";
    {
        std::lock_guard<std::mutex> lock(s.mtx);
        s.frames = std::move(frames);
        s.source_note = note;
        s.results.clear();
        s.iou.clear();
    }
    s.tracking_done.store(false);
    s.loaded.store(true);
}

// Прогон трекера по всем кадрам + сводная статистика + CSV по кадрам.
void track_all(TrackingLabState& s)
{
    s.tracking_done.store(false);
    TrackerParams p;
    std::vector<cv::Mat> frames;
    {
        std::lock_guard<std::mutex> lock(s.mtx);
        p = s.params;
        frames = s.frames; // копии заголовков cv::Mat (данные не копируются)
    }
    ObjectTracker tracker;
    if (!tracker.init(frames[0], p))
        std::cerr << "track: в эталоне слишком мало ключевых точек - нужен более текстурный объект\n";

    std::vector<TrackResult> results(frames.size());
    std::vector<double> iou(frames.size(), -1.0);
    s.progress.store(0);
    for (size_t i = 0; i < frames.size(); ++i)
    {
        if (s.stop) return;
        results[i] = tracker.process(frames[i]);
        if (!g_gt.empty() && g_visible[i] && results[i].found) iou[i] = quad_iou(results[i].quad, g_gt[i]);
        s.progress.store(static_cast<int>(i) + 1);
    }

    TrackStats st;
    st.frames = static_cast<int>(frames.size());
    st.has_gt = !g_gt.empty();
    double iou_sum = 0;
    int iou_n = 0;
    for (size_t i = 0; i < results.size(); ++i)
    {
        st.found += results[i].found;
        st.n_orb += results[i].mode == TrackMode::Orb;
        st.n_flow += results[i].mode == TrackMode::Flow;
        st.n_held += results[i].mode == TrackMode::Held;
        st.mean_ms += results[i].ms_total;
        if (st.has_gt)
        {
            if (g_visible[i]) { ++st.visible; st.found_visible += results[i].found; }
            else if (results[i].found) ++st.false_positive;
            if (iou[i] >= 0) { iou_sum += iou[i]; ++iou_n; }
        }
    }
    st.mean_ms /= std::max(1, st.frames);
    st.mean_iou = iou_n ? iou_sum / iou_n : 0.0;

    {   // CSV по кадрам: для таблиц и графиков отчёта
        const std::string file = "tracking_frames.csv";
        std::ofstream out(file);
        if (out)
        {
            std::cerr << "track: writing " << file << "\n";
            out << "frame,found,mode,keypoints,matches,inliers,tracked,ms_detect,ms_match,ms_ransac,ms_flow,ms_total,iou\n";
            static const char* kMode[] = {"lost", "orb", "flow", "held"};
            for (size_t i = 0; i < results.size(); ++i)
                out << i << "," << results[i].found << "," << kMode[static_cast<int>(results[i].mode)] << ","
                    << results[i].n_kp << "," << results[i].n_matches << "," << results[i].n_inliers << ","
                    << results[i].n_tracked << "," << results[i].ms_detect << "," << results[i].ms_match << ","
                    << results[i].ms_ransac << "," << results[i].ms_flow << "," << results[i].ms_total << "," << iou[i] << "\n";
        }
    }
    {
        std::lock_guard<std::mutex> lock(s.mtx);
        s.results = std::move(results);
        s.iou = std::move(iou);
        s.ref_keypoints = tracker.ref_keypoints();
        s.stats = st;
    }
    s.results_version.fetch_add(1);
    s.tracking_done.store(true);
    std::cerr << "track: done, found " << st.found << "/" << st.frames << ", mean " << st.mean_ms << " ms/frame\n";
}

// ---------------------------------------------------------------------------------------------
// Бенчмарк. Те же приёмы, что в ЛР1 (bench::time_ms: прогрев + медиана, bench::push: точки по одной).
// Серии: поиск по эталону (ORB) по стадиям + один шаг оптического потока. Стадии берём из TrackResult,
// итог поиска — через time_ms (прогрев + медиана).
//   «Поиск»: перед каждым вызовом сбрасываем историю, поэтому всегда выполняется полный ORB-поиск.
//   «Поток»: сеем трекер на кадре frame, затем замеряем обработку следующего кадра (ветка потока).
// ---------------------------------------------------------------------------------------------
void measure_point(TrackingLabState& s, std::vector<BenchSeries>& v, double x,
                   const cv::Mat& ref, const cv::Mat& frame, const cv::Mat& frame_next, const TrackerParams& p)
{
    ObjectTracker tr;
    if (!tr.init(ref, p)) return;

    std::vector<double> d, m, r;
    const double total = bench::time_ms([&] {
        tr.reset_tracking();
        const TrackResult res = tr.process(frame);
        d.push_back(res.ms_detect);
        m.push_back(res.ms_match);
        r.push_back(res.ms_ransac);
        bench::sink = static_cast<uint8_t>(res.n_inliers);
    }, 9);
    // первый элемент каждого вектора — прогревочный вызов, выбрасываем
    auto med = [](std::vector<double>& a) { a.erase(a.begin()); return bench::median(a); };
    bench::push(s, v, 0, x, med(d));
    bench::push(s, v, 1, x, med(m));
    bench::push(s, v, 2, x, med(r));
    bench::push(s, v, 3, x, total);

    std::vector<double> fl;
    for (int i = 0; i < 9; ++i)
    {
        tr.reset_tracking();
        tr.process(frame);                              // посев: поиск по эталону
        const TrackResult res = tr.process(frame_next); // следующий кадр: оптический поток
        if (res.mode == TrackMode::Flow) fl.push_back(res.ms_flow);
        bench::sink = static_cast<uint8_t>(res.n_inliers);
    }
    if (!fl.empty()) bench::push(s, v, 4, x, bench::median(fl));
}

cv::Mat resize_to_width(const cv::Mat& img, int width)
{
    cv::Mat out;
    cv::resize(img, out, cv::Size(width, width * img.rows / img.cols), 0, 0, cv::INTER_AREA);
    return out;
}

void benchmark(TrackingLabState& s)
{
    s.bench_done.store(false);
    cv::Mat ref, frame, frame_next;
    TrackerParams p;
    {
        std::lock_guard<std::mutex> lock(s.mtx);
        if (s.frames.size() < 2) return;
        ref = s.frames[0];
        const size_t k = s.frames.size() / 4;  // кадр, где объект, как правило, уже виден
        frame = s.frames[k];
        frame_next = s.frames[k + 1];
        p = s.params;
        const auto names = {"поиск: detect+describe", "поиск: match (BF+ratio)", "поиск: RANSAC", "поиск: итого", "оптический поток (1 кадр)"};
        s.stage_vs_features = bench::make_series(names);
        s.stage_vs_width    = bench::make_series(names);
    }

    for (int nf : {200, 500, 1000, 1500, 2000, 3000, 5000})
    {
        if (s.stop) return;
        TrackerParams q = p;
        q.nfeatures = nf;
        measure_point(s, s.stage_vs_features, nf, ref, frame, frame_next, q);
    }
    for (int w : {320, 480, 640, 800, 960})
    {
        if (s.stop) return;
        if (w > ref.cols) continue;
        TrackerParams q = p;
        if (q.has_roi) // область объекта задана в пикселях первого кадра — масштабируем вместе с кадром
            for (auto& c : q.roi) c *= static_cast<float>(w) / ref.cols;
        measure_point(s, s.stage_vs_width, w, resize_to_width(ref, w), resize_to_width(frame, w),
                      resize_to_width(frame_next, w), q);
    }
    {
        std::lock_guard<std::mutex> lock(s.mtx);
        bench::write_csv("bench_tracking_vs_features.csv", s.stage_vs_features);
        bench::write_csv("bench_tracking_vs_width.csv", s.stage_vs_width);
    }
    std::cerr << "track: benchmark done\n";
    s.bench_done.store(true);
}

} // namespace

void RunTrackingLab(TrackingLabState& s)
{
    load_source(s);
    track_all(s);
    benchmark(s);

    // Дальше ждём запросов от UI: другое видео / новые параметры / повтор бенчмарка.
    while (!s.stop)
    {
        if (s.reload.exchange(false))       { load_source(s); track_all(s); }
        else if (s.rerun.exchange(false))   track_all(s);
        else if (s.rebench.exchange(false)) benchmark(s);
        else std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
}
