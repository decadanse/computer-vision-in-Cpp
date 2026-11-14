#import "TrackingLabView.h"

#include <algorithm>
#include <cstring>
#include <string>

#include "imgui.h"
#include "implot.h"

#include "GpuUtils.h"

namespace {

// Вертикальная отметка текущего кадра на графике (две точки — безопасно для любой версии ImPlot).
void mark_current(int cur, double ymax)
{
    const double mx[2] = {(double)cur, (double)cur}, my[2] = {0.0, ymax};
    ImPlot::PlotLine("кадр", mx, my, 2);
}

void plot_per_frame(const char* title, const char* ylabel, const std::vector<double>& xs,
                    const std::vector<double>& ys, int cur, float width)
{
    if (ImPlot::BeginPlot(title, ImVec2(width, 200)))
    {
        ImPlot::SetupAxes("кадр", ylabel, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        if (!ys.empty())
        {
            ImPlot::PlotLine(ylabel, xs.data(), ys.data(), (int)ys.size());
            mark_current(cur, *std::max_element(ys.begin(), ys.end()));
        }
        ImPlot::EndPlot();
    }
}

} // namespace

void DrawTrackingLab(TrackingLabState& s, id<MTLDevice> device)
{
    // ---- локальный кэш UI. Копии обновляются только при смене версии результатов, чтобы не копировать
    //      данные с inlier-ами 60 раз в секунду под мьютексом (тот же принцип, что в ЛР1) ----
    static std::vector<cv::Mat> frames;
    static std::vector<TrackResult> results;
    static std::vector<double> iou;
    static TrackStats stats;
    static std::string source_note;
    static int ref_kp = 0;
    static int cached_version = -1;

    static gpu::Tex tex;
    static int cur = 0, tex_frame = -1;
    static bool playing = false, show_points = true, first_call = true;
    static float fps = 30.f, accum = 0.f, ransac_f = 3.f;
    static char label[128] = "Object";
    static char path[512] = "";
    static TrackerParams ui_params;
    static bool picking = false;                  // режим выбора области объекта на кадре 0
    static std::vector<cv::Point2f> pick_pts;     // уже выбранные углы (в пикселях кадра)
    static std::vector<double> xs, y_inl, y_ms, y_iou;

    if (first_call)
    {
        std::lock_guard<std::mutex> lock(s.mtx);
        std::strncpy(path, s.video_path.c_str(), sizeof(path) - 1);
        ui_params = s.params;
        ransac_f = (float)ui_params.ransac_thr;
        first_call = false;
    }

    if (s.tracking_done.load() && cached_version != s.results_version.load())
    {
        std::lock_guard<std::mutex> lock(s.mtx);
        frames = s.frames;
        results = s.results;
        iou = s.iou;
        stats = s.stats;
        ref_kp = s.ref_keypoints;
        source_note = s.source_note;
        cached_version = s.results_version.load();
        tex_frame = -1;
        cur = std::min(cur, std::max(0, (int)frames.size() - 1));

        const size_t n = results.size();
        xs.resize(n); y_inl.resize(n); y_ms.resize(n); y_iou.resize(n);
        for (size_t i = 0; i < n; ++i)
        {
            xs[i] = (double)i;
            y_inl[i] = results[i].n_inliers;
            y_ms[i] = results[i].ms_total;
            y_iou[i] = iou[i] < 0 ? 0.0 : iou[i];
        }
    }

    ImGui::Begin("Object tracking");

    // ---- источник и загрузка другого видео ----
    ImGui::TextWrapped("Источник: %s", source_note.empty() ? "загрузка..." : source_note.c_str());
    ImGui::InputText("Путь к видео", path, sizeof(path));
    ImGui::SameLine();
    if (ImGui::Button("Загрузить"))
    {
        {
            std::lock_guard<std::mutex> lock(s.mtx);
            s.video_path = path;
            ui_params.ransac_thr = ransac_f;
            s.params = ui_params;
        }
        frames.clear(); results.clear(); iou.clear();
        cached_version = -1;
        s.reload.store(true);
    }

    // ---- параметры трекера ----
    ImGui::InputText("Подпись рамки (название предмета)", label, sizeof(label));
    ImGui::SliderInt("nfeatures (ORB)", &ui_params.nfeatures, 200, 8000);
    ImGui::SliderFloat("ratio (тест Лоу)", &ui_params.ratio, 0.5f, 0.95f);
    ImGui::SliderFloat("порог RANSAC, px", &ransac_f, 1.f, 10.f);
    ImGui::SliderInt("min inliers", &ui_params.min_inliers, 4, 60);
    // Область объекта: если на первом кадре объект не занимает весь кадр — отметьте 4 угла кликами
    // на кадре 0 (порядок: левый верхний, правый верхний, правый нижний, левый нижний).
    if (ImGui::Checkbox("Выбрать область объекта на кадре 0 (4 клика)", &picking))
    {
        pick_pts.clear();
        if (picking) { cur = 0; playing = false; }
    }
    ImGui::SameLine();
    if (ImGui::Button("Объект = весь кадр")) { ui_params.has_roi = false; pick_pts.clear(); picking = false; }
    ImGui::SameLine();
    ImGui::Text(ui_params.has_roi ? "область задана" : "область: весь кадр");
    if (ImGui::Button("Пересчитать трекинг"))
    {
        {
            std::lock_guard<std::mutex> lock(s.mtx);
            ui_params.ransac_thr = ransac_f;
            s.params = ui_params;
        }
        s.rerun.store(true);
    }
    ImGui::SameLine();
    if (ImGui::Button("Повторить бенчмарк"))
    {
        {
            std::lock_guard<std::mutex> lock(s.mtx);
            ui_params.ransac_thr = ransac_f;
            s.params = ui_params;
        }
        s.rebench.store(true);
    }

    if (!s.tracking_done.load())
        ImGui::Text("Трекинг... (обработано кадров: %d)", s.progress.load());

    // ---- видео с рамкой ----
    if (!frames.empty() && !results.empty())
    {
        const int n = (int)std::min(frames.size(), results.size());

        ImGui::Checkbox("Воспроизведение", &playing);
        ImGui::SameLine();
        ImGui::Checkbox("Показать inlier-ы", &show_points);
        ImGui::SliderFloat("кадров/с", &fps, 1.f, 60.f);
        ImGui::SliderInt("Кадр", &cur, 0, n - 1);

        if (playing)
        {
            accum += ImGui::GetIO().DeltaTime * fps;
            while (accum >= 1.f) { cur = (cur + 1) % n; accum -= 1.f; }
        }
        cur = std::clamp(cur, 0, n - 1);
        if (tex_frame != cur) { gpu::upload(device, tex, frames[cur]); tex_frame = cur; }

        const float draw_w = std::min(ImGui::GetContentRegionAvail().x, 960.f);
        const float scale = draw_w / (float)frames[cur].cols;
        const ImVec2 sz(draw_w, frames[cur].rows * scale);
        ImGui::Image(gpu::tex_id(tex), sz);
        const ImVec2 p0 = ImGui::GetItemRectMin();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const TrackResult& r = results[cur];

        // ---- выбор области: клики по кадру 0 переводятся в пиксели кадра ----
        if (picking && cur == 0 && ImGui::IsItemHovered() && ImGui::IsMouseClicked(0))
        {
            const ImVec2 m = ImGui::GetMousePos();
            pick_pts.emplace_back((m.x - p0.x) / scale, (m.y - p0.y) / scale);
            if (pick_pts.size() == 4)
            {
                for (int i = 0; i < 4; ++i) ui_params.roi[i] = pick_pts[i];
                ui_params.has_roi = true;
                picking = false;
                pick_pts.clear();
            }
        }
        if (cur == 0)
        {
            // показываем уже выбранные точки или заданную область (синим)
            std::vector<ImVec2> v;
            if (!pick_pts.empty()) for (auto& p : pick_pts) v.emplace_back(p0.x + p.x * scale, p0.y + p.y * scale);
            else if (ui_params.has_roi) for (auto& p : ui_params.roi) v.emplace_back(p0.x + p.x * scale, p0.y + p.y * scale);
            for (auto& p : v) dl->AddCircleFilled(p, 4.f, IM_COL32(60, 140, 255, 255));
            if (v.size() >= 2) dl->AddPolyline(v.data(), (int)v.size(), IM_COL32(60, 140, 255, 255),
                                               v.size() == 4 ? ImDrawFlags_Closed : ImDrawFlags_None, 2.0f);
        }

        // Цвет рамки показывает, КАК найден объект: зелёный — поиск ORB, голубой — оптический поток,
        // жёлтый — потерян (размытие / объект вне кадра), показана последняя известная рамка.
        const bool held = r.mode == TrackMode::Held;
        if (r.found || held)
        {
            // Рамка и подпись рисуются средствами ImGui (шрифт с кириллицей), а не cv::putText (без кириллицы).
            const ImU32 col = held ? IM_COL32(255, 200, 0, 255)
                            : r.mode == TrackMode::Flow ? IM_COL32(0, 220, 255, 255) : IM_COL32(0, 255, 0, 255);
            ImVec2 q[4];
            for (int i = 0; i < 4; ++i) q[i] = ImVec2(p0.x + r.quad[i].x * scale, p0.y + r.quad[i].y * scale);
            dl->AddPolyline(q, 4, col, ImDrawFlags_Closed, 3.0f);

            if (show_points && !held)
                for (const auto& pt : r.inliers)
                    dl->AddCircleFilled(ImVec2(p0.x + pt.x * scale, p0.y + pt.y * scale), 2.5f, IM_COL32(255, 60, 60, 255));

            const ImVec2 ts = ImGui::CalcTextSize(label);
            ImVec2 pos(q[0].x, q[0].y - ts.y - 6.f);                        // над левым верхним углом рамки
            if (pos.y < p0.y) pos.y = q[0].y + 4.f;                        // не помещается — внутрь рамки
            pos.x = std::clamp(pos.x, p0.x + 3.f, std::max(p0.x + 3.f, p0.x + sz.x - ts.x - 3.f));
            dl->AddRectFilled(ImVec2(pos.x - 3, pos.y - 2), ImVec2(pos.x + ts.x + 3, pos.y + ts.y + 2), (held ? IM_COL32(150, 110, 0, 255) : IM_COL32(0, 110, 130, 255)));
            dl->AddText(pos, IM_COL32(255, 255, 255, 255), label);
        }
        else
        {
            dl->AddText(ImVec2(p0.x + 8, p0.y + 8), IM_COL32(255, 80, 80, 255), "объект не найден");
        }

        static const char* kModeName[] = {"не найден", "ORB (поиск)", "поток", "удержание"};
        ImGui::Text("Кадр %d: %s | ключевых точек %d, совпадений %d, inlier-ов %d | detect %.2f мс, match %.2f мс, RANSAC %.2f мс, всего %.2f мс",
                    cur, kModeName[(int)r.mode], r.n_kp, r.n_matches, r.n_inliers,
                    r.ms_detect, r.ms_match, r.ms_ransac, r.ms_total);
        if (cur < (int)iou.size() && iou[cur] >= 0) ImGui::Text("IoU с истинной рамкой: %.3f", iou[cur]);

        // ---- сводка ----
        ImGui::Separator();
        ImGui::Text("Эталон: %d ключевых точек | найден на %d из %d кадров | среднее время %.2f мс/кадр",
                    ref_kp, stats.found, stats.frames, stats.mean_ms);
        ImGui::Text("Режимы: ORB %d | поток %d | удержание %d кадров", stats.n_orb, stats.n_flow, stats.n_held);
        if (stats.has_gt)
            ImGui::Text("Синтетика: найден на %d из %d видимых кадров, ложных срабатываний %d, средний IoU %.3f",
                        stats.found_visible, stats.visible, stats.false_positive, stats.mean_iou);

        // ---- графики по кадрам ----
        const float half = (ImGui::GetContentRegionAvail().x - 8.f) * 0.5f;
        plot_per_frame("Inlier-ы по кадрам", "inliers", xs, y_inl, cur, half);
        ImGui::SameLine();
        plot_per_frame("Время обработки кадра, мс", "ms", xs, y_ms, cur, half);
        if (stats.has_gt) plot_per_frame("IoU (0 = нет объекта / не найден)", "IoU", xs, y_iou, cur, half);
    }

    // ---- бенчмарк: копия серий под мьютексом ----
    std::vector<BenchSeries> bf, bw;
    {
        std::lock_guard<std::mutex> lock(s.mtx);
        bf = s.stage_vs_features;
        bw = s.stage_vs_width;
    }
    ImGui::Separator();
    ImGui::Text(s.bench_done.load() ? "Бенчмарк завершён (CSV сохранены)" : "Бенчмарк идёт...");
    const float pw = (ImGui::GetContentRegionAvail().x - 8.f) * 0.5f;
    gpu::plot_series("Время по стадиям vs nfeatures", "nfeatures", bf, pw);
    ImGui::SameLine();
    gpu::plot_series("Время по стадиям vs ширина кадра", "width, px", bw, pw);

    ImGui::End();
}
