#include "Tracker.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace {

using clk = std::chrono::steady_clock;

double ms_since(clk::time_point a, clk::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

cv::Mat to_gray(const cv::Mat& m)
{
    if (m.channels() == 1) return m;
    cv::Mat g;
    cv::cvtColor(m, g, m.channels() == 4 ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY);
    return g;
}

// Масштабы эталона. ORB сам строит пирамиду, но у него конечный диапазон; когда объект в кадре в 3-4 раза
// мельче, чем в эталоне, нужны копии эталона, уменьшенные заранее (чтобы мелкие детали были «на нулевом уровне»).
const double kScales[] = {1.0, 0.7, 0.5, 0.35};

std::array<cv::Point2f, 4> project(const cv::Mat& H, const std::array<cv::Point2f, 4>& q)
{
    std::vector<cv::Point2f> in(q.begin(), q.end()), out;
    cv::perspectiveTransform(in, out, H);
    std::array<cv::Point2f, 4> r;
    for (int i = 0; i < 4; ++i) r[i] = out[i];
    return r;
}

// Растягивает многоугольник относительно центра в k раз (область поиска вокруг прошлого положения).
std::vector<cv::Point2f> expand(const std::array<cv::Point2f, 4>& q, float k)
{
    cv::Point2f c(0, 0);
    for (const auto& p : q) c += p * 0.25f;
    std::vector<cv::Point2f> r;
    for (const auto& p : q) r.push_back(c + (p - c) * k);
    return r;
}

std::vector<cv::Point> to_int(const std::vector<cv::Point2f>& v)
{
    std::vector<cv::Point> r;
    for (const auto& p : v) r.emplace_back(cvRound(p.x), cvRound(p.y));
    return r;
}

// «Выпрямленный» участок: warp изображения матрицей M в 128x128, затем усреднение до 32x32 (INTER_AREA).
// Усреднение — это ещё и размытие, поэтому сравнение устойчиво к расфокусу и мелкому шуму.
cv::Mat rect_patch(const cv::Mat& gray, const cv::Mat& M)
{
    cv::Mat big, small;
    cv::warpPerspective(gray, big, M, cv::Size(128, 128), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    cv::resize(big, small, cv::Size(32, 32), 0, 0, cv::INTER_AREA);
    small.convertTo(small, CV_32F);
    return small;
}

double quad_area(const std::array<cv::Point2f, 4>& q)
{
    return std::fabs(cv::contourArea(std::vector<cv::Point2f>(q.begin(), q.end())));
}

} // namespace

bool plausible_quad(const std::array<cv::Point2f, 4>& q, cv::Size frame_size)
{
    const std::vector<cv::Point2f> v(q.begin(), q.end());
    if (!cv::isContourConvex(v)) return false;
    const double area = std::fabs(cv::contourArea(v));
    const double frame_area = static_cast<double>(frame_size.width) * frame_size.height;
    return area > 0.0005 * frame_area && area < 4.0 * frame_area;
}

double quad_iou(const std::array<cv::Point2f, 4>& a, const std::array<cv::Point2f, 4>& b)
{
    // Растровый IoU: закрашиваем оба четырёхугольника и считаем пересечение/объединение пикселей.
    // (intersectConvexConvex даёт неверный результат, когда стороны совпадают — например, идентичные рамки.)
    std::vector<cv::Point2f> all(a.begin(), a.end());
    all.insert(all.end(), b.begin(), b.end());
    const cv::Rect box = cv::boundingRect(all);
    if (box.width <= 0 || box.height <= 0) return 0.0;
    const double k = std::min(1.0, std::sqrt(1.0e6 / (static_cast<double>(box.width) * box.height))); // не более ~1 Мпикс
    const cv::Size sz(static_cast<int>(box.width * k) + 2, static_cast<int>(box.height * k) + 2);
    auto fill = [&](const std::array<cv::Point2f, 4>& q) {
        std::vector<cv::Point> p;
        for (const auto& pt : q)
            p.emplace_back(cvRound((pt.x - box.x) * k), cvRound((pt.y - box.y) * k));
        cv::Mat m = cv::Mat::zeros(sz, CV_8U);
        cv::fillConvexPoly(m, p, cv::Scalar(255));
        return m;
    };
    const cv::Mat ma = fill(a), mb = fill(b);
    cv::Mat inter, uni;
    cv::bitwise_and(ma, mb, inter);
    cv::bitwise_or(ma, mb, uni);
    const int u = cv::countNonZero(uni);
    return u > 0 ? static_cast<double>(cv::countNonZero(inter)) / u : 0.0;
}

// ---------------------------------------------------------------------------------------------
// Инициализация: ORB-точки эталона в нескольких масштабах, только внутри области объекта.
// ---------------------------------------------------------------------------------------------
bool ObjectTracker::init(const cv::Mat& reference, const TrackerParams& params)
{
    p_ = params;
    orb_ = cv::ORB::create(p_.nfeatures, p_.scale_factor, p_.nlevels);
    templates_.clear();
    reset_tracking();

    const cv::Mat gray = to_gray(reference);
    const float w = static_cast<float>(gray.cols), h = static_cast<float>(gray.rows);
    roi_ = p_.has_roi ? p_.roi
                      : std::array<cv::Point2f, 4>{cv::Point2f(0, 0), cv::Point2f(w, 0), cv::Point2f(w, h), cv::Point2f(0, h)};

    {   // эталонный участок объекта для проверки по внешнему виду
        const std::vector<cv::Point2f> src(roi_.begin(), roi_.end());
        const std::vector<cv::Point2f> dst = {{0, 0}, {127, 0}, {127, 127}, {0, 127}};
        P_ = cv::getPerspectiveTransform(src, dst);
        ref_patch_ = rect_patch(gray, P_);
    }

    for (double s : kScales)
    {
        cv::Mat img;
        if (s == 1.0) img = gray;
        else cv::resize(gray, img, cv::Size(), s, s, cv::INTER_AREA);
        if (img.cols < 64 || img.rows < 64) continue;

        // маска: ключевые точки берём только внутри области объекта (стол и фон вокруг в эталон не попадают)
        std::vector<cv::Point> poly;
        for (const auto& c : roi_) poly.emplace_back(cvRound(c.x * s), cvRound(c.y * s));
        cv::Mat mask = cv::Mat::zeros(img.size(), CV_8U);
        cv::fillConvexPoly(mask, poly, cv::Scalar(255));

        Template t;
        orb_->detectAndCompute(img, mask, t.kp, t.desc);
        if (t.kp.size() < 8) continue;
        for (const auto& k : t.kp) t.ref_pt.push_back(k.pt * static_cast<float>(1.0 / s)); // в координаты 1-го кадра
        templates_.push_back(std::move(t));
    }
    return !templates_.empty();
}

int ObjectTracker::ref_keypoints() const
{
    return templates_.empty() ? 0 : static_cast<int>(templates_.front().kp.size());
}

void ObjectTracker::reset_tracking()
{
    have_pose_ = false;
    prev_gray_.release();
    prev_pts_.clear();
    ref_pts_.clear();
    lost_run_ = 0;
    since_anchor_ = 0;
    sharp_ema_ = 0.0;
}

// ---------------------------------------------------------------------------------------------
// Сопоставление кадра с каждым масштабом эталона; выбираем масштаб с наибольшим числом inlier-ов.
//   region != nullptr: берём только совпадения, чья точка в кадре лежит внутри области (подсказка).
//   Почему не один общий набор дескрипторов всех масштабов: одна и та же физическая точка в соседних
//   масштабах имеет почти одинаковый дескриптор, и «второй сосед» становится почти таким же хорошим, как
//   первый — тест Лоу отбросил бы лучшие совпадения. Поэтому масштабы сопоставляем по отдельности.
// ---------------------------------------------------------------------------------------------
ObjectTracker::Cand ObjectTracker::match_templates(const std::vector<cv::KeyPoint>& kp, const cv::Mat& desc, float ratio,
                                                   const std::vector<cv::Point2f>* region,
                                                   double& ms_match, double& ms_ransac) const
{
    Cand best;
    if (desc.empty() || desc.rows < 2) return best;

    for (const auto& t : templates_)
    {
        const auto a = clk::now();
        std::vector<std::vector<cv::DMatch>> knn;
        matcher_.knnMatch(t.desc, desc, knn, 2);

        std::vector<cv::Point2f> src, dst;
        for (const auto& m : knn)
        {
            // тест Лоу: лучший сосед должен быть заметно лучше второго
            if (m.size() < 2 || m[0].distance >= ratio * m[1].distance) continue;
            const cv::Point2f pt = kp[m[0].trainIdx].pt;
            if (region && cv::pointPolygonTest(*region, pt, false) < 0) continue;
            src.push_back(t.ref_pt[m[0].queryIdx]);
            dst.push_back(pt);
        }
        const auto b = clk::now();
        ms_match += ms_since(a, b);
        if (src.size() < 4) continue;

        cv::Mat mask;
        const cv::Mat H = cv::findHomography(src, dst, cv::RANSAC, p_.ransac_thr, mask); // Начальная оценка H строится через DLT, то есть SVD матрицы 8×9 из пар точек.
        ms_ransac += ms_since(b, clk::now());
        if (H.empty() || mask.rows != static_cast<int>(src.size())) continue;

        int inl = 0;
        for (int i = 0; i < mask.rows; ++i) inl += mask.at<uchar>(i) ? 1 : 0;
        if (inl > best.inl)
        {
            best.H = H;
            best.inl = inl;
            best.n_matches = static_cast<int>(src.size());
            best.cur.clear();
            best.ref.clear();
            for (int i = 0; i < mask.rows; ++i)
                if (mask.at<uchar>(i)) { best.cur.push_back(dst[i]); best.ref.push_back(src[i]); }
        }
    }
    return best;
}

// Резкость = дисперсия лапласиана: Var[ d2I/dx2 + d2I/dy2 ]. Размытие подавляет высокие частоты, и дисперсия
// падает в десятки раз. Считаем в области вокруг прошлого положения объекта (если оно известно).
double ObjectTracker::sharpness(const cv::Mat& gray) const
{
    cv::Rect r(0, 0, gray.cols, gray.rows);
    if (have_pose_)
    {
        const cv::Rect b = cv::boundingRect(std::vector<cv::Point2f>(last_quad_.begin(), last_quad_.end())) & r;
        if (b.width >= 32 && b.height >= 32) r = b;
    }
    // Нормировка: область приводим к ширине 128 px (иначе резкость зависит от масштаба объекта) и слегка
    // сглаживаем (σ=1.0), чтобы шум сенсора не доминировал в лапласиане.
    cv::Mat norm, lap;
    const double s = 128.0 / r.width;
    cv::resize(gray(r), norm, cv::Size(), s, s, cv::INTER_AREA);
    cv::GaussianBlur(norm, norm, cv::Size(0, 0), 1.0);
    cv::Laplacian(norm, lap, CV_32F);
    cv::Scalar mu, sd;
    cv::meanStdDev(lap, mu, sd);
    return sd[0] * sd[0];
}

// Проверка по внешнему виду: вырезаем из кадра участок под предполагаемым положением объекта
// (выпрямляем гомографией) и считаем нормированную корреляцию с эталонным участком:
//   NCC = sum((a - mean_a)(b - mean_b)) / sqrt(sum((a - mean_a)^2) * sum((b - mean_b)^2)).
// Если рамка «уехала» на фон, NCC около 0; если на объекте — близка к 1 (при размытии падает, но не до нуля).
double ObjectTracker::appearance_ncc(const cv::Mat& gray, const cv::Mat& H) const
{
    const cv::Mat patch = rect_patch(gray, P_ * H.inv());
    cv::Mat r;
    cv::matchTemplate(patch, ref_patch_, r, cv::TM_CCOEFF_NORMED);
    return r.at<float>(0, 0);
}

// ---------------------------------------------------------------------------------------------
// Оптический поток Лукаса-Канаде от последнего удачного кадра:
//   1) переносим точки вперёд (prev -> текущий), затем обратно (текущий -> prev);
//   2) оставляем точки, вернувшиеся к исходному положению (прямо-обратная проверка);
//   3) гомография ЭТАЛОН -> ТЕКУЩИЙ КАДР по оставшимся точкам с RANSAC.
// Каждая точка хранит свои координаты в эталоне, поэтому H всегда считается «от эталона»: ошибки
// не накапливаются по цепочке кадров (нет перемножения гомографий кадр-к-кадру).
// ---------------------------------------------------------------------------------------------
bool ObjectTracker::track_flow(const cv::Mat& gray, Cand& out) const
{
    std::vector<cv::Point2f> nxt, back;
    std::vector<uchar> s1, s2;
    std::vector<float> e1, e2;
    const cv::Size win(21, 21);
    const cv::TermCriteria tc(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);

    cv::calcOpticalFlowPyrLK(prev_gray_, gray, prev_pts_, nxt, s1, e1, win, 3, tc);
    cv::calcOpticalFlowPyrLK(gray, prev_gray_, nxt, back, s2, e2, win, 3, tc);

    std::vector<cv::Point2f> cu, rf;
    for (size_t i = 0; i < prev_pts_.size(); ++i)
    {
        if (!s1[i] || !s2[i]) continue;
        const cv::Point2f d = back[i] - prev_pts_[i];
        if (std::hypot(d.x, d.y) >= p_.fb_thr) continue; // прямо-обратная проверка
        if (nxt[i].x < 0 || nxt[i].y < 0 || nxt[i].x >= gray.cols || nxt[i].y >= gray.rows) continue;
        cu.push_back(nxt[i]);
        rf.push_back(ref_pts_[i]);
    }
    if (static_cast<int>(cu.size()) < std::max(4, p_.min_flow_inliers)) return false;
    // Если из набора выжило слишком мало точек, поток, скорее всего, сорвался (резкое движение, размытие).
    if (cu.size() < prev_pts_.size() * 0.15) return false;

    cv::Mat mask;
    const cv::Mat H = cv::findHomography(rf, cu, cv::RANSAC, p_.ransac_thr, mask);
    if (H.empty() || mask.rows != static_cast<int>(cu.size())) return false;

    Cand c;
    for (int i = 0; i < mask.rows; ++i)
        if (mask.at<uchar>(i)) { c.cur.push_back(cu[i]); c.ref.push_back(rf[i]); }
    c.inl = static_cast<int>(c.cur.size());
    if (c.inl < p_.min_flow_inliers) return false;
    if (c.inl < 0.5 * cu.size()) return false; // точки не согласуются между собой — гомография случайна

    const auto quad = project(H, roi_);
    if (!plausible_quad(quad, gray.size())) return false;
    // «Здравый смысл»: за короткое время площадь рамки не может измениться более чем в 3 раза.
    if (have_pose_)
    {
        const double ratio = quad_area(quad) / std::max(1.0, quad_area(last_quad_));
        if (ratio < 1.0 / 3.0 || ratio > 3.0) return false;
    }
    // Проверка по внешнему виду: не «уехала» ли рамка на фон
    if (appearance_ncc(gray, H) < p_.min_ncc) return false;
    c.H = H;
    c.n_matches = c.inl;
    out = std::move(c);
    return true;
}

// Пополнение набора отслеживаемых точек: когда точек мало (объект отдалился/часть точек потеряна),
// добавляем новые углы Ши-Томаси внутри текущей рамки. Их координаты в эталоне: ref = H^-1 * point.
void ObjectTracker::refill(const cv::Mat& gray, const cv::Mat& H)
{
    const int need = p_.flow_points - static_cast<int>(prev_pts_.size());
    if (need <= 0) return;

    const auto quad = project(H, roi_);
    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8U);
    cv::fillConvexPoly(mask, to_int(std::vector<cv::Point2f>(quad.begin(), quad.end())), cv::Scalar(255));
    for (const auto& p : prev_pts_) cv::circle(mask, p, 6, cv::Scalar(0), -1); // не рядом с уже имеющимися

    std::vector<cv::Point2f> add, add_ref;
    cv::goodFeaturesToTrack(gray, add, need, 0.01, 7, mask, 5); // Строится матрица градиентов M = [ΣIx² ΣIxIy; ΣIxIy ΣIy²] в окне вокруг пикселя. Точка берётся, если меньшее собственное значение λmin велико.
    if (add.empty()) return;
    cv::perspectiveTransform(add, add_ref, H.inv());
    prev_pts_.insert(prev_pts_.end(), add.begin(), add.end());
    ref_pts_.insert(ref_pts_.end(), add_ref.begin(), add_ref.end());
}

// ---------------------------------------------------------------------------------------------
// Обработка кадра.
//   1. Если есть история — пробуем оптический поток от последнего удачного кадра (это работает и через
//      разрыв в несколько кадров: размытые кадры пропускаются, поток «перепрыгивает» их).
//   2. Если потока нет/мало, либо пора перепривязаться — поиск по эталону (ORB). Если объект недавно
//      потерян, сначала ищем в области вокруг прошлого положения с мягким тестом Лоу.
//   3. Если не нашли — удерживаем последнюю рамку до max_hold кадров (режим Held), затем Lost.
// ---------------------------------------------------------------------------------------------
TrackResult ObjectTracker::process(const cv::Mat& frame)
{
    TrackResult r;
    const auto t0 = clk::now();
    const cv::Mat gray = to_gray(frame);

    // ---- 0. кадр размыт? Тогда ничему на нём не верим: удерживаем прошлую рамку и ничего не обновляем.
    //         Состояние (кадр и точки для потока) остаётся от последнего ЧЁТКОГО кадра, и на следующем
    //         чётком кадре поток «перепрыгнет» через размытый промежуток.
    const double sharp = sharpness(gray);
    if (have_pose_ && sharp_ema_ > 0.0 && sharp < p_.blur_ratio * sharp_ema_ && lost_run_ < p_.max_hold)
    {
        ++lost_run_;
        ++since_anchor_;
        r.mode = TrackMode::Held;
        r.quad = last_quad_;
        r.ms_total = ms_since(t0, clk::now());
        return r;
    }

    Cand cur;
    TrackMode mode = TrackMode::Lost;

    // ---- 1. оптический поток ----
    if (have_pose_ && !prev_pts_.empty() && prev_gray_.size() == gray.size())
    {
        const auto a = clk::now();
        if (track_flow(gray, cur)) mode = TrackMode::Flow;
        r.ms_flow = ms_since(a, clk::now());
    }

    // ---- 2. поиск по эталону ----
    const bool need_orb = mode != TrackMode::Flow || lost_run_ > 0 || since_anchor_ >= p_.reanchor_every || cur.inl < p_.reanchor_below;
    if (need_orb && !templates_.empty())
    {
        const auto a = clk::now();
        std::vector<cv::KeyPoint> kp;
        cv::Mat desc;
        orb_->detectAndCompute(gray, cv::noArray(), kp, desc);
        r.ms_detect = ms_since(a, clk::now());
        r.n_kp = static_cast<int>(kp.size());

        Cand c;
        if (have_pose_) // подсказка: ищем вокруг последнего положения, порог мягче
        {
            const auto region = expand(last_quad_, 1.8f);
            c = match_templates(kp, desc, p_.relaxed_ratio, &region, r.ms_match, r.ms_ransac);
            if (c.inl < std::max(6, p_.min_inliers * 2 / 3)) c = Cand();
        }
        if (c.inl == 0) // глобальный поиск, строгий тест
        {
            c = match_templates(kp, desc, p_.ratio, nullptr, r.ms_match, r.ms_ransac);
            if (c.inl < p_.min_inliers) c = Cand();
        }
        r.n_matches = c.n_matches;

        // Проверка по внешнему виду для ORB — мягче, чем для потока: у ORB есть собственное свидетельство (дескрипторы)
        if (c.inl > 0 && plausible_quad(project(c.H, roi_), gray.size()) && appearance_ncc(gray, c.H) >= 0.5 * p_.min_ncc)
        {
            if (mode == TrackMode::Flow)
            {
                // Перепривязка принимается, если согласуется с потоком ИЛИ опирается на больше точек, чем поток
                // (иначе поток мог зацепиться за фон и «застрять», а слабый ложный ORB его не переубедит).
                if (quad_iou(project(cur.H, roi_), project(c.H, roi_)) > 0.5 || c.inl > cur.inl) { cur = std::move(c); mode = TrackMode::Orb; }
            }
            else
            {
                cur = std::move(c);
                mode = TrackMode::Orb;
            }
        }
    }

    // ---- 3. итог и обновление состояния ----
    if (mode != TrackMode::Lost)
    {
        r.found = true;
        r.mode = mode;
        r.quad = project(cur.H, roi_);
        r.n_inliers = cur.inl;
        r.inliers = cur.cur;

        sharp_ema_ = sharp_ema_ <= 0.0 ? sharp : 0.8 * sharp_ema_ + 0.2 * sharp; // обновляем только по чётким кадрам
        last_quad_ = r.quad;
        have_pose_ = true;
        lost_run_ = 0;
        since_anchor_ = (mode == TrackMode::Orb) ? 0 : since_anchor_ + 1;
        prev_gray_ = gray.clone();
        prev_pts_ = cur.cur;
        ref_pts_ = cur.ref;
        const size_t cap = static_cast<size_t>(2 * p_.flow_points); // слишком много точек не нужно: поток дорожает
        if (prev_pts_.size() > cap)
        {
            std::vector<cv::Point2f> a, b;
            const double step = static_cast<double>(prev_pts_.size()) / cap;
            for (size_t i = 0; i < cap; ++i) { a.push_back(prev_pts_[static_cast<size_t>(i * step)]); b.push_back(ref_pts_[static_cast<size_t>(i * step)]); }
            prev_pts_ = std::move(a);
            ref_pts_ = std::move(b);
        }
        refill(prev_gray_, cur.H);
        r.n_tracked = static_cast<int>(prev_pts_.size());
    }
    else
    {
        ++lost_run_;
        ++since_anchor_;
        if (have_pose_ && lost_run_ <= p_.max_hold)
        {
            r.mode = TrackMode::Held; // prev_gray_ и точки сохраняем: следующий кадр снова попробует поток от них
            r.quad = last_quad_;
        }
        else
        {
            have_pose_ = false;
            prev_pts_.clear();
            ref_pts_.clear();
            prev_gray_.release();
            r.mode = TrackMode::Lost;
        }
    }
    r.ms_total = ms_since(t0, clk::now());
    return r;
}
