#pragma once
// Трекинг плоского объекта по ключевым точкам. Схема «обнаружение + сопровождение»:
//
//   ORB (поиск по эталону)  --->  точки объекта  --->  оптический поток Лукаса-Канаде (кадр -> кадр)
//        ^                                                      |
//        +------ повторная привязка к эталону / потеря ---------+
//
// Никакого UI и потоков: чистый алгоритм, запускается и из консоли. Подробно — в ALGORITHM.md.

#include <array>
#include <vector>
#include <opencv2/core.hpp>

// В OpenCV 5.x модули features2d / calib3d переименованы и разбиты. Подключаем то, что есть в сборке.
#if __has_include(<opencv2/features.hpp>)
#include <opencv2/features.hpp>
#endif
#if __has_include(<opencv2/features2d.hpp>)
#include <opencv2/features2d.hpp>
#endif
#if __has_include(<opencv2/calib3d.hpp>)
#include <opencv2/calib3d.hpp>
#endif
#if __has_include(<opencv2/geometry.hpp>)
#include <opencv2/geometry.hpp>
#endif
#if __has_include(<opencv2/calib.hpp>)
#include <opencv2/calib.hpp>
#endif
#include <opencv2/video/tracking.hpp> // calcOpticalFlowPyrLK

struct TrackerParams
{
    // --- ORB и сопоставление ---
    int    nfeatures     = 1500;  // максимум ключевых точек ORB на кадр / на шаблон
    float  scale_factor  = 1.2f;  // шаг пирамиды масштабов ORB
    int    nlevels       = 8;     // число уровней пирамиды
    float  ratio         = 0.75f; // тест Лоу при глобальном поиске: d1 < ratio * d2
    float  relaxed_ratio = 0.85f; // мягкий тест Лоу при поиске около прошлого положения (есть «подсказка»)
    double ransac_thr    = 3.0;   // допуск RANSAC, px (ошибка репроекции)
    int    min_inliers   = 12;    // глобальный поиск: меньше inlier-ов — объекта нет

    // --- область объекта на первом кадре (углы TL, TR, BR, BL в пикселях первого кадра) ---
    bool has_roi = false;                 // false: объект = весь первый кадр
    std::array<cv::Point2f, 4> roi{};

    // --- оптический поток и «память» ---
    int    min_flow_inliers = 8;    // поток считается успешным, если inlier-ов не меньше
    int    flow_points      = 150;  // целевое число отслеживаемых точек (меньше — добавляем новые)
    double fb_thr           = 1.5;  // порог прямо-обратной проверки потока, px
    int    reanchor_every   = 30;   // раз в столько кадров перепривязываемся к эталону (против дрейфа)
    int    reanchor_below   = 25;   // и сразу, если inlier-ов потока стало меньше
    int    max_hold         = 20;   // сколько кадров держим прошлую рамку, если объект потерян
    double min_ncc          = 0.35; // порог проверки по внешнему виду (корреляция участка кадра с эталоном)
    double blur_ratio       = 0.35; // кадр считается размытым, если его резкость < blur_ratio * недавняя резкость
};

enum class TrackMode
{
    Lost,  // объекта нет
    Orb,   // найден поиском по эталону (ORB + гомография)
    Flow,  // найден оптическим потоком от предыдущего удачного кадра
    Held   // потерян, показана последняя известная рамка (до max_hold кадров)
};

struct TrackResult
{
    bool found = false;                 // Orb или Flow
    TrackMode mode = TrackMode::Lost;
    std::array<cv::Point2f, 4> quad{};  // углы рамки в кадре: TL, TR, BR, BL (валидны при found и при Held)
    int n_kp = 0;                       // ключевых точек ORB в кадре (0, если ORB не запускался)
    int n_matches = 0;                  // совпадений после теста Лоу (лучший шаблон)
    int n_inliers = 0;                  // inlier-ов гомографии
    int n_tracked = 0;                  // точек в наборе для потока после шага
    double ms_detect = 0, ms_match = 0, ms_ransac = 0, ms_flow = 0, ms_total = 0;
    std::vector<cv::Point2f> inliers;   // положения inlier-ов в кадре (для отрисовки)
};

class ObjectTracker
{
public:
    // reference — первый кадр видео. Объект — params.roi (или весь кадр).
    // Возвращает false, если в области объекта слишком мало ключевых точек (объект без текстуры).
    bool init(const cv::Mat& reference, const TrackerParams& params);

    TrackResult process(const cv::Mat& frame);

    // Забыть историю (положение, точки, предыдущий кадр); шаблоны эталона остаются.
    void reset_tracking();

    int ref_keypoints() const;
    const TrackerParams& params() const { return p_; }

private:
    // Эталон в одном масштабе: ORB-точки уменьшенной копии и их координаты в системе ПЕРВОГО кадра.
    struct Template
    {
        std::vector<cv::KeyPoint> kp;
        cv::Mat desc;
        std::vector<cv::Point2f> ref_pt; // = kp.pt / scale
    };
    // Кандидат на положение объекта.
    struct Cand
    {
        cv::Mat H;                       // 3x3, CV_64F: координаты первого кадра -> координаты текущего
        int inl = 0;
        int n_matches = 0;
        std::vector<cv::Point2f> cur;    // inlier-ы в текущем кадре
        std::vector<cv::Point2f> ref;    // те же inlier-ы в координатах первого кадра
    };

    Cand match_templates(const std::vector<cv::KeyPoint>& kp, const cv::Mat& desc, float ratio,
                         const std::vector<cv::Point2f>* region, double& ms_match, double& ms_ransac) const;
    bool track_flow(const cv::Mat& gray, Cand& out) const;
    double appearance_ncc(const cv::Mat& gray, const cv::Mat& H) const; // сходство участка кадра с эталоном, [-1; 1]
    double sharpness(const cv::Mat& gray) const;                        // резкость: дисперсия лапласиана в области объекта
    void refill(const cv::Mat& gray, const cv::Mat& H);

    TrackerParams p_;
    cv::Ptr<cv::ORB> orb_;
    cv::BFMatcher matcher_{cv::NORM_HAMMING, false};
    std::vector<Template> templates_;
    std::array<cv::Point2f, 4> roi_{};
    cv::Mat P_;          // 3x3: координаты первого кадра -> «выпрямленный» участок объекта 128x128
    cv::Mat ref_patch_;  // эталонный участок объекта 32x32 (для проверки по внешнему виду)

    // ---- состояние сопровождения ----
    bool have_pose_ = false;                 // есть последнее известное положение (для подсказки и удержания)
    std::array<cv::Point2f, 4> last_quad_{};
    cv::Mat prev_gray_;                      // последний УДАЧНЫЙ кадр (поток всегда считается от него)
    std::vector<cv::Point2f> prev_pts_;      // отслеживаемые точки на prev_gray_
    std::vector<cv::Point2f> ref_pts_;       // те же точки в координатах первого кадра
    double sharp_ema_ = 0.0;                 // скользящее среднее резкости по удачным чётким кадрам
    int lost_run_ = 0;                       // подряд кадров без объекта
    int since_anchor_ = 0;                   // кадров с последней привязки к эталону
};

// Выпуклый четырёхугольник разумной площади (отсекает вырожденные гомографии).
bool plausible_quad(const std::array<cv::Point2f, 4>& q, cv::Size frame_size);

// IoU двух четырёхугольников, растровый (метрика качества при известной истинной рамке).
double quad_iou(const std::array<cv::Point2f, 4>& a, const std::array<cv::Point2f, 4>& b);
