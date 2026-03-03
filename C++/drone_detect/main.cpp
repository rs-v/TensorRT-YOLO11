/**
 * 无人机部署主程序
 *
 * 功能：通过 SpireCV Pro SDK 读取无人机摄像头视频流，
 *       使用 YOLO11 TensorRT 进行实时目标检测，
 *       并将处理后的视频流推送到指定的推流节点（RTMP/GStreamer）。
 *
 * SpireCV Pro SDK: https://gitee.com/spirecv/spirecv-pro.git
 *
 * 用法:
 *   ./drone_detect [摄像头ID或流地址] [推流URL]
 *
 * 示例:
 *   ./drone_detect 0 rtmp://localhost:1935/live/stream
 *   ./drone_detect rtsp://192.168.1.1:8554/main rtmp://192.168.1.100:1935/live/drone
 */

#include <iostream>
#include <string>
#include <chrono>
#include <csignal>
#include <sstream>

#include <opencv2/opencv.hpp>

// SpireCV Pro SDK 头文件（需安装 SpireCV Pro SDK）
// https://gitee.com/spirecv/spirecv-pro.git
#ifdef HAVE_SPIRECV
#include "spirecv/camera/camera.h"
#include "spirecv/media/video_streamer.h"
#endif

#include "infer.h"

using namespace cv;
using namespace std;


static volatile bool g_running = true;

void signal_handler(int sig)
{
    g_running = false;
}


/**
 * 构建用于 RTMP 推流的 GStreamer 管道字符串（FFmpeg 后端回退方案）。
 */
string build_gst_pipeline(const string& stream_url, int width, int height, int fps)
{
    ostringstream oss;
    oss << "appsrc ! videoconvert ! video/x-raw,format=I420 ! "
        << "x264enc speed-preset=ultrafast tune=zerolatency ! "
        << "flvmux streamable=true ! rtmpsink location=" << stream_url;
    return oss.str();
}


int run(const string& camera_source, const string& stream_url,
        const string& trt_plan, int gpu_id = 0,
        int width = 1280, int height = 720, int fps = 30)
{
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // -------- 打开无人机摄像头 --------
    VideoCapture cap;

#ifdef HAVE_SPIRECV
    // 使用 SpireCV Pro SDK 打开摄像头
    spirecv::camera::Camera spire_cam;
    bool spire_opened = false;
    try {
        int cam_id = 0;
        bool is_url = (camera_source.find("://") != string::npos);
        if (!is_url) {
            cam_id = stoi(camera_source);
        }
        if (is_url) {
            spire_opened = spire_cam.open(camera_source);
        } else {
            spire_opened = spire_cam.open(cam_id);
        }
    } catch (...) {
        spire_opened = false;
    }

    if (!spire_opened) {
        cout << "SpireCV camera open failed, falling back to OpenCV." << endl;
        cap.open(camera_source);
    } else {
        cout << "SpireCV Pro camera opened: " << camera_source << endl;
    }
#else
    // 回退：使用 OpenCV VideoCapture（支持设备 ID 或流 URL）
    bool is_numeric = false;
    try {
        size_t pos = 0;
        stoi(camera_source, &pos);
        is_numeric = (pos == camera_source.size());
    } catch (...) {
        is_numeric = false;
    }
    if (is_numeric) {
        cap.open(stoi(camera_source));
    } else {
        cap.open(camera_source);
    }
#endif

#ifdef HAVE_SPIRECV
    if (!spire_opened && !cap.isOpened()) {
#else
    if (!cap.isOpened()) {
#endif
        cerr << "Error: Cannot open camera source: " << camera_source << endl;
        return -1;
    }

    // 读取实际帧尺寸
#ifdef HAVE_SPIRECV
    if (spire_opened) {
        width  = spire_cam.getWidth()  > 0 ? spire_cam.getWidth()  : width;
        height = spire_cam.getHeight() > 0 ? spire_cam.getHeight() : height;
    } else {
#endif
        cap.set(CAP_PROP_FRAME_WIDTH,  width);
        cap.set(CAP_PROP_FRAME_HEIGHT, height);
        cap.set(CAP_PROP_FPS, fps);
        width  = (int)cap.get(CAP_PROP_FRAME_WIDTH);
        height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
#ifdef HAVE_SPIRECV
    }
#endif

    cout << "Camera: " << width << "x" << height << " @ " << fps << " fps" << endl;

    // -------- 初始化推流输出 --------
    VideoWriter writer;

#ifdef HAVE_SPIRECV
    // 使用 SpireCV Pro SDK 推流
    spirecv::media::VideoStreamer spire_streamer;
    bool spire_stream_opened = false;
    try {
        spire_stream_opened = spire_streamer.open(stream_url, fps, width, height);
    } catch (...) {
        spire_stream_opened = false;
    }

    if (!spire_stream_opened) {
        cout << "SpireCV VideoStreamer failed, falling back to GStreamer." << endl;
    } else {
        cout << "SpireCV Pro streaming to: " << stream_url << endl;
    }

    if (!spire_stream_opened) {
#endif
        // 回退：GStreamer RTMP 推流
        string gst_pipeline = build_gst_pipeline(stream_url, width, height, fps);
        writer.open(gst_pipeline, CAP_GSTREAMER, 0, fps, Size(width, height));
        if (!writer.isOpened()) {
            cerr << "Warning: GStreamer pipeline failed. "
                 << "Processed frames will not be streamed." << endl;
        } else {
            cout << "GStreamer streaming to: " << stream_url << endl;
        }
#ifdef HAVE_SPIRECV
    }
#endif

    // -------- 初始化 YOLO11 TensorRT 检测器 --------
    YoloDetector detector(trt_plan, gpu_id, 0.45f, 0.25f);
    cout << "YOLO11 TensorRT model loaded: " << trt_plan << endl;

    cout << "Pipeline: camera -> YOLO11 TensorRT -> stream(" << stream_url << ")" << endl;

    Mat frame;
    int frame_count = 0;
    long total_us = 0;

    // -------- 主循环：读帧 -> 检测 -> 推流 --------
    while (g_running) {
        bool ret = false;

#ifdef HAVE_SPIRECV
        if (spire_opened) {
            ret = spire_cam.read(frame);
        } else {
            ret = cap.read(frame);
        }
#else
        ret = cap.read(frame);
#endif

        if (!ret || frame.empty()) {
            cerr << "Warning: Failed to read frame, retrying..." << endl;
            continue;
        }

        frame_count++;

        auto t0 = chrono::system_clock::now();

        // YOLO11 推理
        vector<Detection> results = detector.inference(frame);

        auto t1 = chrono::system_clock::now();
        long elapsed_us = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
        total_us += elapsed_us;

        // 绘制检测结果
        YoloDetector::draw_image(frame, results);

        // 叠加帧率和检测数量
        int avg_fps = (total_us > 0) ? (int)(frame_count * 1000000L / total_us) : 0;
        putText(frame,
                format("FPS: %d  Objects: %d", avg_fps, (int)results.size()),
                Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8,
                Scalar(0, 255, 0), 2, LINE_AA);

        if (frame_count % 30 == 0) {
            cout << "Frame " << frame_count
                 << ": FPS=" << avg_fps
                 << ", Detections=" << results.size()
                 << ", Infer=" << elapsed_us / 1000 << "ms" << endl;
        }

        // 推送处理后的帧到推流节点
#ifdef HAVE_SPIRECV
        if (spire_stream_opened) {
            spire_streamer.write(frame);
        } else if (writer.isOpened()) {
            writer.write(frame);
        }
#else
        if (writer.isOpened()) {
            writer.write(frame);
        }
#endif
    }

    // -------- 释放资源 --------
    cout << "Releasing resources... Processed " << frame_count << " frames." << endl;

#ifdef HAVE_SPIRECV
    if (spire_opened)       spire_cam.release();
    if (spire_stream_opened) spire_streamer.release();
#endif
    if (cap.isOpened())    cap.release();
    if (writer.isOpened()) writer.release();

    if (frame_count > 0 && total_us > 0) {
        cout << "Average FPS: " << frame_count * 1000000L / total_us << endl;
    }

    return 0;
}


int main(int argc, char* argv[])
{
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " [camera_source] [stream_url] [trt_plan(optional)] [gpu_id(optional)]" << endl;
        cerr << "  camera_source : 摄像头 ID（如 0）或流地址（如 rtsp://192.168.1.1/main）" << endl;
        cerr << "  stream_url    : 推流目标地址（如 rtmp://localhost:1935/live/stream）" << endl;
        cerr << "  trt_plan      : TensorRT plan 文件路径（默认 ../detect/build/yolo11s.plan）" << endl;
        cerr << "  gpu_id        : GPU 设备 ID（默认 0）" << endl;
        cerr << endl;
        cerr << "Example:" << endl;
        cerr << "  " << argv[0] << " 0 rtmp://localhost:1935/live/stream" << endl;
        cerr << "  " << argv[0] << " rtsp://192.168.1.1:8554/main rtmp://192.168.1.100:1935/live/drone" << endl;
        return -1;
    }

    string camera_source = argv[1];
    string stream_url    = argv[2];
    string trt_plan      = (argc > 3) ? argv[3] : "../detect/build/yolo11s.plan";
    int    gpu_id        = (argc > 4) ? atoi(argv[4]) : 0;

    return run(camera_source, stream_url, trt_plan, gpu_id);
}
