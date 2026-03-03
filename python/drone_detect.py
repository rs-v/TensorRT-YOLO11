# -*- coding: utf-8 -*-
"""
无人机部署脚本：基于 SpireCV Pro SDK 读取摄像头视频流，运行 YOLO11 TensorRT 目标检测，
并将处理后的视频流推送到推流节点。

SpireCV Pro SDK: https://gitee.com/spirecv/spirecv-pro.git

用法:
    python drone_detect.py [参数]

示例:
    python drone_detect.py --stream-url rtmp://localhost:1935/live/stream
    python drone_detect.py --camera-id 0 --stream-url rtmp://192.168.1.100:1935/live/drone
"""

import os
import sys
import time
import argparse
import subprocess

import cv2
import numpy as np

# 尝试导入 SpireCV Pro SDK（用于无人机摄像头读取和视频推流）
try:
    import spirecv
    SPIRECV_AVAILABLE = True
    print("SpireCV Pro SDK detected.")
except ImportError:
    SPIRECV_AVAILABLE = False
    print("SpireCV Pro SDK not found, falling back to OpenCV for camera input and GStreamer/FFmpeg for streaming.")

# 引入 YOLO11 检测模块（与 python/detect/ 目录下保持一致）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "detect"))
from infer import YoloDetector


def make_parser():
    parser = argparse.ArgumentParser(
        description="YOLO11 TensorRT 无人机目标检测：读取 SpireCV Pro 摄像头，推流到指定节点"
    )
    parser.add_argument(
        "--detect-model",
        type=str,
        default="./detect/model.plan",
        help="YOLO11 TensorRT plan 文件路径",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="摄像头设备 ID（SpireCV Pro 摄像头索引或 USB 摄像头 ID）",
    )
    parser.add_argument(
        "--camera-url",
        type=str,
        default="",
        help="摄像头 RTSP/HTTP 流地址（优先级高于 --camera-id）",
    )
    parser.add_argument(
        "--stream-url",
        type=str,
        default="rtmp://localhost:1935/live/stream",
        help="视频推流目标地址（RTMP URL）",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="视频帧宽度（像素）",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="视频帧高度（像素）",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="视频推流帧率",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="推理所用 GPU 设备 ID",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.25,
        help="目标检测置信度阈值",
    )
    parser.add_argument(
        "--nms-thresh",
        type=float,
        default=0.45,
        help="NMS IoU 阈值",
    )
    return parser


class DroneCamera:
    """
    无人机摄像头封装类。
    优先使用 SpireCV Pro SDK 读取摄像头，不可用时回退至 OpenCV VideoCapture。
    """

    def __init__(self, camera_id=0, camera_url="", width=1280, height=720, fps=30):
        self._width = width
        self._height = height
        self._fps = fps
        self._cap = None

        if camera_url:
            # 通过 URL（RTSP/HTTP）打开摄像头
            self._open_by_url(camera_url)
        elif SPIRECV_AVAILABLE:
            self._open_spirecv(camera_id)
        else:
            self._open_opencv(camera_id)

    def _open_spirecv(self, camera_id):
        """使用 SpireCV Pro SDK 打开无人机摄像头。"""
        try:
            self._cap = spirecv.Camera()
            self._cap.open(camera_id)
            if not self._cap.isOpened():
                raise RuntimeError("SpireCV Camera open failed")
            print(f"SpireCV Pro camera {camera_id} opened successfully.")
        except Exception as e:
            print(f"SpireCV camera open error: {e}, falling back to OpenCV.")
            self._open_opencv(camera_id)

    def _open_opencv(self, camera_id):
        """使用 OpenCV VideoCapture 打开摄像头。"""
        self._cap = cv2.VideoCapture(camera_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera with ID {camera_id}")
        print(f"OpenCV camera {camera_id} opened (fallback).")

    def _open_by_url(self, url):
        """通过流地址（RTSP/HTTP）打开摄像头。"""
        self._cap = cv2.VideoCapture(url)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera stream: {url}")
        print(f"Camera stream opened: {url}")

    def isOpened(self):
        return self._cap is not None and self._cap.isOpened()

    def read(self):
        """读取一帧，返回 (success: bool, frame: np.ndarray)。"""
        if self._cap is None:
            return False, None
        return self._cap.read()

    def get_width(self):
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_height(self):
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class DroneStreamer:
    """
    视频推流封装类。
    优先使用 SpireCV Pro SDK 推流，不可用时回退至 GStreamer/FFmpeg 管道推流（RTMP）。
    """

    def __init__(self, stream_url, fps, width, height):
        self._stream_url = stream_url
        self._fps = fps
        self._width = width
        self._height = height
        self._writer = None
        self._ffmpeg_proc = None

        if SPIRECV_AVAILABLE:
            self._open_spirecv()
        else:
            self._open_ffmpeg()

    def _open_spirecv(self):
        """使用 SpireCV Pro SDK 打开推流。"""
        try:
            self._writer = spirecv.VideoStreamer(
                self._stream_url, self._fps, self._width, self._height
            )
            if not self._writer.isOpened():
                raise RuntimeError("SpireCV VideoStreamer open failed")
            print(f"SpireCV Pro streaming to {self._stream_url}")
        except Exception as e:
            print(f"SpireCV VideoStreamer error: {e}, falling back to FFmpeg.")
            self._open_ffmpeg()

    def _open_ffmpeg(self):
        """使用 FFmpeg 子进程向 RTMP 服务器推流。"""
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self._width}x{self._height}",
            "-r", str(self._fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-f", "flv",
            self._stream_url,
        ]
        try:
            self._ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            print(f"FFmpeg streaming to {self._stream_url}")
        except FileNotFoundError:
            print("Warning: FFmpeg not found. Processed frames will not be streamed.")
            self._ffmpeg_proc = None

    def write(self, frame):
        """
        向推流节点发送一帧。
        frame: BGR 格式的 numpy.ndarray (H, W, 3)
        """
        if frame is None:
            return

        # 确保帧尺寸与设置一致
        if frame.shape[1] != self._width or frame.shape[0] != self._height:
            frame = cv2.resize(frame, (self._width, self._height))

        if self._writer is not None:
            self._writer.write(frame)
        elif self._ffmpeg_proc is not None and self._ffmpeg_proc.stdin:
            try:
                self._ffmpeg_proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("Warning: FFmpeg pipe broken, streaming stopped.")
                self._ffmpeg_proc = None

    def release(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._ffmpeg_proc is not None:
            try:
                self._ffmpeg_proc.stdin.close()
                self._ffmpeg_proc.wait(timeout=5)
            except Exception:
                self._ffmpeg_proc.terminate()
            self._ffmpeg_proc = None


def main(args):
    # -------- 初始化 YOLO11 TensorRT 检测器 --------
    if not os.path.exists(args.detect_model):
        print(f"Error: Detection model not found: {args.detect_model}")
        sys.exit(1)

    print(f"Loading YOLO11 TensorRT model: {args.detect_model}")
    detector = YoloDetector(
        trt_plan=args.detect_model,
        gpu_id=args.gpu_id,
        nms_thresh=args.nms_thresh,
        conf_thresh=args.conf_thresh,
    )
    print("Model loaded successfully.")

    # -------- 打开无人机摄像头 --------
    try:
        camera = DroneCamera(
            camera_id=args.camera_id,
            camera_url=args.camera_url,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
    except RuntimeError as e:
        print(f"Error: {e}")
        detector.release()
        sys.exit(1)

    # 读取实际帧尺寸（摄像头可能覆盖设置值）
    try:
        actual_width = camera.get_width() or args.width
        actual_height = camera.get_height() or args.height
    except Exception:
        actual_width = args.width
        actual_height = args.height

    # -------- 打开推流节点 --------
    streamer = DroneStreamer(
        stream_url=args.stream_url,
        fps=args.fps,
        width=actual_width,
        height=actual_height,
    )

    print(
        f"Pipeline started: camera({actual_width}x{actual_height}@{args.fps}fps)"
        f" -> YOLO11 TensorRT -> stream({args.stream_url})"
    )

    frame_count = 0
    total_time = 0.0

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("Warning: Failed to read frame from camera, retrying...")
                time.sleep(0.01)
                continue

            frame_count += 1
            t0 = time.time()

            # -------- YOLO11 推理 --------
            detect_res = detector.inference(frame)

            # -------- 绘制检测结果 --------
            YoloDetector.draw_image(detect_res, frame)

            elapsed = time.time() - t0
            total_time += elapsed
            current_fps = frame_count / total_time

            # 在画面上叠加帧率和检测数量信息
            cv2.putText(
                frame,
                f"FPS: {current_fps:.1f}  Objects: {len(detect_res)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if frame_count % 30 == 0:
                print(
                    f"Frame {frame_count}: FPS={current_fps:.1f}, "
                    f"Detections={len(detect_res)}, "
                    f"Infer={elapsed * 1000:.1f}ms"
                )

            # -------- 推送处理后的帧到推流节点 --------
            streamer.write(frame)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    # -------- 释放资源 --------
    print(f"Releasing resources... Processed {frame_count} frames.")
    streamer.release()
    camera.release()
    detector.release()

    if frame_count > 0 and total_time > 0:
        print(f"Average FPS: {frame_count / total_time:.1f}")


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
