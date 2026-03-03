# TensorRT 部署 YOLO11 目标检测、关键点检测、实例分割、目标跟踪

## 一. 项目简介

- 基于 `TensorRT-v8` ，部署`YOLO11` 目标检测、关键点检测、实例分割、目标跟踪 4 项任务；
- 支持在 `Jetson` 系列、 `Linux x86_64` 服务器上部署；

- 无需安装支持`cuda`的`OpenCV`，前后处理的张量操作都是作者通过`cuda`编程实现；
- 模型转换方式：`.pth` -> `.onnx` -> `.plan(.engine)`；
- 作者使用 `Python` 和 `C++` 2 种 `api` 分别做了实现；
- 均采用了面向对象的方式，便于结合到其他项目当中；
- `C++` 版本的还会编译为动态链接库，便于在其他项目种作为接口调用；

## 二. 项目效果

|               原图                |               目标检测                |
| :-------------------------------: | :-----------------------------------: |
|      ![004](assets/005.jpeg)      | ![004_detect](assets/005_detect.jpeg) |
|          **关键点检测**           |             **实例分割**              |
| ![004_pose](assets/005_pose.jpeg) |    ![004_seg](assets/005_seg.jpeg)    |

- ByteTrack目标跟踪

![result](./assets/result.gif)

## 三. 推理速度

|        | detect | pose  | segment |
| :----: | :----: | :---: | :-----: |
|  C++   |  3 ms  | 4 ms  |  6 ms   |
| python | 10 ms  | 13 ms |  45 ms  |

- 这里的推理时间包含前处理、模型推理、后处理
- 这里基于 `x86_64 Linux ` 服务器，`Ubuntu`系统，显卡为`GeForce RTX 2080 Ti`

## 四. 环境配置

1. 基本要求：

- `TensorRT 8.0+`
- `OpenCV 3.4.0+`

**如果基本要求已满足，可直接进入各目录下运行各任务**

**环境构建可以参考下面内容：**

2. 如果是 `Linux x86_64` 服务器上，建议使用 `docker`

- 具体环境构建，可参考这个链接 [构建TensorRT环境](https://github.com/emptysoal/tensorrt-experiment) 的环境构建部分，也是作者的项目

3. 如果是边缘设备，如：`Jetson Nano`

- 烧录 `Jetpack 4.6.1 ` 系统镜像，网上烧录镜像的资料还是很多的，这里就不赘述了
- `Jetpack 4.6.1 ` 系统镜像原装环境如下：

| CUDA | cuDNN | TensorRT | OpenCV |
| ---- | ----- | -------- | ------ |
| 10.2 | 8.2   | 8.2.1    | 4.1.1  |

## 五. 项目运行

- 本项目`Python`和`C++`目录下均包含`detect`、`pose` 和 `segment`；
- 按照各自目录下的 `README` 分别实现目标检测、关键点检测、实例分割 、目标跟踪 4 种任务。

- [C++ api detect](https://github.com/emptysoal/TensorRT-YOLO11/tree/main/C%2B%2B/detect)
- [C++ api pose](https://github.com/emptysoal/TensorRT-YOLO11/tree/main/C%2B%2B/pose)
- [C++ api segment](https://github.com/emptysoal/TensorRT-YOLO11/tree/main/C%2B%2B/segment)
- [C++ api track](https://github.com/emptysoal/TensorRT-YOLO11/tree/main/C%2B%2B/)
- [Python api detect](https://github.com/emptysoal/TensorRT-YOLO11/tree/main/python/detect)
- [Python api pose](https://github.com/emptysoal/TensorRT-YOLO11/tree/main/python/pose)
- [Python api segment](https://github.com/emptysoal/TensorRT-YOLO11/tree/main/python/segment)
- [Python api track](https://github.com/emptysoal/TensorRT-YOLO11/tree/main/python)

## 六. 无人机部署（SpireCV Pro SDK）

本项目支持集成 [SpireCV Pro SDK](https://gitee.com/spirecv/spirecv-pro.git) 进行无人机端部署，实现从无人机摄像头读取视频流、YOLO11 实时检测，并将处理后的视频流推送到指定推流节点。

### 6.1 功能说明

- **视频流读取**：优先使用 SpireCV Pro SDK 读取无人机摄像头；若 SDK 不可用，自动回退至 OpenCV VideoCapture（支持 USB 摄像头、RTSP 流）
- **实时目标检测**：基于 YOLO11 TensorRT 进行推理，叠加检测框和置信度信息
- **视频推流**：优先使用 SpireCV Pro SDK 推流；若不可用，回退至 FFmpeg（Python）或 GStreamer（C++）进行 RTMP 推流

### 6.2 环境依赖

- SpireCV Pro SDK（可选，推荐）：https://gitee.com/spirecv/spirecv-pro.git
- FFmpeg（Python 回退方案）：`apt install ffmpeg`
- GStreamer + gstreamer1.0-plugins-bad（C++ 回退方案，需 OpenCV 支持 GStreamer）

### 6.3 Python 版使用

```bash
cd python

# 基本用法（读取本地摄像头 0，推流到本地 RTMP 服务器）
python drone_detect.py \
    --detect-model ./detect/model.plan \
    --camera-id 0 \
    --stream-url rtmp://localhost:1935/live/stream

# 通过 RTSP 流读取无人机摄像头
python drone_detect.py \
    --detect-model ./detect/model.plan \
    --camera-url rtsp://192.168.1.1:8554/main \
    --stream-url rtmp://192.168.1.100:1935/live/drone \
    --width 1280 --height 720 --fps 30
```

**参数说明**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--detect-model` | TensorRT plan 文件路径 | `./detect/model.plan` |
| `--camera-id` | 摄像头设备 ID | `0` |
| `--camera-url` | 摄像头流地址（优先于 `--camera-id`） | `""` |
| `--stream-url` | 推流目标地址（RTMP URL） | `rtmp://localhost:1935/live/stream` |
| `--width` | 视频宽度 | `1280` |
| `--height` | 视频高度 | `720` |
| `--fps` | 推流帧率 | `30` |
| `--gpu-id` | GPU 设备 ID | `0` |
| `--conf-thresh` | 置信度阈值 | `0.25` |
| `--nms-thresh` | NMS IoU 阈值 | `0.45` |

### 6.4 C++ 版使用

```bash
cd C++/drone_detect
mkdir build && cd build

# 如已安装 SpireCV Pro SDK，编辑 CMakeLists.txt 取消 SPIRECV_ROOT 相关注释
cmake ..
make -j$(nproc)

# 运行（摄像头 ID 或 RTSP 流地址 + RTMP 推流地址）
./drone_detect 0 rtmp://localhost:1935/live/stream
./drone_detect rtsp://192.168.1.1:8554/main rtmp://192.168.1.100:1935/live/drone
```

### 6.5 SpireCV Pro SDK 集成说明

当系统安装了 SpireCV Pro SDK 后：

**Python**：`import spirecv` 自动生效，`DroneCamera` 和 `DroneStreamer` 类会自动切换至 SDK 接口。

**C++**：在 `C++/drone_detect/CMakeLists.txt` 中取消以下注释并设置正确路径：

```cmake
set(SPIRECV_ROOT "/usr/local/spirecv")
# ... 取消对应注释块
```

然后重新编译即可启用 `HAVE_SPIRECV` 宏，代码会自动使用 SpireCV Pro 的摄像头和推流接口。
