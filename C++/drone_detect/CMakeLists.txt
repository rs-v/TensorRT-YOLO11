cmake_minimum_required(VERSION 3.10)

project(drone_detect)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE release)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# ============= cuda ============
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# ============= tensorrt ============
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("Embed_platform on (Jetson / aarch64)")
  include_directories(/usr/include/aarch64-linux-gnu)
  link_directories(/usr/lib/aarch64-linux-gnu)
else()
  message("Embed_platform off (x86_64)")
  include_directories(/usr/include/x86_64-linux-gnu)
  link_directories(/usr/lib/x86_64-linux-gnu)
endif()

# ============ opencv ============
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ============ yolo11 infer shared lib ============
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../detect/include)
file(GLOB_RECURSE YOLO_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/../detect/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../detect/src/*.cu
)
cuda_add_library(yolo_infer SHARED ${YOLO_SRCS})
target_link_libraries(yolo_infer cudart nvinfer nvonnxparser ${OpenCV_LIBS})

# ============ SpireCV Pro SDK（可选）============
# 默认不启用 SpireCV（使用 OpenCV/GStreamer 回退）
# 如已安装 SpireCV Pro SDK（https://gitee.com/spirecv/spirecv-pro.git），
# 请取消注释下方整块内容并设置正确的安装路径：
#
# set(SPIRECV_ROOT "/usr/local/spirecv")
# if (EXISTS "${SPIRECV_ROOT}/include/spirecv/camera/camera.h")
#     message("SpireCV Pro SDK found at ${SPIRECV_ROOT}")
#     add_definitions(-DHAVE_SPIRECV)
#     include_directories(${SPIRECV_ROOT}/include)
#     link_directories(${SPIRECV_ROOT}/lib)
#     set(SPIRECV_LIBS spirecv_camera spirecv_media)
# else()
#     message("SpireCV Pro SDK not found at ${SPIRECV_ROOT}, using OpenCV/GStreamer fallback.")
#     set(SPIRECV_LIBS "")
# endif()

# 未启用 SpireCV 时的默认值（启用 SpireCV 后，此行应删除或注释）
set(SPIRECV_LIBS "")

# ======== drone_detect executable ========
add_executable(drone_detect ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(drone_detect yolo_infer ${SPIRECV_LIBS})
