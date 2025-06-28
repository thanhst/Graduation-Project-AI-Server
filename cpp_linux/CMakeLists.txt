cmake_minimum_required(VERSION 3.10)
project(rtpimageproc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Pybind11
find_package(pybind11 REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)

# Thêm thư viện ONNX Runtime thủ công
set(ONNXRUNTIME_DIR "./lib/onnxruntime")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# FFmpeg headers (không cần find_package)
include_directories(/usr/include)

# Tạo module .so dùng cho Python
pybind11_add_module(rtpimageproc rtp_image_process.cpp)

# Link tất cả thư viện
target_link_libraries(rtpimageproc
    PRIVATE
    ${OpenCV_LIBS}
    onnxruntime
    avcodec avformat swscale avutil
)
