cmake_minimum_required(VERSION 3.10)
project(rtpimageproc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 REQUIRED)

find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_DIR "./lib/onnxruntime")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

include_directories(/usr/include)

pybind11_add_module(rtpimageproc rtp_image_process.cpp)

target_link_libraries(rtpimageproc
    PRIVATE
    ${OpenCV_LIBS}
    onnxruntime
    avcodec avformat swscale avutil
)
