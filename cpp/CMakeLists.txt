cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
project(rtpimageproc)

#find package of python3
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${pybind11_DIR})
find_package(pybind11 REQUIRED)

# OpenCV
set(OPENCV_DIR "./lib/opencv")
include_directories(${OPENCV_DIR}/include)
link_directories(${OPENCV_DIR}/lib)

# ONNX Runtime
set(ONNXRUNTIME_DIR "./lib/onnxruntime")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

set(FFMPEG_DIR "./lib/ffmpeg")
include_directories(${FFMPEG_DIR}/include)
link_directories(${FFMPEG_DIR}/lib)

set(FFMPEG_LIBS
    avcodec
    avformat
    avutil
    swscale
)

add_library(rtpimageproc MODULE rtp_image_process.cpp)
target_link_libraries(rtpimageproc
    PRIVATE
    pybind11::module
    opencv_world4110
    onnxruntime
    ${FFMPEG_LIBS}
)

target_compile_definitions(rtpimageproc PRIVATE VERSION_INFO=0.1.0)

# Đặt tên file output là .pyd thay vì .so/.dll
set_target_properties(rtpimageproc PROPERTIES PREFIX "" SUFFIX ".pyd")
