#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <filesystem>

#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

#include <mutex>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

namespace py = pybind11;

constexpr int IMG_SIZE = 48;

std::filesystem::path get_module_dir()
{
#ifdef _WIN32
    HMODULE hMod = nullptr;
    GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                      GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                      (LPCSTR)&get_module_dir, &hMod);
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(hMod, path, MAX_PATH);
    return std::filesystem::path(path).parent_path();
#else
    Dl_info dl_info;
    dladdr((void *)&get_module_dir, &dl_info);
    return std::filesystem::path(dl_info.dli_fname).parent_path();
#endif
}


// ONNX Runtime setup
Ort::Env *env = nullptr;
Ort::Session *session = nullptr;
static std::mutex init_mutex;

void init_onnx()
{
    std::lock_guard<std::mutex> lock(init_mutex);
    if (!env)
        env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "RTPInference");

    if (!session)
    {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        std::filesystem::path model_path = get_module_dir() / "lib" / "model" / "cnn_fer.onnx";
        std::string model_path_str = model_path.string();
        session = new Ort::Session(*env, model_path_str.c_str(), options);
    }
}

Ort::Session &get_session()
{
    if (!session)
        init_onnx();
    return *session;
}

void cleanup_onnx()
{
    delete session;
    delete env;
    session = nullptr;
    env = nullptr;
}

// Simulated decode
cv::Mat decode_h264(const std::vector<uint8_t> &encoded)
{

    const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!codec)
        throw std::runtime_error("H264 codec not found");

    AVCodecContext *ctx = avcodec_alloc_context3(codec);
    if (!ctx)
        throw std::runtime_error("Failed to allocate codec context");

    if (avcodec_open2(ctx, codec, nullptr) < 0)
        throw std::runtime_error("Could not open codec");

    AVPacket *pkt = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    AVFrame *rgb = av_frame_alloc();
    pkt->data = const_cast<uint8_t *>(encoded.data());
    pkt->size = static_cast<int>(encoded.size());

    int ret = avcodec_send_packet(ctx, pkt);
    if (ret < 0)
        throw std::runtime_error("Error sending packet");

    ret = avcodec_receive_frame(ctx, frame);
    if (ret < 0)
        throw std::runtime_error("Error receiving frame");

    SwsContext *sws = sws_getContext(frame->width, frame->height, (AVPixelFormat)frame->format,
                                     frame->width, frame->height, AV_PIX_FMT_BGR24,
                                     SWS_BILINEAR, nullptr, nullptr, nullptr);

    int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_BGR24, frame->width, frame->height, 1);
    std::vector<uint8_t> buffer(num_bytes);
    av_image_fill_arrays(rgb->data, rgb->linesize, buffer.data(), AV_PIX_FMT_BGR24, frame->width, frame->height, 1);

    sws_scale(sws, frame->data, frame->linesize, 0, frame->height, rgb->data, rgb->linesize);

    cv::Mat img(frame->height, frame->width, CV_8UC3, buffer.data());

    // cleanup
    sws_freeContext(sws);
    av_frame_free(&frame);
    av_frame_free(&rgb);
    av_packet_free(&pkt);
    avcodec_free_context(&ctx);

    return img.clone(); // clone to own memory
}

cv::Mat decode_vp8(const std::vector<uint8_t> &encoded)
{

    const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_VP8);
    if (!codec)
        throw std::runtime_error("VP8 codec not found");

    AVCodecContext *ctx = avcodec_alloc_context3(codec);
    if (!ctx)
        throw std::runtime_error("Failed to allocate codec context");

    if (avcodec_open2(ctx, codec, nullptr) < 0)
        throw std::runtime_error("Could not open codec");

    AVPacket *pkt = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    AVFrame *rgb = av_frame_alloc();
    pkt->data = const_cast<uint8_t *>(encoded.data());
    pkt->size = static_cast<int>(encoded.size());

    int ret = avcodec_send_packet(ctx, pkt);
    if (ret < 0)
        throw std::runtime_error("Error sending packet");

    ret = avcodec_receive_frame(ctx, frame);
    if (ret < 0)
        throw std::runtime_error("Error receiving frame");

    SwsContext *sws = sws_getContext(frame->width, frame->height, (AVPixelFormat)frame->format,
                                     frame->width, frame->height, AV_PIX_FMT_BGR24,
                                     SWS_BILINEAR, nullptr, nullptr, nullptr);

    int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_BGR24, frame->width, frame->height, 1);
    std::vector<uint8_t> buffer(num_bytes);
    av_image_fill_arrays(rgb->data, rgb->linesize, buffer.data(), AV_PIX_FMT_BGR24, frame->width, frame->height, 1);

    sws_scale(sws, frame->data, frame->linesize, 0, frame->height, rgb->data, rgb->linesize);

    cv::Mat img(frame->height, frame->width, CV_8UC3, buffer.data());

    // cleanup
    sws_freeContext(sws);
    av_frame_free(&frame);
    av_frame_free(&rgb);
    av_packet_free(&pkt);
    avcodec_free_context(&ctx);

    return img.clone(); // return a deep copy to avoid using freed memory
}

// Face detection
cv::CascadeClassifier &get_face_cascade()
{
    static cv::CascadeClassifier face_cascade;
    static bool loaded = []()
    {
        auto xml_path = get_module_dir() / "lib" / "opencv" / "haarcascade_frontalface_default.xml";
        if (!face_cascade.load(xml_path.string()))
        {
            std::ostringstream oss;
            oss << "Không thể load file: " << xml_path.string();
            throw std::runtime_error(oss.str());
        }
        return true;
    }();
    return face_cascade;
}

// Process ảnh RTP thành tensor
py::array_t<float> process_rtp_image(py::bytes rtp_payload_py, const std::string &codec_type)
{
    std::string rtp_payload_str = rtp_payload_py;
    std::vector<uint8_t> rtp_payload(rtp_payload_str.begin(), rtp_payload_str.end());
    cv::Mat img;
    if (codec_type == "H264")
    {
        img = decode_h264(rtp_payload);
    }
    else if (codec_type == "VP8")
    {
        img = decode_vp8(rtp_payload);
    }
    else
    {
        throw std::runtime_error("Unsupported codec: " + codec_type);
    }
    std::vector<cv::Rect> faces;
    get_face_cascade().detectMultiScale(img, faces, 1.1, 3, 0, cv::Size(30, 30));
    if (faces.empty())
    {
        throw std::runtime_error("No face detected");
    }
    cv::Rect face = faces[0] & cv::Rect(0, 0, img.cols, img.rows);
    if (face.area() <= 0)
    {
        throw std::runtime_error("Invalid face region");
    }
    img = img(face);
    if (img.empty())
    {
        throw std::runtime_error("Image decode failed");
    }
    // ✅ Chuyển sang grayscale
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::resize(img, img, cv::Size(IMG_SIZE, IMG_SIZE));
    img.convertTo(img, CV_32FC1, 1.0 / 255.0);
    img = (img - 0.5f) / 0.5f;
    // ✅ CHW với 1 channel (C = 1)
    std::vector<float> chw_data(IMG_SIZE * IMG_SIZE);
    for (int h = 0; h < IMG_SIZE; ++h)
        for (int w = 0; w < IMG_SIZE; ++w)
            chw_data[h * IMG_SIZE + w] = img.at<float>(h, w);

    // ✅ shape: (1, 1, 48, 48)
    return py::array(py::buffer_info(
        chw_data.data(),
        sizeof(float),
        py::format_descriptor<float>::format(),
        4,
        {1, 1, IMG_SIZE, IMG_SIZE},
        {IMG_SIZE * IMG_SIZE * sizeof(float),
         IMG_SIZE * sizeof(float),
         IMG_SIZE * sizeof(float),
         sizeof(float)}));
}

std::string infer_emotion(py::array_t<float> input_tensor)
{
    auto info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    const std::array<int64_t, 4> input_shape{1, 1, IMG_SIZE, IMG_SIZE};
    auto input_buf = input_tensor.request();
    float *input_data = static_cast<float *>(input_buf.ptr);

    Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(info, input_data, 1 * IMG_SIZE * IMG_SIZE, input_shape.data(), input_shape.size());

    const char *input_names[] = {"input"};
    const char *output_names[] = {"output"};
    auto output_tensors = get_session().Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_ort, 1, output_names, 1);

    float *output_data = output_tensors[0].GetTensorMutableData<float>();

    int max_index = std::distance(output_data, std::max_element(output_data, output_data + 7));
    static const std::vector<std::string> emotions = {"Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"};
    return emotions[max_index];
}

std::string predict_emotion_from_rtp(py::bytes rtp_payload_py, const std::string &codectype)
{
    py::array_t<float> tensor = process_rtp_image(rtp_payload_py, codectype);
    return infer_emotion(tensor);
}

std::string infer_emotion_from_image(py::array_t<uint8_t> image_array)
{
    auto buf = image_array.request();
    if (buf.ndim != 3 || buf.shape[2] != 3)
        throw std::runtime_error("Ảnh đầu vào phải có shape (H, W, 3)");

    int height = static_cast<int>(buf.shape[0]);
    int width = static_cast<int>(buf.shape[1]);

    cv::Mat img(height, width, CV_8UC3, (unsigned char *)buf.ptr);
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // CHUYỂN GRAYSCALE
    cv::resize(img, img, cv::Size(IMG_SIZE, IMG_SIZE));
    img.convertTo(img, CV_32FC1, 1.0 / 255.0); // CHUYỂN float 1 channel

    std::vector<float> chw_data(IMG_SIZE * IMG_SIZE); // CHỈ 1 CHANNEL
    for (int h = 0; h < IMG_SIZE; ++h)
        for (int w = 0; w < IMG_SIZE; ++w)
            chw_data[h * IMG_SIZE + w] = img.at<float>(h, w);

    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    const std::array<int64_t, 4> input_shape{1, 1, IMG_SIZE, IMG_SIZE}; // 1 CHANNEL
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, chw_data.data(), chw_data.size(), input_shape.data(), input_shape.size());

    const char *input_names[] = {"input"};
    const char *output_names[] = {"output"};
    auto output_tensors = get_session().Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    float *output_data = output_tensors[0].GetTensorMutableData<float>();
    int max_index = static_cast<int>(std::max_element(output_data, output_data + 7) - output_data);
    std::cout << "✅ Đã chạy xong ONNX inference!" << std::endl;

    static const std::vector<std::string> emotions = {"Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"};
    return emotions[max_index];
}

PYBIND11_MODULE(rtpimageproc, m)
{
    m.def("process_rtp_image", &process_rtp_image, "Decode and process RTP H264 payload to tensor");
    m.def("infer_emotion", &infer_emotion, "Run emotion recognition from tensor");
    m.def("infer_emotion_from_image", &infer_emotion_from_image, "Run emotion recognition from raw image (numpy array)");
    m.def("predict_emotion_from_rtp", &predict_emotion_from_rtp, "Decode RTP payload and return predicted emotion");
    m.def("cleanup_onnx", &cleanup_onnx, "Free ONNX session and env");
}
