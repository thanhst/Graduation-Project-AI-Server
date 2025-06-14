#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <filesystem>
#include <Windows.h>
#include <mutex>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>

namespace py = pybind11;

constexpr int IMG_SIZE = 48;

std::filesystem::path get_module_dir() {
    HMODULE hMod = nullptr;
    GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                      GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                      (LPCSTR)&get_module_dir, &hMod);
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(hMod, path, MAX_PATH);
    return std::filesystem::path(path).parent_path();
}

// ONNX Runtime setup
Ort::Env* env = nullptr;
Ort::Session* session = nullptr;
static std::mutex init_mutex;

void init_onnx() {
    std::lock_guard<std::mutex> lock(init_mutex);
    if (!env)
        env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "RTPInference");

    if (!session) {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        std::filesystem::path model_path = get_module_dir() / "lib" / "model" / "cnn_fer.onnx";
        std::wstring model_path_w = model_path.wstring();
        session = new Ort::Session(*env, model_path_w.c_str(), options);
    }
}

Ort::Session& get_session() {
    if (!session) init_onnx();
    return *session;
}

void cleanup_onnx() {
    delete session;
    delete env;
    session = nullptr;
    env = nullptr;
}

// Simulated decode
cv::Mat decode_h264_rtp_simulated(const std::vector<uint8_t>& rtp_payload) {
    cv::Mat img(240, 320, CV_8UC3);
    cv::randu(img, cv::Scalar::all(0), cv::Scalar::all(255));
    return img;
}

// Face detection
cv::CascadeClassifier& get_face_cascade() {
    static cv::CascadeClassifier face_cascade;
    static bool loaded = []() {
        auto xml_path = get_module_dir() / "lib" / "opencv" / "haarcascade_frontalface_default.xml";
        if (!face_cascade.load(xml_path.string())) {
            std::ostringstream oss;
            oss << "Không thể load file: " << xml_path.string();
            throw std::runtime_error(oss.str());
        }
        return true;
    }();
    return face_cascade;
}

// Process ảnh RTP thành tensor
py::array_t<float> process_rtp_image(py::bytes rtp_payload_py) {
    std::string rtp_payload_str = rtp_payload_py;
    std::vector<uint8_t> rtp_payload(rtp_payload_str.begin(), rtp_payload_str.end());
    cv::Mat img = decode_h264_rtp_simulated(rtp_payload);

    std::vector<cv::Rect> faces;
    get_face_cascade().detectMultiScale(img, faces, 1.1, 3, 0, cv::Size(30, 30));
    if (!faces.empty()) {
        cv::Rect face = faces[0] & cv::Rect(0, 0, img.cols, img.rows);
        if (face.area() > 0) img = img(face);
    }

    // ✅ Chuyển sang grayscale
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::resize(img, img, cv::Size(IMG_SIZE, IMG_SIZE));
    img.convertTo(img, CV_32FC1, 1.0 / 255.0);

    // ✅ CHW với 1 channel (C = 1)
    std::vector<float> chw_data(IMG_SIZE * IMG_SIZE);
    for (int h = 0; h < IMG_SIZE; ++h)
        for (int w = 0; w < IMG_SIZE; ++w)
            chw_data[h * IMG_SIZE + w] = img.at<float>(h, w);

    // ✅ shape: (1, 1, 48, 48)
    return py::array_t<float>({1, 1, IMG_SIZE, IMG_SIZE}, chw_data);
}


std::string infer_emotion(py::array_t<float> input_tensor) {
    auto info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    const std::array<int64_t, 4> input_shape{1, 1, IMG_SIZE, IMG_SIZE};
    auto input_buf = input_tensor.request();
    float* input_data = static_cast<float*>(input_buf.ptr);

    Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(info, input_data, 1 * IMG_SIZE * IMG_SIZE, input_shape.data(), input_shape.size());

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    auto output_tensors = get_session().Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_ort, 1, output_names, 1);

    float* output_data = output_tensors[0].GetTensorMutableData<float>();

    int max_index = std::distance(output_data, std::max_element(output_data, output_data + 7));
    static const std::vector<std::string> emotions = {"Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"};
    return emotions[max_index];
}

std::string predict_emotion_from_rtp(py::bytes rtp_payload_py) {
    py::array_t<float> tensor = process_rtp_image(rtp_payload_py);
    return infer_emotion(tensor);
}

std::string infer_emotion_from_image(py::array_t<uint8_t> image_array) {
    auto buf = image_array.request();
    if (buf.ndim != 3 || buf.shape[2] != 3)
        throw std::runtime_error("Ảnh đầu vào phải có shape (H, W, 3)");

    int height = static_cast<int>(buf.shape[0]);
    int width = static_cast<int>(buf.shape[1]);

    cv::Mat img(height, width, CV_8UC3, (unsigned char*)buf.ptr);
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);               // CHUYỂN GRAYSCALE
    cv::resize(img, img, cv::Size(IMG_SIZE, IMG_SIZE));
    img.convertTo(img, CV_32FC1, 1.0 / 255.0);                // CHUYỂN float 1 channel

    std::vector<float> chw_data(IMG_SIZE * IMG_SIZE);         // CHỈ 1 CHANNEL
    for (int h = 0; h < IMG_SIZE; ++h)
        for (int w = 0; w < IMG_SIZE; ++w)
            chw_data[h * IMG_SIZE + w] = img.at<float>(h, w);

    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    const std::array<int64_t, 4> input_shape{1, 1, IMG_SIZE, IMG_SIZE}; // 1 CHANNEL
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, chw_data.data(), chw_data.size(), input_shape.data(), input_shape.size());

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    auto output_tensors = get_session().Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    int max_index = static_cast<int>(std::max_element(output_data, output_data + 7) - output_data);
    std::cout << "✅ Đã chạy xong ONNX inference!" << std::endl;

    static const std::vector<std::string> emotions = {"Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"};
    return emotions[max_index];
}


PYBIND11_MODULE(rtpimageproc, m) {
    m.def("process_rtp_image", &process_rtp_image, "Decode and process RTP H264 payload to tensor");
    m.def("infer_emotion", &infer_emotion, "Run emotion recognition from tensor");
    m.def("infer_emotion_from_image", &infer_emotion_from_image, "Run emotion recognition from raw image (numpy array)");
    m.def("predict_emotion_from_rtp", &predict_emotion_from_rtp, "Decode RTP payload and return predicted emotion");
    m.def("cleanup_onnx", &cleanup_onnx, "Free ONNX session and env");
}
