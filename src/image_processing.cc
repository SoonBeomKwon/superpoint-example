#include "image_processing.h"

#include <stdexcept> // for std::runtime_error
#include <iostream>  // for std::cout, std::cerr

// OpenCV 라이브러리 (이미지 로딩 및 조작을 위해)
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp> // for imread

// Google C++ 스타일: 네임스페이스 사용
namespace {
    // 내부용 상수 (예: 정규화 값)
    const float MEAN_R = 0.485f;
    const float STD_R = 0.229f;
    // ... 다른 채널에 대한 값 ...
} // namespace

ImageProcessor::ImageProcessor(int input_width, int input_height, int input_channels)
    : target_input_width_(input_width),
      target_input_height_(input_height),
      target_input_channels_(input_channels) {
    // 생성자에서 유효성 검사
    if (input_width <= 0 || input_height <= 0 || input_channels <= 0) {
        throw std::invalid_argument("Input dimensions must be positive.");
    }
    std::cout << "ImageProcessor initialized for "
              << target_input_width_ << "x" << target_input_height_
              << " with " << target_input_channels_ << " channels." << std::endl;
}

int ImageProcessor::get_original_height() const noexcept {
    return original_image_height_;
}

int ImageProcessor::get_original_width() const noexcept {
    return original_image_width_;
}

std::vector<float> ImageProcessor::preprocess_image(const std::string& image_path) const noexcept {
    // OpenCV로 이미지 로드
    // Google C++ 스타일: cv::imread는 cv::Mat을 반환합니다.
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE); // SuperPoint는 grayscale을 사용하는 것으로 보입니다.
    if (img.empty()) {
        std::cerr << "Error: Failed to load image from " << image_path << std::endl;
        return {}; // 빈 벡터 반환
    }

    original_image_height_ = img.rows;
    original_image_width_ = img.cols;

    // 이미지 크기 조정
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(target_input_width_, target_input_height_), 0, 0, cv::INTER_LINEAR);

    // float 타입으로 변환 및 정규화
    std::vector<float> input_tensor_values;
    input_tensor_values.reserve(static_cast<size_t>(target_input_channels_) * target_input_height_ * target_input_width_);

    // ONNX Runtime은 보통 [Batch Size, Channels, Height, Width] 형식을 기대합니다.
    // 따라서 데이터를 채널 우선(channel-first)으로 쌓습니다.
    // Grayscale (1 채널)의 경우: [1, 1, H, W]
    // RGB (3 채널)의 경우: [1, 3, H, W]
    for (int c = 0; c < target_input_channels_; ++c) {
        for (int h = 0; h < target_input_height_; ++h) {
            for (int w = 0; w < target_input_width_; ++w) {
                // 픽셀 값 가져오기 (0-255)
                float pixel_value = static_cast<float>(resized_img.at<uint8_t>(h, w));

                // 정규화 (예시: ImageNet과 유사한 평균/표준편차 사용)
                // SuperPoint 논문에서는 명시적인 정규화 방법을 언급하지 않았으므로,
                // 일반적인 딥러닝 모델에서 사용하는 방식을 따릅니다.
                // 만약 0-1 범위로 하려면 (pixel_value / 255.0f)를 사용합니다.
                // 여기서는 0-255 범위를 사용하고, 모델이 학습 시 이 범위를 사용했다고 가정합니다.
                // 필요하다면, 모델의 실제 학습 방식을 확인하여 정규화 방식을 수정해야 합니다.
                // pixel_value = (pixel_value - MEAN_R) / STD_R; // 예: 평균 빼고 표준편차 나누기

                input_tensor_values.push_back(pixel_value);
            }
        }
    }

    std::cout << "Image '" << image_path << "' preprocessed. Original size: "
              << original_image_width_ << "x" << original_image_height_
              << ", Target input size: " << target_input_width_ << "x" << target_input_height_ << std::endl;

    return input_tensor_values;
}
