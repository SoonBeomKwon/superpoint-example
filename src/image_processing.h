#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <string>
#include <vector>
#include <cstdint> // for uint8_t

// Google C++ 스타일: 클래스 이름은 CamelCase
class ImageProcessor {
public:
    // 생성자: 모델 입력 크기를 설정합니다.
    // Google C++ 스타일: 생성자 매개변수 이름은 snake_case
    explicit ImageProcessor(int input_width, int input_height, int input_channels = 1);

    // 이미지를 로드하고 모델 입력 텐서에 맞게 전처리합니다.
    // 반환되는 벡터는 [channels, height, width] 또는 [height, width, channels] 형식으로,
    // 모델의 요구사항에 따라 달라집니다. ONNX Runtime은 보통 [N, C, H, W]를 기대하므로,
    // 이 함수는 해당 형식으로 변환된 float 벡터를 반환해야 합니다.
    // Google C++ 스타일: 함수 이름은 snake_case, 매개변수는 snake_case
    // const 참조를 사용하여 불필요한 복사를 방지합니다.
    // noexcept는 함수가 예외를 던지지 않음을 명시합니다 (가능한 경우).
    std::vector<float> preprocess_image(const std::string& image_path) const noexcept;

    // 전처리된 이미지의 원본 크기를 반환합니다.
    int get_original_height() const noexcept;
    int get_original_width() const noexcept;

private:
    int target_input_width_;
    int target_input_height_;
    int target_input_channels_;
    int original_image_height_ = 0;
    int original_image_width_ = 0;
};

#endif // IMAGE_PROCESSING_H
