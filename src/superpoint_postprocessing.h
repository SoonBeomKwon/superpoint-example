#ifndef SUPERPOINT_POSTPROCESSING_H
#define SUPERPOINT_POSTPROCESSING_H

#include <vector>
#include <string>
#include <cstdint> // for uint8_t

// Google C++ 스타일: 구조체 이름은 CamelCase
struct InterestPoint {
    int x = 0;
    int y = 0;
    float score = 0.0f;
};

// Google C++ 스타일: 함수 이름은 snake_case
class SuperPointPostProcessor {
public:
    // 생성자
    SuperPointPostProcessor(int original_image_width, int original_image_height,
                            int detector_output_height, int detector_output_width,
                            int descriptor_output_height, int descriptor_output_width,
                            int descriptor_dim);

    // 관심점 감지 결과(점수 맵)를 후처리하여 관심점 목록을 반환합니다.
    // ONNX 모델에서 나온 raw score map 데이터를 입력으로 받습니다.
    // score_map_data는 [1, 65, detector_output_height, detector_output_width] 형태를 기대합니다.
    std::vector<InterestPoint> process_keypoint_detections(
        const std::vector<float>& score_map_data,
        float score_threshold = 0.015f, // 논문에서 사용된 임계값 (0.015f)
        int nms_radius = 4 // 비최대 억제(NMS) 반경
    ) const;

    // 디스크립터 맵에서 특정 관심점 위치에 대한 디스크립터 벡터를 추출합니다.
    // descriptor_map_data는 [1, descriptor_dim, descriptor_output_height, descriptor_output_width] 형태를 기대합니다.
    // (논문에서는 [Hc x Wc x D] 형태로 언급되었으나, ONNX에서는 보통 [N, C, H, W]로 표현됩니다.)
    std::vector<float> extract_descriptor(
        const std::vector<float>& descriptor_map_data,
        int point_x, int point_y) const;

private:
    // 내부적으로 비최대 억제(NMS)를 수행하는 헬퍼 함수
    // Google C++ 스타일: private 함수 이름도 snake_case
    std::vector<InterestPoint> apply_non_max_suppression(
        const std::vector<float>& scores,
        int height, int width,
        float threshold, int radius
    ) const;

    // 2D 좌표를 원본 이미지 좌표계로 변환하는 헬퍼 함수
    InterestPoint map_to_original_coordinates(int mapped_x, int mapped_y, float score) const;

    int original_image_width_;
    int original_image_height_;
    int detector_output_height_;
    int detector_output_width_;
    int descriptor_output_height_;
    int descriptor_output_width_;
    int descriptor_dim_;
};

#endif // SUPERPOINT_POSTPROCESSING_H
