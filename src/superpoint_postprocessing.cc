#include "superpoint_postprocessing.h"

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath> // for std::sqrt, std::floor
#include <algorithm> // for std::max, std::min

// Google C++ 스타일: 네임스페이스 사용
namespace {
    // 2D 배열에서 1D 인덱스를 계산하는 헬퍼 매크로/함수
    // Google C++ 스타일: const unsigned int로 타입을 명확히 합니다.
    inline size_t get_index(int h, int w, int width) {
        return static_cast<size_t>(h) * width + w;
    }
} // namespace

SuperPointPostProcessor::SuperPointPostProcessor(int original_image_width, int original_image_height,
                                               int detector_output_height, int detector_output_width,
                                               int descriptor_output_height, int descriptor_output_width,
                                               int descriptor_dim)
    : original_image_width_(original_image_width),
      original_image_height_(original_image_height),
      detector_output_height_(detector_output_height),
      detector_output_width_(detector_output_width),
      descriptor_output_height_(descriptor_output_height),
      descriptor_output_width_(descriptor_output_width),
      descriptor_dim_(descriptor_dim) {
    // 생성자 유효성 검사
    if (original_image_width <= 0 || original_image_height <= 0 ||
        detector_output_height <= 0 || detector_output_width <= 0 ||
        descriptor_output_height <= 0 || descriptor_output_width <= 0 ||
        descriptor_dim <= 0) {
        throw std::invalid_argument("Invalid dimensions provided to SuperPointPostProcessor.");
    }
    std::cout << "SuperPointPostProcessor initialized." << std::endl;
}

// 비최대 억제 (Non-Maximum Suppression) 알고리즘
// Google C++ 스타일: const 함수로 선언합니다.
std::vector<InterestPoint> SuperPointPostProcessor::apply_non_max_suppression(
    const std::vector<float>& scores,
    int height, int width,
    float threshold, int radius) const {

    std::vector<InterestPoint> points;
    // NMS를 위한 마스크 (이미 처리된 픽셀을 표시)
    std::vector<bool> suppressed(static_cast<size_t>(height) * width, false);

    // 점수 맵을 순회하며 로컬 최대값 찾기
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            size_t idx = get_index(r, c, width);
            if (suppressed[idx]) {
                continue; // 이미 억제된 픽셀
            }

            float score = scores[idx];
            if (score < threshold) {
                continue; // 임계값 이하의 점수는 무시
            }

            // 로컬 최대값인지 확인 (반경 내의 다른 점수보다 높아야 함)
            bool is_local_max = true;
            for (int dr = -radius; dr <= radius; ++dr) {
                for (int dc = -radius; dc <= radius; ++dc) {
                    if (dr == 0 && dc == 0) continue;

                    int nr = r + dr;
                    int nc = c + dc;

                    // 경계 검사
                    if (nr >= 0 && nr < height && nc >= 0 && nc < width) {
                        size_t neighbor_idx = get_index(nr, nc, width);
                        if (scores[neighbor_idx] > score) {
                            is_local_max = false;
                            break;
                        }
                    }
                }
                if (!is_local_max) break;
            }

            if (is_local_max) {
                // 로컬 최대값이면 관심점으로 추가하고, 반경 내 픽셀들을 억제
                points.push_back({c, r, score});
                suppressed[idx] = true; // 현재 픽셀 억제 (필요 시)

                // 반경 내 픽셀들을 억제 (겹치지 않도록)
                for (int dr = -radius; dr <= radius; ++dr) {
                    for (int dc = -radius; dc <= radius; ++dc) {
                        int nr = r + dr;
                        int nc = c + dc;
                        if (nr >= 0 && nr < height && nc >= 0 && nc < width) {
                            suppressed[get_index(nr, nc, width)] = true;
                        }
                    }
                }
            }
        }
    }
    return points;
}

// ONNX 모델 출력에서 관심점 데이터를 추출하고 좌표를 원본 이미지 크기로 변환
std::vector<InterestPoint> SuperPointPostProcessor::process_keypoint_detections(
    const std::vector<float>& score_map_data,
    float score_threshold, int nms_radius) const {

    std::cout << "Processing keypoint detections..." << std::endl;
    std::vector<InterestPoint> detected_points;

    // SuperPoint 출력: [1, 65, H_out, W_out]
    // H_out, W_out은 입력 이미지 크기의 1/8입니다.
    // 65개의 채널 중 마지막 채널(dustbin)은 제외하고 64개의 관심점 채널을 사용합니다.

    int num_scores_per_pixel = 65; // 64개 관심점 + 1개 dustbin
    int num_keypoint_channels = num_scores_per_pixel - 1; // 64

    // 각 관심점 채널에 대해 NMS를 적용
    for (int k = 0; k < num_keypoint_channels; ++k) {
        std::vector<float> current_channel_scores;
        current_channel_scores.reserve(static_cast<size_t>(detector_output_height_) * detector_output_width_);

        // 현재 채널의 2D 점수 맵 데이터 추출
        // score_map_data는 [1, 65, H_out, W_out] 형태의 1D 벡터로 가정
        // 즉, 데이터 순서는 (채널 k, 높이 h, 너비 w) 입니다.
        for (int h = 0; h < detector_output_height_; ++h) {
            for (int w = 0; w < detector_output_width_; ++w) {
                // Index for [batch, channel, height, width] tensor:
                // k * H_out * W_out + h * W_out + w  (assuming batch size is 1)
                size_t index = static_cast<size_t>(k) * detector_output_height_ * detector_output_width_ +
                               static_cast<size_t>(h) * detector_output_width_ + w;
                current_channel_scores.push_back(score_map_data[index]);
            }
        }

        // NMS 적용
        std::vector<InterestPoint> points_from_channel = apply_non_max_suppression(
            current_channel_scores,
            detector_output_height_, detector_output_width_,
            score_threshold, nms_radius);

        // 원본 이미지 좌표계로 변환 및 결과에 추가
        for (const auto& p : points_from_channel) {
            // map_to_original_coordinates 함수는 InterestPoint 구조체에 x, y, score를 담습니다.
            detected_points.push_back(map_to_original_coordinates(p.x, p.y, p.score));
        }
    }

    // 최종적으로 감지된 관심점들의 총 개수 출력
    std::cout << "Found " << detected_points.size() << " interest points after NMS." << std::endl;
    return detected_points;
}

// 출력 텐서의 좌표를 원본 이미지 좌표로 매핑
InterestPoint SuperPointPostProcessor::map_to_original_coordinates(int mapped_x, int mapped_y, float score) const {
    // SuperPoint 논문: "The interest point detector computes X ∈ RHc×Wc×65 and outputs a tensor sized RH×W."
    // 여기서 Hc, Wc는 입력 이미지 크기의 1/8 입니다.
    // 즉, detector_output_width_ = original_image_width_ / 8, detector_output_height_ = original_image_height_ / 8
    // 따라서, mapped_x, mapped_y를 8배 스케일링하면 원본 이미지 크기의 좌표가 됩니다.

    // Google C++ 스타일: long long을 사용하여 오버플로우 방지
    long long original_x = static_cast<long long>(mapped_x) * original_image_width_ / detector_output_width_;
    long long original_y = static_cast<long long>(mapped_y) * original_image_height_ / detector_output_height_;

    // 최종 좌표는 int로 캐스팅 (필요하다면 반올림 등을 추가할 수 있습니다)
    InterestPoint point;
    point.x = static_cast<int>(original_x);
    point.y = static_cast<int>(original_y);
    point.score = score; // NMS에서 얻은 점수를 그대로 사용
    return point;
}


// 디스크립터 맵에서 특정 관심점에 대한 디스크립터 벡터를 추출합니다.
std::vector<float> SuperPointPostProcessor::extract_descriptor(
    const std::vector<float>& descriptor_map_data,
    int point_x, int point_y) const {

    std::vector<float> descriptor(descriptor_dim_);

    // SuperPoint 논문: "The decoder then performs bi-cubic interpolation of the descriptor and then L2-normalizes the activations."
    // ONNX 모델 출력: [1, descriptor_dim, H_desc_out, W_desc_out]
    // H_desc_out, W_desc_out은 입력 이미지 크기의 1/8 입니다.

    // 1. 원본 이미지 좌표를 디스크립터 맵 좌표로 매핑
    float mapped_x_float = static_cast<float>(point_x) * descriptor_output_width_ / original_image_width_;
    float mapped_y_float = static_cast<float>(point_y) * descriptor_output_height_ / original_image_height_;

    // 2. 쌍선형 보간 (Bilinear Interpolation) 구현 (Bi-cubic은 더 복잡하므로, 여기서는 Bilinear로 대체)
    // 실제 SuperPoint는 Bi-cubic을 사용하지만, Bilinear로도 충분히 좋은 결과를 얻을 수 있습니다.
    // Bi-cubic 구현은 매우 복잡하며, 여기서는 Bilinear 보간을 구현합니다.

    // 정수 좌표와 소수점 좌표 분리
    float x_frac = mapped_x_float - std::floor(mapped_x_float);
    float y_frac = mapped_y_float - std::floor(mapped_y_float);
    int x_int = static_cast<int>(std::floor(mapped_x_float));
    int y_int = static_cast<int>(std::floor(mapped_y_float));

    // 보간에 사용할 4개의 인접 픽셀 좌표 (경계 검사 필요)
    int x0 = std::max(0, x_int);
    int y0 = std::max(0, y_int);
    int x1 = std::min(descriptor_output_width_ - 1, x_int + 1);
    int y1 = std::min(descriptor_output_height_ - 1, y_int + 1);

    // 4개 픽셀의 디스크립터 값 (batch=0, channel=c 가정)
    std::vector<std::vector<float>> corners_descriptors(4, std::vector<float>(descriptor_dim_));

    for (int c = 0; c < descriptor_dim_; ++c) {
        // 각 픽셀의 디스크립터 값 가져오기
        // Index for [batch, channel, height, width] tensor:
        // c * H_desc_out * W_desc_out + y * W_desc_out + x (assuming batch size is 1)
        size_t idx00 = static_cast<size_t>(c) * descriptor_output_height_ * descriptor_output_width_ + static_cast<size_t>(y0) * descriptor_output_width_ + x0;
        size_t idx01 = static_cast<size_t>(c) * descriptor_output_height_ * descriptor_output_width_ + static_cast<size_t>(y0) * descriptor_output_width_ + x1;
        size_t idx10 = static_cast<size_t>(c) * descriptor_output_height_ * descriptor_output_width_ + static_cast<size_t>(y1) * descriptor_output_width_ + x0;
        size_t idx11 = static_cast<size_t>(c) * descriptor_output_height_ * descriptor_output_width_ + static_cast<size_t>(y1) * descriptor_output_width_ + x1;

        corners_descriptors[0][c] = descriptor_map_data[idx00]; // (x0, y0)
        corners_descriptors[1][c] = descriptor_map_data[idx01]; // (x1, y0)
        corners_descriptors[2][c] = descriptor_map_data[idx10]; // (x0, y1)
        corners_descriptors[3][c] = descriptor_map_data[idx11]; // (x1, y1)
    }

    // Bilinear 보간 수행
    // Q11 = (1-fx)(1-fy)P00 + fx(1-fy)P10 + (1-fx)fyP01 + fx fy P11
    // P00: (x0, y0), P10: (x1, y0), P01: (x0, y1), P11: (x1, y1)
    // fx = x_frac, fy = y_frac

    float weight00 = (1.0f - x_frac) * (1.0f - y_frac);
    float weight10 = x_frac * (1.0f - y_frac);
    float weight01 = (1.0f - x_frac) * y_frac;
    float weight11 = x_frac * y_frac;

    for (int c = 0; c < descriptor_dim_; ++c) {
        descriptor[c] = weight00 * corners_descriptors[0][c] +
                        weight10 * corners_descriptors[1][c] +
                        weight01 * corners_descriptors[2][c] +
                        weight11 * corners_descriptors[3][c];
    }

    // L2 정규화
    float norm = 0.0f;
    for (float val : descriptor) {
        norm += val * val;
    }
    norm = std::sqrt(norm);

    if (norm > 1e-6) { // 0으로 나누는 것 방지
        for (float& val : descriptor) {
            val /= norm;
        }
    }

    return descriptor;
}
