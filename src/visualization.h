#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <string>
#include <vector>

#include <opencv2/core.hpp> // for cv::Mat

#include "superpoint_postprocessing.h" // for InterestPoint struct

// Google C++ 스타일: 함수 이름은 snake_case
// 이미지를 시각화하여 파일로 저장합니다.
void visualize_superpoint_results(
    const std::string& output_path,
    const cv::Mat& original_image, // 원본 컬러 또는 그레이스케일 이미지
    const std::vector<InterestPoint>& points,
    const std::vector<std::vector<float>>& descriptors, // 각 point에 대한 디스크립터 (필요시)
    int max_points_to_draw = 1000 // 그릴 최대 관심점 개수
);

#endif // VISUALIZATION_H
