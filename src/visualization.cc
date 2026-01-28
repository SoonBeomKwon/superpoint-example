#include "visualization.h"

#include <iostream>
#include <algorithm> // for std::min

#include <opencv2/imgproc.hpp> // for drawing functions
#include <opencv2/highgui.hpp> // for imwrite

// Google C++ 스타일: 네임스페이스 사용 (익명 네임스페이스)
namespace {
    // 관심점 마커 색상
    const cv::Scalar POINT_COLOR(0, 255, 0); // Green
    // 관심점 마커 반경
    const int POINT_RADIUS = 3;
    // 관심점 마커 두께
    const int POINT_THICKNESS = 2;

    // 디스크립터 텍스트 색상
    const cv::Scalar DESCRIPTOR_TEXT_COLOR(255, 0, 0); // Blue
    const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
    const double FONT_SCALE = 0.4;
    const int FONT_THICKNESS = 1;
} // namespace

void visualize_superpoint_results(
    const std::string& output_path,
    const cv::Mat& original_image,
    const std::vector<InterestPoint>& points,
    const std::vector<std::vector<float>>& descriptors, // 이 예시에서는 사용되지 않음
    int max_points_to_draw) {

    std::cout << "Visualizing results to: " << output_path << std::endl;

    // 원본 이미지를 컬러로 복사하여 그리기
    cv::Mat output_image;
    if (original_image.channels() == 1) {
        cv::cvtColor(original_image, output_image, cv::COLOR_GRAY2BGR);
    } else {
        output_image = original_image.clone();
    }

    // 그릴 관심점의 개수 제한
    int num_points_to_draw = std::min(static_cast<int>(points.size()), max_points_to_draw);

    // 관심점 그리기
    for (int i = 0; i < num_points_to_draw; ++i) {
        const auto& p = points[i];

        // 관심점 위치에 원 그리기
        cv::circle(output_image, cv::Point(p.x, p.y), POINT_RADIUS, POINT_COLOR, POINT_THICKNESS);

        // (옵션) 관심점 점수 표시 (가독성을 위해 모든 점수에 표시하지 않을 수 있음)
        // std::string score_text = std::to_string(static_cast<int>(p.score * 1000)); // 3자리 정수로 표시
        // cv::putText(output_image, score_text, cv::Point(p.x + POINT_RADIUS + 2, p.y - POINT_RADIUS - 2),
        //             FONT_FACE, FONT_SCALE, DESCRIPTOR_TEXT_COLOR, FONT_THICKNESS);

        // (옵션) 디스크립터 관련 정보 표시 (예: 인덱스)
        // if (!descriptors.empty() && descriptors[i].size() > 0) {
        //     std::string descriptor_index_text = std::to_string(i);
        //     cv::putText(output_image, descriptor_index_text, cv::Point(p.x + POINT_RADIUS + 2, p.y + POINT_RADIUS + 8),
        //                 FONT_FACE, FONT_SCALE, DESCRIPTOR_TEXT_COLOR, FONT_THICKNESS);
        // }
    }

    // 이미지 저장
    if (!cv::imwrite(output_path, output_image)) {
        std::cerr << "Error: Failed to save visualization image to " << output_path << std::endl;
    } else {
        std::cout << "Visualization saved successfully to " << output_path << std::endl;
    }
}
