#include <iostream>
#include <vector>
#include <string>
#include <numeric> // for std::iota
#include <memory>  // for std::unique_ptr
#include <algorithm> // for std::min

// ONNX Runtime 헤더
#include <onnxruntime_cxx_api.h>

// 사용자 정의 헤더
#include "image_processing.h"
#include "superpoint_postprocessing.h"
#include "visualization.h"

// OpenCV 헤더 (메인 함수에서 이미지 로딩/처리를 위해)
#include <opencv2/opencv.hpp>

// Google C++ 스타일: 상수 정의
constexpr int DEFAULT_MODEL_INPUT_WIDTH = 640;
constexpr int DEFAULT_MODEL_INPUT_HEIGHT = 480;
constexpr int DEFAULT_MODEL_CHANNELS = 1; // SuperPoint는 Grayscale 입력으로 가정
constexpr float DEFAULT_SCORE_THRESHOLD = 0.015f; // 논문에서 사용된 임계값
constexpr int DEFAULT_NMS_RADIUS = 4; // 논문에서 사용된 NMS 반경

// ----------------------------------------------------------------------------
// Helper 함수 (internal)
// ----------------------------------------------------------------------------

// ONNX 텐서의 차원을 std::vector<int64_t>로 변환하는 함수
// Google C++ 스타일: const 참조 사용, noexcept
std::vector<int64_t> get_tensor_shape(const OrtApiTypeInfo* api_type_info) {
    const OrtTensorTypeAndShapeInfo* tensor_info = api_type_info->GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape;
    tensor_info->GetShape(&shape);
    return shape;
}

// ONNX 텐서의 데이터를 float 벡터로 추출하는 함수
// Google C++ 스타일: const 참조 사용, noexcept
std::vector<float> get_tensor_data(const OrtValue& tensor) {
    std::vector<float> data;
    const OrtApiTypeInfo* api_type_info = tensor.GetTypeInfo();
    const OrtTensorTypeAndShapeInfo* tensor_info = api_type_info->GetTensorTypeAndShapeInfo();
    size_t total_elements = tensor_info->GetTensorElementCount();
    data.resize(total_elements);
    tensor.GetTensorMutableData<float>(data.data());
    return data;
}

// ----------------------------------------------------------------------------
// 메인 함수
// ----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    // Google C++ 스타일: 사용법 안내
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <onnx_model_path> <image_path> <output_visualization_path>" << std::endl;
        std::cerr << "Example: " << argv[0] << " models/superpoint.onnx images/test_image.jpg output/result.png" << std::endl;
        return -1;
    }

    const std::string onnx_model_path = argv[1];
    const std::string image_path = argv[2];
    const std::string output_visualization_path = argv[3];

    try {
        // 1. ONNX Runtime 환경 및 세션 초기화
        // Google C++ 스타일: RAII(Resource Acquisition Is Initialization)를 위해 unique_ptr 사용 (또는 Ort::Session과 같은 클래스 활용)
        // OrtApi와 OrtEnv는 필요에 따라 수동으로 관리하거나, Ort::Environment, Ort::Session 클래스를 사용하는 것이 더 안전합니다.
        // 여기서는 C API를 직접 사용하므로, OrtApi, OrtEnv, OrtSession, OrtAllocator 등을 명시적으로 관리해야 합니다.

        OrtApi api; // ORT API 객체
        OrtEnv env(ORT_LOGGING_LEVEL_WARNING, "SuperPointInference"); // 환경 객체
        OrtSessionOptions session_options; // 세션 옵션

        // 세션 옵션 설정 (필요에 따라)
        // session_options.SetIntraOpNumThreads(1); // 싱글 스레드 추론 (CPU 병목 확인용)
        // session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL); // 그래프 최적화

        // ONNX 모델 로딩
        // Ort::Session session(env, onnx_model_path.c_str(), session_options); // C++ API 사용 시
        OrtSession* session_ptr = nullptr; // C API 사용 시
        api.CreateSession(env, onnx_model_path.c_str(), &session_options, &session_ptr);
        std::unique_ptr<OrtSession, decltype(&OrtApi::ReleaseSession)> session(session_ptr, api.ReleaseSession);

        // 입력/출력 노드 이름 및 차원 정보 가져오기
        char** input_node_names_ptr = nullptr;
        size_t num_input_nodes;
        session->GetInputCount(&num_input_nodes);
        session->GetInputName(0, env.allocator(), &input_node_names_ptr); // 첫 번째 입력 노드 이름
        std::unique_ptr<char*, decltype(&OrtApi::Free)> input_node_names(input_node_names_ptr, env.allocator().Free);
        std::string input_name = *input_node_names;

        char** output_node_names_ptr = nullptr;
        size_t num_output_nodes;
        session->GetOutputCount(&num_output_nodes);
        session->GetOutputName(0, env.allocator(), &output_node_names_ptr); // 첫 번째 출력 노드 이름 (관심점 감지)
        std::unique_ptr<char*, decltype(&OrtApi::Free)> output_node_names(output_node_names_ptr, env.allocator().Free);

        // 출력 노드 이름 확인 (SuperPoint는 두 개의 출력을 가짐: 관심점, 디스크립터)
        if (num_output_nodes < 2) {
            throw std::runtime_error("Model must have at least two outputs (keypoints and descriptors).");
        }
        // 두 번째 출력 노드 이름을 가져옵니다.
        char* descriptor_output_name_c = nullptr;
        session->GetOutputName(1, env.allocator(), &descriptor_output_name_c);
        std::unique_ptr<char*, decltype(&OrtApi::Free)> descriptor_output_name_ptr(descriptor_output_name_c, env.allocator().Free);
        std::string descriptor_output_name = *descriptor_output_name_ptr;

        // 입력 텐서 정보 가져오기
        OrtApiTypeInfo* input_type_info = session->GetInputTypeInfo(0);
        std::vector<int64_t> input_shape = get_tensor_shape(input_type_info);
        // C++ API 사용 시: const OrtTensorTypeAndShapeInfo* input_tensor_info = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        // std::vector<int64_t> input_shape = input_tensor_info->GetShape();

        if (input_shape.size() < 4) {
            throw std::runtime_error("Expected input tensor with at least 4 dimensions (N, C, H, W).");
        }
        // 입력 크기 추출
        int model_batch_size = input_shape[0]; // 보통 1
        int model_channels = input_shape[1];   // SuperPoint는 Grayscale (1) 사용 가정
        int model_input_height = input_shape[2];
        int model_input_width = input_shape[3];

        std::cout << "Model input expected: " << model_batch_size << "x" << model_channels << "x"
                  << model_input_height << "x" << model_input_width << std::endl;

        // 2. 이미지 전처리
        ImageProcessor image_processor(model_input_width, model_input_height, model_channels);
        std::vector<float> input_tensor_values = image_processor.preprocess_image(image_path);
        if (input_tensor_values.empty()) {
            std::cerr << "Error: Image preprocessing failed." << std::endl;
            return -1;
        }

        // 입력 텐서 생성
        std::vector<int64_t> input_shape_actual = {model_batch_size, model_channels, model_input_height, model_input_width};
        OrtMemoryInfo memory_info = OrtMemoryInfo::CreateCpu(0, OrtArenaAllocator::ALCommit);
        auto input_tensor = OrtValue::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape_actual.data(), input_shape_actual.size());

        // 3. 추론 실행
        std::vector<OrtValue> output_tensors;
        std::vector<const char*> output_names = {input_name.c_str(), descriptor_output_name.c_str()}; // 입력 이름과 출력 이름 목록
        std::vector<const char*> input_names = {input_name.c_str()}; // 입력 이름 목록

        session->Run(session_options, input_names.data(), (const OrtValue*) &input_tensor, 1, output_names.data(), output_names.size(), output_tensors.data());
        // C++ API: session.Run({input_name.c_str()}, {input_tensor}, {keypoint_output_name.c_str(), descriptor_output_name.c_str()});

        // 4. 출력 텐서 처리
        if (output_tensors.size() < 2) {
            throw std::runtime_error("Expected at least two outputs from the model.");
        }

        OrtValue& keypoint_output_tensor = output_tensors[0];
        OrtValue& descriptor_output_tensor = output_tensors[1];

        // 출력 텐서 데이터 추출
        std::vector<float> keypoint_scores = get_tensor_data(keypoint_output_tensor);
        std::vector<float> descriptor_maps = get_tensor_data(descriptor_output_tensor);

        // 출력 텐서의 shape 정보 가져오기
        OrtApiTypeInfo* kp_type_info = keypoint_output_tensor.GetTypeInfo();
        std::vector<int64_t> keypoint_output_shape = get_tensor_shape(kp_type_info);
        OrtApiTypeInfo* desc_type_info = descriptor_output_tensor.GetTypeInfo();
        std::vector<int64_t> descriptor_output_shape = get_tensor_shape(desc_type_info);

        // 출력 텐서 차원 확인
        if (keypoint_output_shape.size() < 4 || descriptor_output_shape.size() < 4) {
            throw std::runtime_error("Expected output tensors with at least 4 dimensions (N, C, H, W).");
        }

        int kp_channels = keypoint_output_shape[1];
        int kp_output_height = keypoint_output_shape[2];
        int kp_output_width = keypoint_output_shape[3];

        int desc_channels = descriptor_output_shape[1]; // Descriptor dimension
        int desc_output_height = descriptor_output_shape[2];
        int desc_output_width = descriptor_output_shape[3];

        // 후처리를 위한 객체 생성
        SuperPointPostProcessor post_processor(
            image_processor.get_original_width(), image_processor.get_original_height(),
            kp_output_height, kp_output_width,
            desc_output_height, desc_output_width,
            desc_channels // Descriptor dimension
        );

        // 관심점 감지 결과 후처리
        std::vector<InterestPoint> detected_points = post_processor.process_keypoint_detections(
            keypoint_scores, DEFAULT_SCORE_THRESHOLD, DEFAULT_NMS_RADIUS);

        // 디스크립터 추출
        std::vector<std::vector<float>> all_descriptors;
        all_descriptors.reserve(detected_points.size());
        for (const auto& point : detected_points) {
            all_descriptors.push_back(post_processor.extract_descriptor(descriptor_maps, point.x, point.y));
        }

        // 5. 결과 시각화
        // OpenCV Mat 객체로 원본 이미지 로드 (시각화용)
        cv::Mat visualization_image = cv::imread(image_path, cv::IMREAD_COLOR); // 컬러로 로드
        if (visualization_image.empty()) {
            std::cerr << "Error: Failed to load image for visualization from " << image_path << std::endl;
            return -1;
        }

        visualize_superpoint_results(output_visualization_path, visualization_image, detected_points, all_descriptors);

        // 6. 결과 출력 (간략하게)
        std::cout << "\n--- SuperPoint Inference Complete ---" << std::endl;
        std::cout << "Detected " << detected_points.size() << " interest points." << std::endl;
        std::cout << "Visualization saved to: " << output_visualization_path << std::endl;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Standard Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
