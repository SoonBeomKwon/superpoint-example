# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(SuperPointInference)
set(PROJECT_VERSION "1.0.0")

# Set C++ standard to C++17 (recommended for modern C++ features)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Disable C++ extensions for better portability

# --- Find ONNX Runtime ---
# ONNX Runtime은 CMake를 통해 찾거나, 수동으로 경로를 지정해야 합니다.
# 아래는 CMake를 통해 찾으려는 시도입니다. 만약 실패한다면,
# ONNX Runtime의 include 디렉토리와 라이브러리 디렉토리를 직접 지정해야 합니다.
# 예: set(ONNXRUNTIME_ROOT "path/to/onnxruntime")
# find_package(onnxruntime REQUIRED) # ONNX Runtime이 CMake config를 제공하는 경우

# ONNX Runtime 경로를 수동으로 지정하는 경우 (더 일반적)
# ONNX Runtime 설치 경로에 따라 이 부분을 수정해야 합니다.
# 예: Linux/macOS: /usr/local/lib/onnxruntime, Windows: C:/onnxruntime/lib
set(ONNXRUNTIME_ROOT_DIR "" CACHE PATH "Root directory of ONNX Runtime installation")
if(NOT ONNXRUNTIME_ROOT_DIR)
    message(WARNING "ONNXRUNTIME_ROOT_DIR is not set. Please set it to the root of your ONNX Runtime installation.")
    message(WARNING "Example: For Linux/macOS, it might be /usr/local/onnxruntime")
    message(WARNING "         For Windows, it might be C:/onnxruntime")
    # 기본값으로 몇 가지 일반적인 경로를 시도해 볼 수 있습니다.
    # find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h HINTS ${ONNXRUNTIME_ROOT_DIR}/include)
    # find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime HINTS ${ONNXRUNTIME_ROOT_DIR}/lib)
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT_DIR}/include
    PATH_SUFFIXES include
    DOC "ONNX Runtime include directory"
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS ::MATH_PLACEHOLDER_0::{ONNXRUNTIME_ROOT_DIR}
    PATH_SUFFIXES lib lib64
    DOC "ONNX Runtime library"
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found. Please set ONNXRUNTIME_ROOT_DIR to your ONNX Runtime installation path.")
endif()

message(STATUS "Found ONNX Runtime include directory: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "Found ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

# --- Find OpenCV ---
# OpenCV는 3D 포인트 시각화에 사용될 수 있습니다.
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# --- Add executable ---
add_executable(superpoint_inference
    src/onnx_inference.cpp
    src/image_processing.cpp # 이미지 로딩 및 전처리를 위한 파일
    src/superpoint_postprocessing.cpp # 관심점 및 디스크립터 후처리를 위한 파일
    src/visualization.cpp # 시각화 기능을 위한 파일
)

# --- Link libraries ---
target_include_directories(superpoint_inference PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(superpoint_inference PRIVATE
    ${ONNXRUNTIME_LIBRARY}
    ${OpenCV_LIBS}
)

# Set output directory for executables and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# --- Installation rules (optional) ---
# install(TARGETS superpoint_inference DESTINATION bin)

# --- Testing (optional) ---
# enable_testing()
# add_test(NAME SuperPointTest COMMAND $<TARGET_FILE:superpoint_inference> <path_to_test_model> <path_to_test_image>)
