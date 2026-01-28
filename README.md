# SuperPoint Inference C++

이 프로젝트는 Ubuntu 환경에서 SuperPoint 모델을 사용하여 이미지에서 관심점(interest points)을 감지하고 해당 디스크립터(descriptors)를 추출하는 C++ 추론 코드를 제공합니다. Google C++ 스타일 가이드와 CMake 빌드 시스템을 따릅니다.

## 1. Prerequisites (필수 구성 요소)

이 프로젝트를 빌드하고 실행하기 전에 시스템에 다음 소프트웨어가 설치되어 있어야 합니다.

### 1.1. C++ 컴파일러

최신 C++ 표준 (C++17 권장)을 지원하는 C++ 컴파일러가 필요합니다.

*   **GCC**: Ubuntu에서 가장 일반적으로 사용되는 컴파일러입니다.
    ```bash
    sudo apt update
    sudo apt install build-essential g++
    ```
    `g++ --version` 명령어로 설치 및 버전을 확인할 수 있습니다.

### 1.2. CMake

프로젝트 빌드 시스템을 생성하기 위한 도구입니다.

*   **CMake**:
    ```bash
    sudo apt install cmake
    ```
    `cmake --version` 명령어로 설치 및 버전을 확인할 수 있습니다.

### 1.3. OpenCV

이미지 로딩, 전처리, 시각화 기능을 위해 OpenCV 라이브러리가 필요합니다.

*   **OpenCV (Headless 빌드)**: GUI 없이 라이브러리만 필요한 경우, 일반적으로 더 빠르게 설치할 수 있습니다.
    ```bash
    sudo apt install libopencv-dev
    ```
    또는 소스에서 빌드하는 것을 권장합니다. 소스 빌드는 최신 버전을 사용하거나 특정 옵션을 설정하는 데 유용합니다. (소스 빌드 가이드는 OpenCV 공식 문서를 참고하세요.)

    **주의**: `libopencv-dev` 패키지는 시스템에 따라 OpenCV 버전이 다를 수 있습니다. 특정 버전을 사용해야 하는 경우 소스에서 빌드하는 것이 좋습니다.

### 1.4. ONNX Runtime

ONNX 형식의 SuperPoint 모델을 로드하고 추론을 수행하기 위한 라이브러리입니다.

*   **ONNX Runtime (C++ API)**:
    ONNX Runtime은 공식적으로 다양한 플랫폼용 라이브러리를 제공합니다. Ubuntu에서는 미리 빌드된 라이브러리를 다운로드하거나, 소스에서 직접 빌드할 수 있습니다.

    **옵션 1: 미리 빌드된 라이브러리 사용 (권장)**
    1.  **ONNX Runtime 다운로드**: ONNX Runtime GitHub 릴리스 페이지에서 Ubuntu용 `onnxruntime-linux-<arch>-<version>.tgz` 파일을 다운로드합니다.
        *   예: `onnxruntime-linux-x64-1.15.0.tgz` (버전 및 아키텍처는 다를 수 있습니다.)
    2.  **압축 해제**: 다운로드한 파일을 원하는 위치에 압축 해제합니다.
        ```bash
        tar -xzf onnxruntime-linux-x64-1.15.0.tgz -C /usr/local/ # 예시 경로
        ```
        `/usr/local/` 경로는 예시이며, 원하는 위치로 변경할 수 있습니다. 예를 들어 `$HOME/libs/onnxruntime`과 같이 사용자 홈 디렉토리 내에 설치할 수도 있습니다.
    3.  **CMake에서 경로 설정**: 빌드 시 `ONNXRUNTIME_ROOT_DIR` 변수에 압축 해제한 ONNX Runtime 디렉토리의 **루트 경로**를 지정해야 합니다.
        *   예: `/usr/local/onnxruntime-linux-x64-1.15.0`

    **옵션 2: 소스에서 빌드 (고급)**
    ONNX Runtime GitHub 저장소를 클론하여 소스 코드로부터 직접 빌드합니다. 이 방법은 더 많은 설정이 필요하며, 빌드 시간이 오래 걸릴 수 있습니다. ONNX Runtime 빌드 문서를 참고하세요.

---

## 2. 프로젝트 빌드 (Ubuntu)

프로젝트를 빌드하기 위해 CMake와 Make (또는 다른 빌드 도구)를 사용합니다.

### 2.1. 프로젝트 설정

1.  **프로젝트 디렉토리 구조**:
    ```
    your_project_root/
    ├── CMakeLists.txt
    ├── models/                 # .onnx 모델 파일
    │   └── superpoint.onnx     # 실제 SuperPoint 모델 파일 (직접 준비 필요)
    ├── images/                 # 테스트 이미지
    │   └── test_image.jpg      # 실제 테스트 이미지 (직접 준비 필요)
    ├── output/                 # 결과 시각화 이미지가 저장될 디렉토리 (자동 생성)
    ├── src/
    │   ├── onnx_inference.cpp
    │   ├── image_processing.h
    │   ├── image_processing.cpp
    │   ├── superpoint_postprocessing.h
    │   ├── superpoint_postprocessing.cpp
    │   ├── visualization.h
    │   ├── visualization.cpp
    └── build/                  # 빌드 파일이 생성될 디렉토리
    ```
2.  **모델 및 이미지 준비**:
    *   `models/` 디렉토리에 학습된 SuperPoint 모델의 `.onnx` 파일을 준비합니다. (예: `superpoint.onnx`)
    *   `images/` 디렉토리에 테스트할 이미지 파일을 준비합니다. (예: `test_image.jpg`)

### 2.2. 빌드 단계

1.  **빌드 디렉토리 생성**: 프로젝트 루트 디렉토리에서 `build` 디렉토리를 생성하고 해당 디렉토리로 이동합니다.
    ```bash
    mkdir build
    cd build
    ```

2.  **CMake 구성**: `cmake` 명령어를 사용하여 빌드 시스템을 구성합니다.
    *   **필수**: ONNX Runtime의 설치 루트 디렉토리를 `ONNXRUNTIME_ROOT_DIR` 변수로 지정해야 합니다.
    *   **선택**: OpenCV의 설치 경로를 `CMAKE_PREFIX_PATH`로 지정할 수 있습니다. (시스템 설치(`libopencv-dev`)의 경우 자동으로 찾아질 수도 있습니다.)

    **예시 (ONNX Runtime이 `/usr/local/onnxruntime-linux-x64-1.15.0`에 있고 OpenCV가 시스템에 설치된 경우):**
    ```bash
    cmake .. \
        -DONNXRUNTIME_ROOT_DIR=/usr/local/onnxruntime-linux-x64-1.15.0 \
        -DCMAKE_BUILD_TYPE=Release # 또는 Debug
    ```

    **팁**:
    *   `-DONNXRUNTIME_ROOT_DIR`에 **ONNX Runtime 라이브러리 및 헤더 파일이 포함된 최상위 디렉토리**를 지정하세요. (예: `/usr/local/onnxruntime-linux-x64-1.15.0/` 또는 `/home/youruser/libs/onnxruntime/`)
    *   `-DCMAKE_BUILD_TYPE=Release`는 최적화된 릴리스 빌드를, `Debug`는 디버깅 정보를 포함하는 빌드를 생성합니다. 릴리스 빌드가 성능이 더 좋습니다.
    *   만약 CMake가 ONNX Runtime이나 OpenCV를 찾지 못하면, 해당 라이브러리들의 include 디렉토리와 library 디렉토리를 직접 지정해야 할 수 있습니다 (CMakeLists.txt 파일 수정 필요).

3.  **빌드 실행**: CMake 구성이 성공하면, `make` 명령어를 사용하여 프로젝트를 빌드합니다.
    ```bash
    make
    ```
    또는
    ```bash
    cmake --build . --config Release # Release 모드로 빌드
    ```

4.  **빌드 결과 확인**: 빌드가 성공하면 `build/bin/` 디렉토리에 실행 파일 (`superpoint_inference`)이 생성됩니다.

---

## 3. 실행 방법

빌드가 완료되면 `build/bin/` 디렉토리에서 생성된 실행 파일을 사용하여 추론을 수행할 수 있습니다.

```bash
# build 디렉토리 내에서 실행
./bin/superpoint_inference <path_to_onnx_model> <path_to_image> <output_visualization_path>
