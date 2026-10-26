## Project Structure

워크스페이스 구조는 아래와 같습니다.

```
tiptop
├─ curobo/
├─ cutamp/
├─ FoundationPose/
├─ FoundationStereo/
├─ M2T2/
├─ SAM3/
├─ ros2_ws/
|  └─ README_CUSTOM.md
├─ .gitignore
├─ .gitmodules
├─ LICENSE
├─ README_CUSTOM.md
└─ README.md
```

---

## Installation

### 1. Pixi 설치

```bash
curl -fsSL https://pixi.sh/install.sh | bash

# Restart your shell or run:
source ~/.bashrc  # or ~/.zshrc depending on your shell
```

### 2. TiPToP, cuRobo, cuTAMP 설치

```bash
# Clone the repository
git clone https://github.com/uon-dasol/tiptop.git
cd tiptop

export TIPTOP_DIR="$PWD"

# Install Python dependencies
pixi install

# Setup and install cuRobo and cuTAMP (this may take 5-15 minutes)
pixi run setup-planners

# ZED 카메라를 사용하는 경우에만 실행
# pixi run install-zed

# Orbbec Gemini 336L 카메라를 사용하는 경우에만 실행
# pixi run install-orbbec
```

### 3. M2T2 설치

```bash
# Clone the repository
cd "$TIPTOP_DIR"
git clone https://github.com/uon-dasol/M2T2.git
cd M2T2

# Install dependencies (this may take 2-5 minutes)
pixi run setup

# Download pretrained weights
pixi run download-weights
```

### 4. FoundationStereo 설치

```bash
# Clone the repository
cd "$TIPTOP_DIR"
git clone https://github.com/williamshen-nz/FoundationStereo.git
cd FoundationStereo

# Install dependencies (this should take 1-2 minutes)
pixi run setup
```

<details>
<summary><code>nvcc was not found</code> 또는 <code>CUDA_HOME ... not set</code> 오류 발생 시</summary>

아래 명령은 FoundationStereo 폴더에서 해당 오류가 발생한 경우에만 실행합니다.

```bash
# 이 프로젝트의 Pixi 환경에 CUDA 개발 도구 추가
pixi add "cuda-toolkit=12.*"

# 컴파일러 설치 확인
pixi run nvcc --version

# CUDA 경로를 지정하고 설치 재시도
pixi run bash -c 'CUDA_HOME="$CONDA_PREFIX" python -m pip install flash-attn --no-build-isolation'
```

설치 성공 후 아래 체크포인트 다운로드 단계로 진행합니다.

</details>

```bash
# Download pretrained checkpoints
pixi run download-checkpoints
```

### 5. ROS2 Workspace 설치

```bash
cd "$TIPTOP_DIR"
git submodule update --init --recursive
cd ros2_ws
```

현재 디렉터리에서 이어서 진행할 외부 레포 준비, 개발 환경 설정, ROS2 빌드 및 실행은
[ROS2 워크스페이스 설치 및 실행 안내](ros2_ws/README_CUSTOM.md#5-development-environment-setup)를 따릅니다.
ROS2 개발 환경은 Docker로 구성하며, 위의 Pixi 설치와 별도로 진행합니다.

### 6. FoundationPose 설치

FoundationPose는 Pixi와 별도의 conda 환경(`foundationpose`)에서 실행합니다.

```bash
# FoundationPose는 서브모듈(dasol 브랜치)입니다. 5단계에서 이미 받았다면 건너뛰어도 됩니다
cd "$TIPTOP_DIR"
git submodule update --init FoundationPose
cd FoundationPose

conda env create -f environment.yml
conda activate foundationpose

# PyTorch 설치 (CUDA 12.4, 버전 고정)
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# CUDA 툴킷 설치. label 채널로 버전을 고정해야 하위 패키지도 고정됨
conda install -y -c nvidia/label/cuda-12.4.0 cuda-toolkit

# gcc/g++ 13.4.0으로 다운그레이드 (CUDA 12.4 nvcc는 gcc 13까지만 지원)
conda install -y -c conda-forge "gcc=13.4.0" "gxx=13.4.0"

# PyTorch3D / NVDiffRast 소스 빌드 (커밋 고정)
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@2bce7110d5621ef2a349a72ac3aacbfba8bfb461"
python -m pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae"

# 나머지 의존성 설치 + mycpp 빌드
python -m pip install -r requirements.txt
bash build_all_conda.sh
```

모델 가중치는 [FoundationPose 공식 폴더](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing)에서
받아 `FoundationPose/weights/`에 둡니다 (`2023-10-28-18-33-37`, `2024-01-11-20-02-45`).
물체의 CAD·파지 DB는 [`tiptop/config/known_objects.json`](tiptop/config/known_objects.json)의
`cad_root`·`db_root` 아래에서 읽습니다. 현재 기본 경로는 아래와 같습니다.

- CAD: `/media/uon/data1/3d_model/2026_real_world_test_objects/<물체>/edited/<물체>.obj` (+ `.mtl`, 텍스처)
- 파지 DB: `/media/uon/data1/Dataset_2026/isaacsim_grasp_data_gen/<물체>/UON_3finger_gripper/<물체>_merged_grasp_zero_pose.json`

```bash
cd "$TIPTOP_DIR"

# 서버 실행 (포트 13346, GPU 사용). `FOUNDATIONPOSE_READY port=13346`가 출력되면 준비된 것입니다
bash scripts/run_foundation_pose_server.sh
```

참고 문서:

- [`FoundationPose/tiptop_server/server.py`](FoundationPose/tiptop_server/server.py): 위 스크립트가 띄우는 서버 (M2T2, FoundationStereo처럼 서버 코드는 해당 저장소에 있음)
- [`FoundationPose/readme_custom.md`](FoundationPose/readme_custom.md): `dasol` 브랜치에 추가된 실행기와 사용법
- [`FoundationPose/readme.md`](FoundationPose/readme.md): 원본 안내 (가중치 "Data prepare", 문제 해결)
- [data_gen](https://github.com/uonrobotics/data_gen) README의 "Create FoundationPose Inference Environment": 위 conda 환경 절차의 출처

### 7. SAM3 설치 (reference 인식 모드)

SAM3는 별도 conda 환경을 사용합니다. Gemini ER 모드만 사용하면 생략합니다.

```bash
cd "$TIPTOP_DIR"
git submodule update --init SAM3
conda create -n sam3 python=3.12 -y
conda activate sam3
python -m pip install 'setuptools<81' numpy==1.26.4 ninja
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e './SAM3[tiptop-server]'
python -m pip install 'git+https://github.com/ronghanghu/cc_torch.git@446e9360ed10d3c8d03925ccb5f220ccb62009de'
```

가중치·reference 준비와 빌드 조건은 [SAM3 가이드](docs/sam3-reference.md)를 따릅니다.
준비 후 서버를 실행합니다. `SAM3_READY`가 출력되면 준비된 상태입니다.

```bash
bash scripts/run_sam3_server.sh
```

---

## Installation Verification

### 1. TiPToP·cuRobo·cuTAMP

```bash
cd "$TIPTOP_DIR"

# Validate cuTAMP is installed and working. You should see a Rerun screen open
pixi run cutamp-demo --motion_plan

# Run tiptop-run command with help flag to check imports all work
pixi run tiptop-run -h
```

### 2. M2T2

```bash
cd "$TIPTOP_DIR/M2T2"

# Run the full demo with meshcat visualization (runs m2t2_demo.sh)
# Ctrl+ B 키를 누른 다음 D를 눌러 세션에서 연결을 해제
pixi run demo
```

<details>
<summary><code>tmux is not installed</code> 오류 발생 시</summary>

tmux를 설치한 뒤 M2T2 폴더에서 데모를 다시 실행합니다.

```bash
sudo apt install tmux
pixi run demo
```

</details>

### 3. FoundationStereo

```bash
cd "$TIPTOP_DIR/FoundationStereo"

# Run the full demo (runs server and client)
# q로 윈도우 끄기
# Ctrl+ B 키를 누른 다음 D를 눌러 세션에서 연결을 해제
pixi run demo
```

### 4. FoundationPose

```bash
cd "$TIPTOP_DIR"

# 환경, 가중치, CAD를 점검합니다 (GPU 사용 안 함). 마지막 줄이 `FoundationPose check: OK`이면 성공
bash scripts/run_foundation_pose_server.sh --check
```

### 5. SAM3

```bash
cd "$TIPTOP_DIR"
bash scripts/run_sam3_server.sh --check
# 서버 실행 후 별도 터미널에서 확인
curl --fail --silent --show-error http://127.0.0.1:13347/health
```

`SAM3 check: OK`와 health의 `status: ready`를 확인합니다.

---

## Perception Modes

| 모드 | 인식 | Gemini 호출 |
| --- | --- | --- |
| `gemini_er` (기본) | Gemini ER → SAM2 | 작업당 최대1회, 허용 필요 |
| `sam3_reference` | reference 이미지 → SAM3 | 없음 |

- [가상 집기·놓기](docs/known-object-pick-place.md): 실행·연결 검사·계획 인자·결과 확인
- [SAM3 모드](docs/sam3-reference.md): reference 설정·모드 선택 명령
- [실물 연결 준비](docs/m1013-hardware-preparation.md): 장착·보정·작업공간·gateway 연결

---

## Tools

프로젝트 루트에서 물체 CAD/파지 방향/접촉점과 별도의 그리퍼 관절을 확인합니다.

```bash
xdg-open docs/analysis/uon-gripper-assets/grasp-viewer.html
```
