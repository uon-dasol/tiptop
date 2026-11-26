# TiPToP Development Environment

[TiPToP](https://tiptop-robot.github.io/)를 Doosan M1013 로봇팔, UON 3지 그리퍼, Orbbec Gemini 336L 카메라 구성으로 옮긴 auto picking 프로젝트다.
지시문으로 지정한 등록 물체(known object)를 영상에서 인식하고, 파지 DB로 파지를 계획해 집어서 목적지(고정 박스 또는 Gemini가 찾은 면)에 놓는다. 원본 TiPToP 사용법은 [README](README.md)를 참고한다.

| 구분 | 원본 TiPToP | 이 저장소 |
|---|---|---|
| 로봇·그리퍼 | Franka FR3 + Robotiq 2F-85 | Doosan M1013 + UON 3지 그리퍼 |
| 카메라 | ZED (손목) | Orbbec Gemini 336L (고정) |
| 인식 | Gemini ER + SAM2 (임의 물체) | 등록 물체: SAM3 reference 또는 Gemini + SAM2로 분할, FoundationPose로 자세 추정 |
| 파지 | M2T2로 생성 | 파지 DB 조회 (Isaac으로 사전 생성) |
| 계획 | cuTAMP | cuTAMP (동일) |
| 실행 | Bamboo 컨트롤러 | ROS 2 gateway |

## Project Structure

워크스페이스 구조는 아래와 같다.

```
tiptop
├─ curobo/             # pixi run setup-planners로 설치
├─ cutamp/             # pixi run setup-planners로 설치
├─ docs/               # 문서
├─ experiments/        # 실험 도구
├─ FoundationPose/     # 물체 자세 추정 (서브모듈)
├─ FoundationStereo/   # 스테레오 깊이 (선택 설치)
├─ M2T2/               # 파지 생성 (선택 설치)
├─ ros2_ws/            # ROS 워크스페이스 (서브모듈)
├─ SAM3/               # 물체 분할, reference 모드 (서브모듈)
├─ scripts/            # 서버 실행·M1013 보조 스크립트
├─ simulation/         # Isaac Sim
├─ tiptop/             # 메인 패키지
└─ tiptop_outputs/     # 실행 결과 (git 제외)
```

---

## Installation

가상 환경(Isaac) 사용 시, Isaac Sim 설치 환경이 필요하다(개발에는 6.2.0을 사용했다).
기본 위치는 `~/isaac-sim/`이며, 다르면 `ISAAC_PYTHON`에 `python.sh` 경로를 지정한다.

### 1. Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash

# 셸을 다시 시작하거나 아래를 실행
source ~/.bashrc  # 사용하는 셸에 따라 ~/.zshrc
```

### 2. TiPToP, cuRobo, cuTAMP

```bash
git clone --recurse-submodules https://github.com/uon-dasol/tiptop.git
cd tiptop

export TIPTOP_DIR="$PWD"

# 이미 저장소를 복제한 경우 서브모듈 초기화
git submodule update --init --recursive

# Python 의존성 설치
pixi install

# cuRobo, cuTAMP 설치
pixi run setup-planners
```

Orbbec 카메라를 사용하는 경우에만 추가한다.

```bash
pixi run install-orbbec
pixi run python -m pyorbbecsdk.shared.setup_env
```

### 3. FoundationPose

```bash
cd "$TIPTOP_DIR/FoundationPose"

conda env create -f environment.yml
conda activate foundationpose

# PyTorch 설치 (CUDA 12.4, 버전 고정)
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# CUDA 툴킷 설치. label 채널로 버전을 고정해야 하위 패키지도 고정됨
conda install -y -c nvidia/label/cuda-12.4.0 cuda-toolkit

# gcc/g++ 13.4.0으로 다운그레이드 (CUDA 12.4 nvcc는 gcc 13까지만 지원)
conda install -y -c conda-forge 'gcc=13.4.0' 'gxx=13.4.0'

# PyTorch3D / NVDiffRast 소스 빌드 (커밋 고정)
python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@2bce7110d5621ef2a349a72ac3aacbfba8bfb461'
python -m pip install --no-build-isolation 'git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae'

# 나머지 의존성 설치 + mycpp 빌드
python -m pip install -r requirements.txt
bash build_all_conda.sh
```

[가중치](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing)를
`FoundationPose/weights/`에 둔다. 필요한 폴더는 `2023-10-28-18-33-37`, `2024-01-11-20-02-45`다.
추가 사항은 [FoundationPose 안내](FoundationPose/readme.md)를 참고한다.

### 4. SAM3

```bash
cd "$TIPTOP_DIR"

conda create -n sam3 python=3.12 -y
conda activate sam3

python -m pip install 'setuptools<81' numpy==1.26.4 ninja

# PyTorch 설치 (CUDA 12.8)
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

# CUDA 툴킷 설치
conda install -y -c nvidia/label/cuda-12.8.1 cuda-toolkit

python -m pip install -e './SAM3[tiptop-server]'

# cc_torch 소스 빌드 (커밋 고정)
python -m pip install --no-build-isolation 'git+https://github.com/ronghanghu/cc_torch.git@446e9360ed10d3c8d03925ccb5f220ccb62009de'

python -m pip install pycocotools
```

`sam3.pt`를 로컬 캐시에 준비하거나 서버의 `--checkpoint`로 지정한다.
가중치 접근 권한과 reference 설정은 [SAM3 가이드](docs/sam3-reference.md)를 따른다.

### 5. ROS Workspace

ROS 드라이버·컨테이너·빌드는 [ROS 워크스페이스](ros2_ws/README_CUSTOM.md)를 따른다.

### 6. Object Data

CAD·파지 DB 경로는 `tiptop/config/known_objects.json`의 `cad_root`·`db_root`에서 관리한다.
기본 경로는 NAS(`/nas`)라 마운트돼 있어야 하고, 폴더 구조는 [물체 데이터](docs/known-object-pick-place.md#물체-데이터)를 따른다.

### 7. Shell Configuration

서버 스크립트는 기본적으로 `~/miniforge3/envs/`의 Python을 쓰므로 miniconda 등 다른 경로면 `~/.bashrc`에 실제 경로를 지정한다.
뷰어의 서비스 버튼은 `bash -ic`로 명령을 실행하므로 alias도 `~/.bashrc`에 둔다.
Gemini 인식 모드(`gemini_er`)를 쓰려면 Google API 키도 지정한다.

```bash
export FOUNDATIONPOSE_PYTHON="$HOME/miniconda3/envs/foundationpose/bin/python"
export SAM3_PYTHON="$HOME/miniconda3/envs/sam3/bin/python"
export GOOGLE_API_KEY="<your-api-key>"
alias fp='conda activate foundationpose'
alias sam='conda activate sam3'
```

---

## Optional Installation

### 1. FoundationStereo

```bash
cd "$TIPTOP_DIR"
git clone https://github.com/williamshen-nz/FoundationStereo.git
cd FoundationStereo

# 의존성 설치
pixi run setup

# 사전 학습 가중치 다운로드
pixi run download-checkpoints
```

<details>
<summary><code>nvcc was not found</code> 또는 <code>CUDA_HOME ... not set</code> 오류 발생 시</summary>

아래 명령은 FoundationStereo 폴더에서 해당 오류가 발생한 경우에만 실행한다.

```bash
# 이 프로젝트의 Pixi 환경에 CUDA 개발 도구 추가
pixi add 'cuda-toolkit=12.*'

# CUDA 경로를 지정하고 설치 재시도
pixi run bash -c 'CUDA_HOME="$CONDA_PREFIX" python -m pip install flash-attn --no-build-isolation'
```

</details>

### 2. M2T2

```bash
cd "$TIPTOP_DIR"
git clone https://github.com/uon-dasol/M2T2.git
cd M2T2

# 의존성 설치
pixi run setup

# 사전 학습 가중치 다운로드
pixi run download-weights
```

---

## Installation Verification

### 1. TiPToP, cuRobo, cuTAMP

```bash
cd "$TIPTOP_DIR"

# cuTAMP 동작 확인 (Rerun 화면이 열린다)
pixi run cutamp-demo --motion_plan

# import 확인
pixi run tiptop-run -h
```

### 2. FoundationPose

```bash
cd "$TIPTOP_DIR"

# 환경, 가중치, CAD 점검 (GPU 사용 안 함). 마지막 줄이 `FoundationPose check: OK`이면 성공
bash scripts/run_foundation_pose_server.sh --check
```

### 3. SAM3

```bash
cd "$TIPTOP_DIR"

# 환경 점검. `SAM3 check: OK`가 출력되면 성공
bash scripts/run_sam3_server.sh --check

# 서버를 시작한 뒤 별도 터미널에서 확인
curl --fail --silent --show-error http://127.0.0.1:13347/health
```

health 응답의 `status`가 `ready`이면 정상이다.

### 4. FoundationStereo (Optional)

```bash
cd "$TIPTOP_DIR/FoundationStereo"

# 서버와 클라이언트 전체 데모 실행
# q로 창 닫기
# Ctrl+B를 누른 다음 D를 눌러 세션에서 연결을 해제
pixi run demo
```

### 5. M2T2 (Optional)

```bash
cd "$TIPTOP_DIR/M2T2"

# meshcat 시각화 포함 전체 데모 실행 (m2t2_demo.sh)
# Ctrl+B를 누른 다음 D를 눌러 세션에서 연결을 해제
pixi run demo
```

<details>
<summary><code>tmux is not installed</code> 오류 발생 시</summary>

tmux를 설치한 뒤 해당 폴더에서 데모를 다시 실행한다.

```bash
sudo apt install tmux
pixi run demo
```

</details>

---

## Usage

뷰어를 실행하고 화면에서 서비스를 시작해 작업을 실행한다.

```bash
cd "$TIPTOP_DIR" && pixi run python -m tiptop.scripts.serve_perception_viewer   # http://127.0.0.1:8091
```

서비스별 포트는 아래와 같다.

| 포트 | 서비스 |
|---|---|
| 8090 | gateway |
| 8091 | 뷰어 |
| 13344 | FoundationStereo |
| 13345 | 관측 서버 (Isaac 가상) |
| 13346 | FoundationPose |
| 13347 | SAM3 |
| 13348 | 관측 서버 (실물) |

스테레오 추론이 필요하면 `bash scripts/run_foundation_stereo_server.sh`로 `13344` 서버를 실행한다.

---

## Documentation

| 문서 | 내용 |
|---|---|
| [작업 실행](docs/known-object-pick-place.md) | 인식·계획·실행·결과 확인 |
| [SAM3 reference](docs/sam3-reference.md) | 인식 서버와 후보 선택 |
| [뷰어](docs/perception-viewer.md) | 인식 결과, 실행 패널, 서비스 시작·종료 |
| [가상 고정 카메라](docs/m1013-fixed-camera.md) | Isaac 실험실 환경 (`--lab`) |
| [실물 준비](docs/m1013-hardware-preparation.md) | 관측 서버·설정·보정 |
| [실물 기동](docs/m1013-real-connection.md) | 팔·그리퍼·gateway·상태 확인·정지 |
| [그리퍼 설정](docs/m1013-gripper-check.md) | TCP·DB 보정·개폐 전류 |
| [로봇 한계](docs/m1013-robot-limits.md) | 컨트롤러 제한과 계획 한계 |
| [충돌 설정](docs/m1013-collision.md) | 충돌 구·모델 원본·외부 패키지 패치 |
| [실험 도구](experiments/README.md) | 저장 결과·경로 재현·지지면 비교 |
