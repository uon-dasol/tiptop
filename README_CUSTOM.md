# TiPToP 개발 환경

현재 M1013 실험실 구성의 설치·실행 안내다. 원본 TiPToP 사용법은 [README](README.md)를 참고한다.

## 기본 설치

```bash
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc

git clone https://github.com/uon-dasol/tiptop.git
cd tiptop
export TIPTOP_DIR="$PWD"
git submodule update --init --recursive
pixi install
pixi run setup-planners
```

Orbbec 카메라를 사용하는 경우에만 추가한다. ZED는 `pixi run install-zed`를 사용한다.

```bash
pixi run install-orbbec
pixi run python -m pyorbbecsdk.shared.setup_env
```

ROS 드라이버·컨테이너·빌드는 [ROS 워크스페이스](ros2_ws/README_CUSTOM.md)를 따른다.
CAD·파지 DB 경로는 `tiptop/config/known_objects.json`의 `cad_root`·`db_root`에서 관리한다.

## FoundationPose

별도 conda 환경에서 설치한다.

```bash
cd "$TIPTOP_DIR/FoundationPose"
conda env create -f environment.yml
conda activate foundationpose
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
conda install -y -c nvidia/label/cuda-12.4.0 cuda-toolkit
conda install -y -c conda-forge 'gcc=13.4.0' 'gxx=13.4.0'
python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@2bce7110d5621ef2a349a72ac3aacbfba8bfb461'
python -m pip install --no-build-isolation 'git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae'
python -m pip install -r requirements.txt
bash build_all_conda.sh
```

[가중치](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing)를
`FoundationPose/weights/`에 둔다. 필요한 폴더는 `2023-10-28-18-33-37`, `2024-01-11-20-02-45`다.
추가 사항은 [FoundationPose 안내](FoundationPose/readme.md)를 참고한다.

## SAM3

Reference 모드를 사용할 때 별도 conda 환경에 설치한다.

```bash
cd "$TIPTOP_DIR"
conda create -n sam3 python=3.12 -y
conda activate sam3
python -m pip install 'setuptools<81' numpy==1.26.4 ninja
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
conda install -y -c nvidia/label/cuda-12.8.1 cuda-toolkit
python -m pip install -e './SAM3[tiptop-server]'
python -m pip install --no-build-isolation 'git+https://github.com/ronghanghu/cc_torch.git@446e9360ed10d3c8d03925ccb5f220ccb62009de'
python -m pip install pycocotools
```

`sam3.pt`를 로컬 캐시에 준비하거나 서버의 `--checkpoint`로 지정한다.
가중치 접근 권한과 reference 설정은 [SAM3 가이드](docs/sam3-reference.md)를 따른다.

## 선택 설치

현재 `--lab`과 실물 센서 깊이 경로에는 FoundationStereo가 필요하지 않다.
스테레오 추론을 사용할 때만 설치한다.

```bash
cd "$TIPTOP_DIR"
git clone https://github.com/williamshen-nz/FoundationStereo.git
cd FoundationStereo
pixi run setup
pixi run download-checkpoints
```

`nvcc`·`CUDA_HOME` 오류가 발생하면 해당 폴더에서 설치를 보완한다.

```bash
pixi add 'cuda-toolkit=12.*'
pixi run bash -c 'CUDA_HOME="$CONDA_PREFIX" python -m pip install flash-attn --no-build-isolation'
```

기존 M2T2 경로를 사용할 때는 별도로 설치한다.

```bash
cd "$TIPTOP_DIR"
git clone https://github.com/uon-dasol/M2T2.git
cd M2T2
pixi run setup
pixi run download-weights
```

## 서버 실행

FoundationPose는 `13346`, SAM3는 `13347`을 사용한다. 스크립트는 기본적으로 `~/miniforge3/envs/`의 Python을 쓰므로
miniconda 등 다른 경로면 `~/.bashrc`에 실제 경로를 지정한다. 뷰어의 서비스 버튼은 `bash -ic`로 명령을 실행하므로 alias도 `~/.bashrc`에 둔다.

```bash
export FOUNDATIONPOSE_PYTHON="$HOME/miniconda3/envs/foundationpose/bin/python"
export SAM3_PYTHON="$HOME/miniconda3/envs/sam3/bin/python"
alias fp='conda activate foundationpose'
alias sam='conda activate sam3'
```

프로젝트 루트에서 각각 별도 터미널로 실행하거나 [뷰어](docs/perception-viewer.md#서비스-시작종료)에서 시작한다.

```bash
bash scripts/run_foundation_pose_server.sh --check   # 설치 확인
bash scripts/run_foundation_pose_server.sh --preload-objects all
bash scripts/run_sam3_server.sh --check              # SAM3_READY가 나오면 정상
bash scripts/run_sam3_server.sh
```

스테레오 추론이 필요하면 `bash scripts/run_foundation_stereo_server.sh`로 `13344` 서버를 실행한다.

## 문서

| 문서 | 내용 |
|---|---|
| [작업 실행](docs/known-object-pick-place.md) | 인식·계획·실행·결과 확인 |
| [SAM3 reference](docs/sam3-reference.md) | 인식 서버와 후보 선택 |
| [뷰어](docs/perception-viewer.md) | 인식 결과, 실행 패널, 서비스 시작·종료 |
| [가상 고정 카메라](docs/m1013-fixed-camera.md) | Isaac 실험실 환경 (`--lab`) |
| [실물 준비](docs/m1013-hardware-preparation.md) | 관측 서버·설정·보정 |
| [실물 기동](docs/m1013-real-connection.md) | 팔·그리퍼·gateway |
| [그리퍼 설정](docs/m1013-gripper-check.md) | TCP·DB 보정·개폐 전류 |
| [로봇 한계](docs/m1013-robot-limits.md) | 컨트롤러 제한과 계획 한계 |
| [충돌 설정](docs/m1013-collision.md) | 충돌 구·모델 원본·외부 패키지 패치 |
| [경로 계획](docs/m1013-planning.md), [cuTAMP 연결 검사](docs/m1013-tamp.md), [가상 ROS 실행](docs/m1013-execution.md) | 가상 ROS·RViz 검증 |
| [Isaac 접촉 파지](docs/m1013-isaac.md), [팔 왕복 시험](docs/m1013-arm-trial.md) | 물리 검증 |
| [실험 도구](experiments/README.md) | 저장 결과·경로 재현·지지면 비교 |
