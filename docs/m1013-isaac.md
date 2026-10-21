# M1013 + UON Isaac 접촉 파지 실행

후추통을 책상에서 집어 10cm 들고, Y방향으로 12cm 옮겨 놓은 뒤 그리퍼를 열고 후퇴한다.
현재 안내는 알려진 물체 하나와 지정 초기 자세를 사용하는 검증용 실행에 해당한다.

## 환경과 입력

[프로젝트 설치 안내](../README_CUSTOM.md)에 따라 TiPToP 환경을 준비하고 프로젝트 루트로 이동한다.
Isaac Sim 6.2는 별도로 설치한다. 아래 명령은 `~/isaac-sim/6.2.0`에 설치한 경우이며,
설치 위치가 다르면 명령의 `python.sh` 경로를 바꾼다.

- 경로 준비: `pixi run python`으로 TiPToP 환경을 사용한다. CUDA 사용 환경이 필요하다.
- 시뮬레이션: `~/isaac-sim/6.2.0/python.sh --no-ros-env`로 Isaac 내장 Python을 사용한다. Pixi로 실행하지 않는다.
- 팔 모델: `ros2_ws/src/drivers/manipulator/doosan-robot2/dsr_description2/usd/m1013.usd`.
- 그리퍼 모델: `UON_3finger_gripper/UON_3finger_gripper.usd`.
- 물체 모델: `data/black_pepper_shaker/3d_model/edited/black_pepper_shaker_rigid.usd`.
- 장착/TCP 설정: `ros2_ws/src/description/m1013_uon_3f_description/config/mount.yaml`.

USD 자산은 Git 저장소에 포함하지 않으므로 별도로 배치해야 한다.
팔 USD의 `joint_6` armature는 검증 조건인 **1.0**을 사용한다. 실행기가 이 값을 설정해 주지는 않는다.
아래 명령의 spinner 토크 상한 **5 N·m**는 생성 장면에만 적용하며 원본 USD를 변경하지 않는다.

## 준비된 입력으로 실행

이 작업 환경에 보존된 `tiptop_outputs/m1013/isaac/pick-place-cap5/pick.json`을 사용한다.
아래 세 명령을 순서대로 실행한다. `--output` 폴더는 입력 `pick.json`을 읽는 위치이기도 하므로,
처음 실행할 때는 복사 단계를 생략하면 안 된다. 같은 출력 폴더를 재사용하면 결과가 갱신된다.

```bash
mkdir -p tiptop_outputs/m1013/isaac/pick-place-run
cp tiptop_outputs/m1013/isaac/pick-place-cap5/pick.json tiptop_outputs/m1013/isaac/pick-place-run/pick.json
~/isaac-sim/6.2.0/python.sh --no-ros-env simulation/isaac_sim/pick_pepper.py \
  --physics-hz 240 --spinner-max-torque 5 --snapshot \
  --output tiptop_outputs/m1013/isaac/pick-place-run
```

완료 후 GUI 창은 유지된다. 창 없이 검사하고 종료하려면 `--headless --check`를 추가한다.
GUI에서 Pause/Stop 또는 창 닫기는 실행 취소로 처리하며, 중간부터 재개하지 않는다.
준비 당시 모델과 파일 해시가 다르면 실행을 거부한다. 위 입력에는 토크 상한 5 옵션을 유지한다.

## 입력을 다시 준비할 때

다음 로컬 입력이 추가로 필요하다.

- `tiptop_outputs/m1013/tamp/diagnosis-wide/candidate-ik.json`: 선별된 파지 후보와 팔 IK.
- 같은 폴더의 `initial.npz`: 물체 충돌 구.
- `data/black_pepper_shaker/UON_3finger_gripper/` 아래 병합 파지 DB와 `output_grasp/` 수집 기록.

후보 파일은 별도로 준비된 분석 결과다. 현재 명령은 후보 생성부터 자동 수행하지 않으므로,
Git 저장소만 새로 복제한 환경에서는 이 입력도 확보해야 한다.

먼저 선별 목록의 첫 후보로 집기 경로를 준비하고 실제 집기 성공을 확인한다.

```bash
pixi run python simulation/isaac_sim/pick_pepper.py --prepare \
  --candidate-rank 0 --output tiptop_outputs/m1013/isaac/pick-run
~/isaac-sim/6.2.0/python.sh --no-ros-env simulation/isaac_sim/pick_pepper.py \
  --physics-hz 240 --spinner-max-torque 5 --check \
  --output tiptop_outputs/m1013/isaac/pick-run
```

성공한 집기의 입력과 관측된 닫힘각을 이어받아 운반·놓기 경로를 준비한다.

```bash
pixi run python simulation/isaac_sim/pick_pepper.py --prepare-placement \
  --pick-source tiptop_outputs/m1013/isaac/pick-run --place-offset 0 0.12 \
  --output tiptop_outputs/m1013/isaac/pick-place-run
```

이후 앞의 Isaac 실행 명령을 사용한다. 이미 준비한 `pick.json`을 복사하는 단계는 건너뛴다.
새 입력은 준비에 사용한 모델 해시와 토크 상한을 기록하며, 실행 조건이 다르면 거부한다.

## 결과 확인

- 집기만 실행: `CONTACT_PICK_SUCCESS`, `check.json`의 `success`와 `physical_grasp_verified`가 `true`.
- 운반·놓기까지 실행: `CONTACT_PICK_PLACE_SUCCESS`, `success`와 `physical_placement_verified`가 `true`.
- `--output` 폴더에 `check.json`을 저장하고, `--snapshot` 지정 시 `held.png` 또는 `placed.png`를 저장한다.
- 표준출력 로그는 자동 저장하지 않는다. 로그 파일이 필요하면 실행 명령에 `> isaac.log 2>&1`을 붙인다.

물체는 자유 강체로 움직인다. 성공 판정은 실제 들림·접촉 유지와 놓은 뒤 위치·정지 상태,
책상 지지 및 손가락 접촉 해제를 확인한다. 계획기의 물체 부착은 충돌 계산에만 사용한다.

## 기타 확인 모드

| 파일 | 용도 |
|---|---|
| `simulation/isaac_sim/m1013_uon.py` | 팔·그리퍼 조립 검사. `--plan FILE`로 저장된 로봇 명령 재생 가능 |
| `simulation/isaac_sim/grasp_scene.py` | `--prepare`로 IK 준비 후 파지 좌표 확인. 물체 고정·접촉 비활성 |
| `simulation/isaac_sim/replay_grasp_record.py` | 팔 없이 DB 파지를 재현. 기본 중력 0이며 `--gravity`, `--table`로 조건 지정 |

각 파일의 옵션은 `--help`로 확인한다. `grasp_scene.py --prepare`는 TiPToP Python을,
장면 실행은 Isaac Python을 사용한다. 조립·좌표 확인·로봇 명령 재생의 성공은 물리 파지 성공과 구분한다.

## 적용 범위

- 물체 배치가 준비한 자세에서 크게 벗어나면 중단한다. 지정 초기 관절 자세까지의 이동 경로와 동적 재계획은 포함하지 않는다.
- 5 N·m는 현재 시험의 토크 상한이며 실물 정격값이 아니다. 접촉력 유지 피드백 제어는 구현돼 있지 않다.
- 장착/TCP·질량·관성의 실물 보정과 다른 물체·배치의 성공은 별도 검증이 필요하다.
- 팔 USD의 단위 표기와 실제 좌표 차이를 처리하므로 생성 USDA를 GUI에서 직접 열기보다 위 실행기를 사용한다.
