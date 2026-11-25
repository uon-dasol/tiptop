# M1013 + UON Isaac 접촉 파지 검증

후추통을 책상에서 집어 10 cm 들고, Y 방향으로 12 cm 옮겨 놓은 뒤 그리퍼를 열고 후퇴하는 물리 검증이다.
알려진 물체 하나와 지정 초기 자세를 쓰는 검증용 실행이며, 현재 영상 기반 작업은 [작업 실행](known-object-pick-place.md)을 따른다.

## 환경과 입력

- 경로 준비: `pixi run python`(TiPToP 환경, CUDA 필요). 시뮬레이션: `~/isaac-sim/6.2.0/python.sh --no-ros-env`(Isaac 내장 Python, Pixi로 실행하지 않음). 설치 위치가 다르면 경로를 바꾼다.
- 팔 모델: `ros2_ws/src/drivers/manipulator/doosan-robot2/dsr_description2/usd/m1013.usd`
- 그리퍼 모델: `ros2_ws/src/description/uon_3f_description/usd/UON_3finger_gripper.usd` (ROS 실행 없이 파일만 읽음)
- 물체 모델: 카탈로그 `cad_root` 아래 `black_pepper_shaker/edited/black_pepper_shaker_rigid.usd` ([물체 데이터](known-object-pick-place.md#물체-데이터))
- 장착/TCP: `ros2_ws/src/description/m1013_uon_3f_description/config/mount.yaml`
- 후보 입력(별도 준비 필요, 저장소에 없음): `tiptop_outputs/m1013/tamp/diagnosis-wide/candidate-ik.json`(선별 파지 후보와 팔 IK), 같은 폴더의 `initial.npz`(물체 충돌 구), 카탈로그 `db_root` 아래 병합 파지 DB
- 팔 USD의 `joint_6` armature는 검증 조건인 **1.0**을 쓴다. 실행기가 설정해 주지 않는다.
- 아래 spinner 토크 상한 **5 N·m**는 생성 장면에만 적용되며 원본 USD는 바꾸지 않는다. 실물 정격값이 아니다.
- 원본 OBJ·URDF·USD는 `/media/uon/data1/UON_3finger_gripper/`에 있고, 재생성은 `uon_3f_description/scripts/import_assets.py --source-dir`로 한다.

## 실행

1. 선별 목록의 첫 후보로 집기 경로를 준비하고 집기 성공을 확인한다.

```bash
pixi run python simulation/isaac_sim/pick_pepper.py --prepare \
  --candidate-rank 0 --output tiptop_outputs/m1013/isaac/pick-run
~/isaac-sim/6.2.0/python.sh --no-ros-env simulation/isaac_sim/pick_pepper.py \
  --physics-hz 240 --spinner-max-torque 5 --check \
  --output tiptop_outputs/m1013/isaac/pick-run
```

2. 성공한 집기의 입력과 닫힘각을 이어받아 운반·놓기 경로를 준비한다.

```bash
pixi run python simulation/isaac_sim/pick_pepper.py --prepare-placement \
  --pick-source tiptop_outputs/m1013/isaac/pick-run --place-offset 0 0.12 \
  --output tiptop_outputs/m1013/isaac/pick-place-run
```

3. 운반·놓기까지 실행한다.

```bash
~/isaac-sim/6.2.0/python.sh --no-ros-env simulation/isaac_sim/pick_pepper.py \
  --physics-hz 240 --spinner-max-torque 5 --snapshot \
  --output tiptop_outputs/m1013/isaac/pick-place-run
```

- 완료 후 GUI 창은 유지된다. 창 없이 검사하고 종료하려면 `--headless --check`를 추가한다.
- GUI에서 Pause/Stop이나 창 닫기는 실행 취소다. 중간부터 재개하지 않는다.
- 준비 당시 모델과 파일 해시, 토크 상한이 다르면 실행을 거부한다. 이 값들은 `pick.json`에 기록된다.
- 표준출력은 자동 저장하지 않는다. 로그가 필요하면 `> isaac.log 2>&1`을 붙인다.

## 결과

| 실행 | 성공 조건 |
|---|---|
| 집기만 | `CONTACT_PICK_SUCCESS`, `check.json`의 `success`·`physical_grasp_verified`가 `true` |
| 운반·놓기까지 | `CONTACT_PICK_PLACE_SUCCESS`, `success`·`physical_placement_verified`가 `true` |

`check.json`은 `--output` 폴더에 저장되고, `--snapshot`이면 `held.png` 또는 `placed.png`도 저장된다.
물체는 자유 강체다. 성공 판정은 실제 들림·접촉 유지, 놓은 뒤 위치·정지 상태, 책상 지지, 손가락 접촉 해제를 확인한다. 계획기의 물체 부착은 충돌 계산에만 쓴다.

## 기타 확인 모드

| 파일 | 용도 |
|---|---|
| `simulation/isaac_sim/m1013_uon.py` | 팔·그리퍼 조립 검사. `--plan FILE`로 저장된 로봇 명령 재생 |
| `simulation/isaac_sim/grasp_scene.py` | `--prepare`(TiPToP Python)로 IK 준비 후 파지 좌표 확인. 물체 고정·접촉 비활성 |
| `simulation/isaac_sim/replay_grasp_record.py` | 팔 없이 DB 파지를 재현. 기본 중력 0이며 `--gravity`, `--table`로 조건 지정 |

옵션은 `--help`로 확인한다. 조립·좌표 확인·명령 재생의 성공은 물리 파지 성공과 구분한다.

## 한계

- 물체 배치가 준비한 자세에서 크게 벗어나면 중단한다. 지정 초기 자세까지의 이동 경로와 동적 재계획은 없다.
- 접촉력 유지 피드백 제어는 없다. 장착/TCP·질량·관성의 실물 보정과 다른 물체·배치의 성공은 별도 검증이 필요하다.
- 팔 USD의 단위 표기와 실제 좌표 차이를 처리하므로 생성 USDA를 GUI에서 직접 열지 말고 위 실행기를 쓴다.
