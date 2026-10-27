# M1013 가상 고정 카메라

기존 wrist 모드를 유지하면서 작업대 위 고정 카메라로 관측·계획하는 선택지다.
Gemini 336L을 기준으로 한 기존 가상 RGB 스테레오와 FoundationStereo를 사용한다.
실물 336L의 IR·컬러 정렬이나 센서 오차를 재현하는 모델은 아니다.
Azure Kinect와 실물 fixed 관측·보정은 이 경로에 포함하지 않는다.

## 1. 설정

| 설정 | wrist(기본) | fixed(선택) |
|---|---|---|
| 파일 | `simulation/isaac_sim/config/wrist_camera.json` | `simulation/isaac_sim/config/fixed_camera.json` |
| 카메라 부모 | `link_6` | `base_link`와 일치하는 가상 월드 |
| 카메라 충돌 | 팔에 부착된 형상 | 월드에 고정된 장애물 |
| 광학 | 이상적인 RGB 스테레오 | 동일한 RGB 스테레오 |

fixed 예제는 카메라 rig를 base 좌표 `(0.65, 0, 1.2)m`에 두고 수직 아래로 향한다.
rig는 스테레오 중점이고, 관측 영상·`world_from_cam`은 왼쪽 광학 프레임 기준이다.
책상 윗면은 Z=0.4m다. `q_capture`의 영 관절값은 가상의 시작 자세이며 실물 이동 명령이 아니다.
현재 설정은 가상 후보이며 실제 영상·계획·물리 집기 성공은 별도로 검증한다.

카메라 body·브래킷은 자동으로 고정 충돌 장애물에 포함한다. 추가 지지대·구조물은
같은 파일의 `fixed_obstacles`에 지정하면 Isaac 장면과 계획기에 함께 반영된다.
단위는 m이며 `base_from_obstacle`은 해당 상자 중심 프레임에서 base로 가는 변환이다.
이름은 영문/밑줄로 시작하고 영문·숫자·밑줄만 사용한다.

```json
{
  "name": "camera_support",
  "dims_m": [0.05, 0.05, 1.2],
  "base_from_obstacle": [
    [1, 0, 0, 1.15],
    [0, 1, 0, 0.5],
    [0, 0, 1, 0.6],
    [0, 0, 0, 1]
  ]
}
```

구조물의 실제 위치를 입력한다. 보이지 않는 공간을 자동으로 빈 공간으로 취급하지 않으며,
로봇 자기 점군 제거·전체 책상·주변 관측 장애물 검사는 유지한다.

## 2. GUI와 연결 검사

프로젝트 루트에서 실행한다. Isaac Sim, FoundationStereo, FoundationPose,
SAM3 서버 준비는 [기존 실행 절차](known-object-pick-place.md#실행-절차)를 따른다.
아래 `ISAAC_PYTHON`에는 이 PC의 Isaac `python.sh` 절대 경로를 넣는다.
Isaac은 ROS/Pixi 환경이 섞이지 않은 별도 터미널에서 실행한다.

```bash
export ISAAC_PYTHON=/absolute/path/to/isaac-sim/python.sh

"$ISAAC_PYTHON" --no-ros-env simulation/isaac_sim/live_perception.py \
  --camera-preparation simulation/isaac_sim/config/fixed_camera.json \
  --asset-root /nas/ochansol/3d_model/2026_real_world_test_objects \
  --output tiptop_outputs/m1013/isaac/fixed-camera-01/runtime
```

`--asset-root`는 장면 물체 USD의 루트만 지정한다. CAD·파지 DB는 현재 카탈로그를 사용한다.
다른 데이터 위치에서는 해당 경로를 바꾼다. 기존 장면의 USD 경로가 맞으면 옵션을 생략할 수 있다.
새 세션에는 새 출력 폴더를 사용한다. 다른 Isaac/Orbbec 관측 서버와 포트 13345를 공유할 수 없다.

`LIVE_CAMERA_READY ... camera_mount=fixed`를 확인한 뒤 호스트 Pixi 터미널에서 검사한다.
이 검사는 새 영상과 연결만 확인하며 추론·팔/그리퍼 동작을 요청하지 않는다.

```bash
pixi run python -m tiptop.scripts.run_known_pick_place \
  --perception-mode sam3_reference \
  --task 'black_pepper_shaker를 집고 빈곳에 놓아' --check
```

관측 서버가 카메라·구조물 설정을 전달하므로 작업 CLI에 같은 파일을 다시 지정할 필요는 없다.
필요하면 `--camera-preparation ...`을 명시해 일치를 검사할 수 있으며 다르면 실패한다.
`--check` 결과의 `camera_mount: fixed`와 각 연결 항목 성공을 확인한다.

## 3. 대표 관측·계획

```bash
pixi run python -m tiptop.scripts.run_known_pick_place \
  --perception-mode sam3_reference \
  --task 'black_pepper_shaker를 집고 빈곳에 놓아' \
  --output tiptop_outputs/m1013/fixed-plan-01
```

`--execute` 없이 마스크·CAD 정합·깊이·빈자리와 계획을 확인한다.
기존 접근각 3°와 계획 허용오차를 유지하며 후보가 없으면 실패 원인을 확인한다.
카메라가 fixed여도 현재 관절값은 계획 시작 자세와 자기 점군 제거에 계속 사용한다.

설정 변경 후에는 runtime을 새로 시작하고 새 관측으로 계획한다.
계획에는 `camera_mount`와 `camera_preparation_sha256`가 저장되며, fixed runtime은
다른 카메라/구조물 설정으로 만든 계획이나 식별 정보가 없는 이전 계획을 거부한다.

## 4. 가상 물리 실행과 wrist 선택

관측·계획 확인 후 동일 작업에 `--execute`를 추가해 GUI에서 대표 1건을 검증한다.
새 명령은 새 관측부터 수행한다. 성공 기준과 실패 후 처리는
[기존 결과 확인](known-object-pick-place.md#결과-확인)을 따른다.

wrist로 실행하려면 현재 가상 runtime을 종료한 뒤 `--camera-preparation`을 생략하거나
기존 `wrist_camera.json`을 지정해 새 세션을 시작한다.
고정 카메라로 바꾼 것만으로 동작 중 영상 추적·재계획이 추가되지는 않는다.
