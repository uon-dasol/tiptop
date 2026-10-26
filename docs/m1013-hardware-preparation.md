# M1013 실물 연결 준비

공통 TiPToP 인식·계획·실행기를 실물 장치에 연결하기 위한 순서다.
가상 실행은 [알려진 물체 실행 가이드](known-object-pick-place.md), ROS 연결은
[M1013 실행 가이드](m1013-execution.md)를 함께 참고한다.
실물 장착·보정·팔/그리퍼 구동과 파지 성공은 현장에서 검증해야 한다.

## 1. 장착과 작업 좌표 확정

설치 후 작업 영역과 사용할 대상이 컬러 영상에 보이는지 확인한다.
실물 카메라 설치 자세를 가상 `q_capture`나 임시 장착 행렬로 대체하지 않는다.

`tiptop/config/m1013_hardware.example.json`을 별도 현장 파일로 복사해 다음을 측정한다.
모든 길이는 m, 관절각은 rad, 작업 좌표계는 로봇 `base_link`다.

| 설정 | 의미 |
|---|---|
| `tool.link_6_from_gripper_body` | 실제 플랜지/어댑터와 기준 그리퍼 body 사이 변환 |
| `tool.gripper_body_from_tcp` | 그리퍼 body에서 실행 TCP로 가는 변환 |
| `tool.db_grasp_from_tcp` | DB의 grasp 좌표에서 측정 TCP로 가는 변환 |
| `camera_profile.parent_from_rig` | `link_6`에서 카메라 충돌 외형 body로 가는 변환 |
| `camera_calibration` | 별도 컬러 광학 프레임 손눈 보정 JSON 경로 |
| `workspace.fixed_obstacles` | 전체 책상·고정 장애물의 치수와 `base_from_obstacle` |

행렬 이름 `parent_from_child`는 child 좌표의 점을 parent 좌표로 변환한다.
카메라 body 충돌 외형과 **컬러 광학 프레임 손눈 보정은 별개**다.
카메라/브래킷 치수와 위치, 기준 그리퍼 형상 일치 여부도 확인해야 한다.
예제의 `null`과 `calibrated:false`는 의도적이다. 플래그만 바꾸면 보정된 것이 아니다.
전체 책상·벽·고정 장비는 실측한 치수와 위치로 `workspace.fixed_obstacles`에 넣는다.
깊이 영상에서 추정하는 지지면·빈자리는 이 고정 장애물 설정을 대신하지 않는다.

## 2. 상태 전용 ROS gateway

ROS 작업 공간에서 `tiptop_execution`을 빌드하고 해당 install 환경을 사용한다.
팔 드라이버 연결 절차는 기존 실행 가이드를 따른다. 보정 표본 수집에는 아래 모드를 쓴다.

```bash
colcon build --packages-select tiptop_execution
source install/setup.bash
ros2 run tiptop_execution gateway --namespace dsr01 --robot-system real --read-only
```

`/state`의 `robot_system=0`, `motion_status=0`, 최신 관절 피드백을 확인한다.
이 모드는 그리퍼 명령 publisher와 모델 애니메이션을 만들지 않으며,
궤적·그리퍼·정지 명령 요청을 거부한다. UON 드라이버를 켤 필요가 없다.
로봇 자세 변경은 현장의 작업자가 별도 절차로 수행하고, 각 표본은 정지 후 수집한다.

## 3. 손눈·TCP 보정

사용할 보드 규격은 현장에서 확정한다. 아래 도구는 ChArUco `DICT_4X4_50`을 사용한다.
회사 보드가 다른 형식·사전이면 그대로 적용하지 말고 보정 도구를 맞춰야 한다.
현재 도구로 새 표적을 만들 때는 프로젝트 루트에서 다음을 실행한다.

```bash
pixi run python -m tiptop.scripts.calibrate_m1013 board \
  --square-m 0.025 --marker-m 0.018 --output tiptop_outputs/calibration/charuco.png
```

기본 7×5 사각형 표적의 실제 인쇄 크기는 175×125mm다. 인쇄 배율을 실제로 재고,
보드를 평평하게 고정한다. 자동 축소·여백으로 크기가 바뀌면 보정 단위도 틀어진다.

카메라와 그리퍼를 최종 장착하고, 보드를 작업 공간에 고정한다.
최소 10개 자세에서 보드가 보이도록 독립된 두 축 이상의 회전을 포함해 수집한다.

```bash
pixi run python -m tiptop.scripts.calibrate_m1013 capture-hand-eye \
  --board tiptop_outputs/calibration/charuco.json --serial CAMERA_SERIAL \
  --output tiptop_outputs/calibration/hand-eye/sample-01.json

pixi run python -m tiptop.scripts.calibrate_m1013 solve-hand-eye \
  tiptop_outputs/calibration/hand-eye/sample-*.json \
  --output tiptop_outputs/calibration/hand-eye-candidate.json
```

각 capture는 장치를 움직이지 않고 한 표본만 기록한다. 서로 다른 카메라·표적 크기·프레임을
섞을 수 없다. 이미 왜곡 제거된 컬러 영상에 왜곡 계수를 다시 적용하지 않는다.
6개 이상 ChArUco 코너, 재투영 RMS ≤0.8px, 촬영 전후 관절 변화 ≤0.001rad를 요구한다.

손눈 결과의 보드 고정 일관성 기준은 최대 위치 오차 5mm·회전 1°다.
이 값은 절대 정확도 검증을 대신하지 않는다. 별도의 측정점과 사용하지 않은 자세에서
좌표를 대조한 후에만 현장 보정 파일로 활성화한다. 계산기는 자동 활성화하지 않는다.

TCP 위치는 고정 접촉점에 대해 자세를 바꾸며 표본을 수집할 수 있다.

```bash
pixi run python -m tiptop.scripts.calibrate_m1013 capture-pivot \
  --output tiptop_outputs/calibration/pivot/sample-01.json
pixi run python -m tiptop.scripts.calibrate_m1013 solve-pivot \
  tiptop_outputs/calibration/pivot/sample-*.json --output tiptop_outputs/calibration/pivot-candidate.json
```

Pivot은 TCP **위치만** 구한다. TCP 방향과 DB grasp 정합은 측정한 기준점으로 확인한다.
`fit-datums INPUT.json --output OUTPUT.json`은 `source_frame`, `target_frame`,
대응되는 `source_points_m`, `target_points_m`을 입력받아 `target_from_source`를 계산한다.
최소 3개의 일직선상에 있지 않은 점이 필요하다. 반환값은 여전히 검토할 후보이다.

## 4. 측정 모델 사전 검사

현장 프로파일과 컬러 손눈 파일을 준비한 뒤 다음을 실행한다.

```bash
pixi run python -m tiptop.scripts.prepare_m1013_hardware path/to/hardware.json
pixi run python -m tiptop.scripts.prepare_m1013_hardware path/to/hardware.json \
  --output tiptop_outputs/calibration/measured-model
```

첫 명령은 설정만 읽고 장치를 열지 않는다. 두 번째는 계획용 GPU를 사용해 별도 URDF,
카메라 충돌 외형, 변경한 TCP 기준의 열린 그리퍼 충돌 구, `robot.yml`을 만든다.
원본 URDF·CAD·DB를 바꾸지 않는다. 잘못된 단위·프레임·미보정 입력은 거부한다.

## 5. 실물 관측에서 계획까지만 연결

FoundationStereo(13344), FoundationPose(13346), 상태 전용 gateway(8090)를 먼저 준비한다.
Isaac과 관측 포트 13345를 동시에 쓰지 않는다.

```bash
pixi run python -m tiptop.scripts.serve_orbbec_observation \
  --calibration path/to/hand-eye.json --check
pixi run python -m tiptop.scripts.serve_orbbec_observation \
  --calibration path/to/hand-eye.json --output tiptop_outputs/real-observations

pixi run python -m tiptop.scripts.run_known_pick_place \
  --task '후추통 집고 빈 곳에 놓아' --hardware-profile path/to/hardware.json \
  --output tiptop_outputs/real-plan-01 --allow-gemini-once
```

마지막 명령만 Gemini를 1회 사용한다. 자동 재시도하지 않으며 API 호출 허락이 필요하다.
관측 서버는 보정 카메라 일련번호와 실제 로봇 모드를 확인하고 정지 관절값으로 영상 FK를 계산한다.
기본 깊이는 IR 양안→FoundationStereo→컬러 격자 정렬이다.
`--depth-source sensor`는 명시적인 비교 선택이며 실패 시 자동 전환하지 않는다.
이 단계에서는 `--execute`를 생략해 계획까지만 확인한다. 실제 실행에는 측정된
`--hardware-profile`과 명시적인 `--gateway-url`이 모두 필요하며, 명령은 실행 gateway로 보낸다.
읽기 전용 관측 backend는 `/execute`를 거부한다.

SAM3 reference 모드를 쓰려면 마지막 명령에서 `--allow-gemini-once`를 빼고
`--perception-mode sam3_reference --task 'black_pepper_shaker를 집고 빈곳에 놓아'`를 사용한다.
이 경우 SAM3 서버(13347)도 필요하며 Gemini는 호출하지 않는다.

카메라 시야, SAM 마스크, FP의 CAD 정합, DB 접근 방향, 전체 고정 장애물과 계획 궤적을
확인해야 한다. 로봇 자기 점군 제거 결과도 확인한다. 모델·보정 오차로 실제 물체 점이 지워지지 않아야 한다.

## 6. 현장 구동 전에 남은 일

UON 드라이버는 현재 `control_msgs/msg/GripperCommand`의 position·max_effort 필드에
native 위치 count·전류 count를 사용한다. 표준 메시지 이름만 보고 m·N 단위로 보내면 안 된다.
구현된 연결은 토픽이며 GripperCommand action 서버를 가정하지 않는다.

`ros2_ws/src/execution/tiptop_execution/config/uon_native.example.json`을 복사해
실측 스트로크, 열림·닫힘 endpoint, 모터 전류 상한과 접촉 판정 임계값을 채운다.
전류는 signed 16-bit로 해석하며 접촉 추정은 물체를 실제 들었다는 증거가 아니다.
드라이버 시작/종료에 모터 활성화·이동 동작이 있으므로 오프라인 검사 용도로 켜지 않는다.

설치·정합·기준점·구동 한계와 관측 계획을 현장에서 확인한 다음에만 상태 전용 gateway를 종료하고,
기존 실행 절차의 명시적 실물 모드로 전환한다.

```bash
# 장착·보정·그리퍼 설정과 관측 계획 확인을 마친 뒤 실행한다.
ros2 run tiptop_execution gateway --namespace dsr01 --robot-system real \
  --gripper-config path/to/uon-native-measured.json
pixi run python -m tiptop.doosan run tiptop_outputs/real-plan-01/plan.json
```

저장 계획에는 실제/가상 모드와 측정 프로파일·컬러 보정의 해시가 기록된다.
실물 실행기는 가상 계획이나 계획 후 보정 파일이 바뀐 경우를 거부한다.
환경의 물체가 옮겨졌다면 관측·계획부터 다시 수행해야 한다.

연속 작업과 직접 실행은 [실물 어댑터](known-object-pick-place.md#실물-어댑터)를 따른다.
실행 준비만 검사하려면 같은 작업 명령에 `--check --execute`와 측정 프로파일·gateway를 지정한다.
이 검사는 로봇을 움직이지 않는다. 검사 뒤에는 `--check`를 빼고 새로 관측·계획한다.

native 그리퍼는 최신 피드백과 명령/상태 토픽 연결 수를 검사한다. 닫기 완료는 endpoint 도달 또는
안정된 위치·전류로 추정하며, 조건을 만족하면 다음 팔 동작으로 진행한다. 물체 접촉과 기구 걸림을
구분하는 판정은 아니므로 현장에서 확인한다. 시간 초과·취소·피드백 오류 때는 가능한 최신 위치로
hold를 시도한다. 이는 위치 재명령이며 토크 차단이나 실제 파지력 유지 보장이 아니다.
명령에는 실측한 native count를 사용한다. 공통 실행기의 정규화된 force/speed가 실제 힘·속도에
그대로 적용되는 것은 아니다.

제어 완료 결과는 `execution_success:true`, `success:null`이다. 실물의 물체 유지·방출은
별도로 확인해야 하며, 절대 깊이·손눈/TCP 정확도·팔 추종·접촉 판정도 현장 검증 대상이다.
