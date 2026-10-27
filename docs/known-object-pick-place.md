# M1013 가상 집기·놓기

작업 지시와 현재 카메라 영상을 바탕으로 물체를 인식하고, UON 파지 DB와 계획기를 이용해
Isaac Sim에서 집기·빈자리 배치를 수행한다.

저장 결과 조회·물리 실험 재현·영상/DB 진단은 [실험 도구 가이드](../experiments/README.md)를 참고한다.
실물 장착·보정·장치 연결 준비는 [실물 연결 준비](m1013-hardware-preparation.md)를 참고한다.
Gemini ER/SAM2와 Gemini 없는 SAM3 reference 모드의 선택·별도 서버 설치는
[SAM3 reference 모드](sam3-reference.md)를 참고한다. 기본 모드는 기존 `gemini_er`다.
기존 wrist 대신 가상 상부 고정 카메라를 선택하려면 [고정 카메라 가이드](m1013-fixed-camera.md)를 따른다.

## 물체 데이터

`tiptop/config/known_objects.json`은 아래 CSV의 28종 폴더명·한글 이름·외형 설명을 등록한다.
CSV는 등록 목록의 출처이며, 실행 중 자동으로 다시 읽거나 목록을 갱신하지 않는다.

- 목록: `/media/uon/data1/3d_model/2026_real_world_test_objects/2026_real_world_test_objects_attr.csv`
- `cad_root`: `/media/uon/data1/3d_model/2026_real_world_test_objects`
- `db_root`: `/media/uon/data1/Dataset_2026/isaacsim_grasp_data_gen`
- CAD: `<cad_root>/<물체명>/edited/<물체명>.obj` (축척 0.001, mm→m)
- 파지 DB: `<db_root>/<물체명>/UON_3finger_gripper/<물체명>_merged_grasp_zero_pose.json`

원본을 직접 읽으며 누락 시 오류를 낸다. 프로젝트 `data/`로 자동 대체하지 않는다.
경로가 달라지면 카탈로그의 두 root를 바꾼다. FoundationPose도 같은 `cad_root`를 기본
허용 경로로 사용한다. 이미 실행 중인 FoundationPose 서버는 변경 후 재시작해야 한다.
`--asset-root`를 별도 지정할 경우 클라이언트의 CAD 경로가 그 아래에 있어야 한다.
가상 장면의 USD·CAD·DB 경로는 `simulation/isaac_sim/config/three_objects.json`에도 명시되어 있다.

28종 등록은 데이터 연결 범위다. 기본 Isaac 장면에는 기존 세 물체만 있으며,
클래스당 인스턴스 하나를 지원한다. 28종 전체의 인식·파지 성능은 미검증이다.

## 실행 절차

프로젝트 루트에서 각 단계를 별도 터미널로 실행한다.
Isaac runtime 출력 경로의 `manual-pick-place-01`은 새 세션을 만들 때 새 이름으로 바꾼다.
작업 CLI는 `--output`을 생략해 자동 번호 경로를 사용할 수 있다.
관측/실행 backend는 URL과 관측 ID로 연결하므로 runtime과 작업 폴더의 상위 이름이 같을 필요는 없다.
한 세션에서는 장면과 프로그램을 유지하고 여러 작업을 수행한다.

### 1. FoundationStereo 서버 실행

```bash
bash scripts/run_foundation_stereo_server.sh
```

### 2. FoundationPose 서버 실행

```bash
bash scripts/run_foundation_pose_server.sh
```

### 3. Isaac Sim 실행

```bash
# 가상 장면과 현재 관측/실행 backend. 새 세션을 시작할 때 새 폴더명을 사용한다.
/home/uon/isaac-sim/6.2.0/python.sh --no-ros-env \
  simulation/isaac_sim/live_perception.py \
  --output tiptop_outputs/m1013/isaac/manual-pick-place-01/runtime
```

`LIVE_CAMERA_READY`와 두 추론 서버의 준비 완료를 확인한다.

### 4. 작업 지시 입력

```bash
# GEMINI_API_KEY 또는 GOOGLE_API_KEY가 등록된 터미널에서 실행.
# --allow-gemini-once는 실제 API를 사용한다. 자동 재시도는 하지 않는다.
pixi run python -m tiptop.scripts.run_known_pick_place \
  --interactive \
  --output tiptop_outputs/m1013/isaac/manual-pick-place-01/task \
  --allow-gemini-once --execute --max-approach-tilt-deg 3
```

`작업>` 프롬프트에 `후추통 집고 빈 곳에 놓아`처럼 입력한다. 종료는 `/quit`이다.
각 지시는 새 관측에서 Gemini 최대 1회→마스크/깊이/CAD 자세→파지 DB→계획→실행을 수행한다.
등록된 대상은 CSV 기준 28종이며 기본 가상 장면에는 후추통, 가죽 관리제 병, 미키 바가지가 있다.
클래스당 한 인스턴스 제약은 유지한다. 다른 대상을 임의로 골라 집지 않는다.

성공 후 옮겨진 장면을 그대로 유지한다. 인식/포즈/파지 후보/계획이 실패하면 `TASK_FAILED`와
해당 작업 결과를 남기고 다음 입력을 받는다. 사용자가 물체를 옮겼다면 다음 지시에서 다시 관측한다.
자동 재시도·각도 완화·ROI 재사용·장면 리셋은 하지 않는다. 계획에는 기존 충돌 검사 복귀 경로가 포함된다.
가상 동작 중 제어 실패는 정지로 처리하며, 팔이 정지하고 그리퍼가 열린 상태를 복구하기 전에는
새 동작을 받지 않는다. 현재 물체를 들고 있을 가능성이 있는 상태에서 자동으로 열거나 복귀하지 않는다.

`--execute`를 생략하면 계획 저장까지만 수행한다. 단발 실행은 `--interactive` 대신
`--task '미키 바가지 집고 빈 곳에 놓아'`를 사용한다. 단발 명령을 다시 실행해도 Isaac은 재시작할 필요가 없다.
`--active-task`는 토큰 절약을 위한 동일 작업 진단용이며 interactive 모드와 함께 쓸 수 없다.

### 연결 사전 검사

서버를 준비한 뒤 `--check`로 선택한 경로의 연결을 확인한다.

```bash
pixi run python -m tiptop.scripts.run_known_pick_place \
  --perception-mode sam3_reference \
  --task 'car_leather_conditioner_bottle를 집고 빈곳에 놓아' \
  --reference-scene Home/Uon_1F_Home/floor --check
```

새 카메라 프레임·내부 보정·관절/FK 입력, 필요한 깊이/포즈 서버, CAD·DB 파일,
SAM3 서버의 reference 이미지·BBox·identity를 검사한다. Gemini 모드에서는 API 키 설정과
선택한 SAM2 서버 또는 로컬 가중치를 확인한다. Gemini 인증 요청·모델 추론·다운로드·
팔/그리퍼 명령은 하지 않는다. 정상은 `success:true`와 종료 코드0, 실패는 항목별 이유와 종료 코드1이다.
카메라를 새로 관측하므로 이전 미실행 계획의 관측 ID는 갱신된다. 검사 뒤 작업은 새로 실행한다.

실물에서는 측정 `--hardware-profile`을 추가한다. 상태 전용 gateway도 별도로 확인하려면
`--gateway-url`을 추가할 수 있다. 실제 실행 준비까지 검사하려면 `--check --execute`와
명시적 gateway를 함께 쓴다. 이 조합도 움직이지 않으며, native 그리퍼 피드백·로봇 정지 상태·
관측 이후 관절 변화·이전 제어 실패·trajectory/stop 인터페이스와 그리퍼 토픽 연결 수를 확인한다.
새 `connections` 상태를 제공하도록 ROS `tiptop_execution`을 다시 빌드해야 한다.

이 검사는 연결 확인이며 대상의 검출·포즈 정확도·충돌 없는 계획·실제 파지 성공을 보장하지 않는다.
정상 작업에서도 필요한 FS/FP 서버와 SAM2 연결은 Gemini 호출 전에 확인하며,
실물 실행 직전에 현재 관절·연결 상태를 다시 검사한다.

### 접근 각도

`--max-approach-tilt-deg`는 파지 시점 그리퍼 접근축과 월드 수직 아래 방향 사이의 **최대 각도**다.
기본값은 **3°**이며 수직축 주위 회전(yaw)이나 손안 물체의 기울기와는 다르다.
이 범위에서 가능한 후보가 없으면 실패로 보고하고 15°로 자동 확대하지 않는다.
사용자가 필요에 따라 `--max-approach-tilt-deg 5`처럼 조정할 수 있다(유한한 0~90°).
각도는 충돌 여유의 보장이 아니다. 그리퍼·카메라·팔·전체 책상과 관측 장애물 충돌 검사를 유지한다.
이 값은 파지 후보의 접근 자세를 제한하며 운반 경로 전체의 자세 상한을 뜻하지 않는다.

### 계획 시간과 허용오차

MotionGen의 요청별 탐색 예산은 기본 1초다. 필요하면 `--motion-timeout-s 2`처럼 조정한다.
자동으로 시간을 늘리거나 작업을 다시 실행하지 않는다. 내부 계산 한 번을 중간에 강제 종료하는
시간이나 전체 작업 시간, 로봇의 실제 이동 속도와는 다르다.

| 인자 | 기본값 | 의미 |
|---|---|---|
| `--placement-support-tolerance-m` | `0.01` | 계획한 물체 충돌 구 바닥과 지지 목표 높이의 차이 |
| `--placement-xy-tolerance-m` | `0.01` | 배치 영역 밖으로 벗어난 충돌 구 중심 거리의 합 |
| `--motion-position-tolerance-m` | `0.01` | MotionGen 말단 목표 위치 수렴 오차(m) |
| `--motion-rotation-tolerance` | `0.1` | MotionGen 쿼터니언 회전 수렴 임계값 |
| `--motion-timeout-s` | `1` | 개별 MotionGen 요청의 탐색 예산(초) |
| `--release-clearance-m` | 자동 계산 | 추정 책상에서 CAD 바닥까지의 방출 목표 여유(m) |

방출 여유는 기본적으로 `지지 높이 허용오차 + MotionGen 위치 허용오차 + 0.005m`로 계산한다.
기본값에서는 25mm다.
직접 지정할 때도 앞의 두 허용오차 합보다 커야 하며, 잘못된 값은 인식 전에 거부한다.
이는 모델상 위치 오차를 고려한 여유이며 카메라 보정·실물 추종·회전 오차까지 보장하지 않는다.
기본 방출 높이25mm의 물리 집기·놓기 결과는 미검증이다.

계획 전 파지 후보 IK 기준은 위치1mm·회전 임계값0.01이다.
이 IK는 DB 후보의 도달 가능성 검사이며 MotionGen 경로 목표의 수렴 기준과 별개다.
`plan.json`과 `planning-result.json`의 `planning_settings`에 적용한 값을 기록한다.
`planning_seconds`는 기존 cuTAMP 소요 시간이고, `planning_total_seconds`는 후보 검사·초기화를
포함한 전용 계획 함수의 전체 시간이다. 후보 충돌 검사·IK 시간도 따로 기록한다.

### 실물 어댑터

[실물 준비 절차](m1013-hardware-preparation.md)의 측정 모델·손눈/TCP·native gateway를 준비한 뒤,
같은 명령에 `--hardware-profile path/to/measured.json --gateway-url http://127.0.0.1:8090`를 추가한다.
`--camera-url`은 Orbbec 관측 서버를 지정한다. 관측 서버 자체는 계속 읽기 전용이며,
실행은 명시한 gateway에 기존 공통 실행기로 보낸다. 실제/가상 모드·보정 해시·현재 관절을 실행 전에 검사한다.
native 그리퍼의 최신 피드백도 요구하며, gateway에 실패/취소가 남아 있으면 상태를 확인하고 복구해야 한다.
`--gateway-url` 없이 실물 `--execute`를 사용할 수 없다. 실물 구동·파지는 현장 검증 대상이다.

실물의 현재 피드백은 제어 명령 완료를 확인하며 실제 물체 유지/방출을 증명하지 않는다.
따라서 `execution_completed_unverified`, `execution_success:true`, `success:null`로 기록한다.
실물 물체 성공을 확인한 것처럼 표시하지 않는다. 실제 낙하·파지 확인은 현장 검증 대상으로 남아 있다.

## 결과 확인

실행 결과는 지정한 출력 경로 아래에 저장된다.

| 파일 | 내용 |
|---|---|
| `task/<작업 ID>/task-result.json` | 성공/실패/미검증 상태, 실패 단계와 원인 |
| `task/<작업 ID>/observation.npz` | 인식 실패 전에도 저장하는 정확한 카메라/FK 입력 |
| `task/<작업 ID>/semantic-task.json` | 지시·검출 ROI·의미 추론 출처 |
| `task/<작업 ID>/gemini-error.json` | API 실패 코드·메시지·허용된 진단 헤더, 자동 재시도 0 |
| `task/<작업 ID>/plan.json` | 현재 관측에서 계산한 실행 계획 |
| `task/<작업 ID>/planning-result.json` | 계획 결과와 적용한 최대 접근각 |
| `task/<작업 ID>/empty-region.json` | 선정된 배치 영역 |
| `runtime/executions/<관측 ID>/execution.json` | 해당 작업의 집기·배치 평가(가상) |

interactive 모드는 작업별 하위 폴더를 자동 생성한다. 단발 모드는 지정한 출력 폴더에 직접 저장한다.
`--output`을 생략하면 `tiptop_outputs/m1013/tasks/task-001`부터 사용 가능한 번호를 자동으로 선택한다.
직접 지정한 출력 폴더에 이전 기록이 있으면 덮어쓰지 않고 다음 번호를 선택한다.
`sam3-test-02/task`는 `sam3-test-03/task`로 이어지며 실제 작업 경로를 `TASK_OUTPUT`으로 표시한다.
번호가 없는 이름은 `이름-001`처럼 접미사를 붙인다. 중단된 시도와 Gemini 호출 기록도 보존한다.
가상의 `runtime/execution.json`은 최신 결과의 호환 사본이다. 결과 조회는
`/result?observation_id=<관측 ID>`로 해당 작업을 지정해 이전 성공/실패와 섞이지 않게 한다.

가상 작업 결과의 `finished`, `success`, `inside_selected_region`,
`physical_placement_verified`가 모두 `true`인지 확인한다.
`class_id`와 `instance_id`가 지시한 대상인지 확인한다. `other_objects`는 나머지 물체별
최대 이동과 최종 위치를 기록하며, 어느 물체든 5mm를 넘게 이동하면 실행을 중단한다.
최종 놓인 방향과 초기 대비 원점 높이 차이는 성공 조건이 아니다. 누워 있어도 영역 내에 방출되고
책상 지지를 받으면 허용한다. 원점 높이 변화와 자세는 로그에 남긴다. 의도한 열기 전에 물체가
들린 뒤 손가락 접촉을 잃고 책상에 지속 접촉하면 낙하로 기록하고 중단한다.

## 지원 범위 및 제한 사항

- 장면에 요청 클래스가 없거나 같은 클래스가 여러 개이면 실패한다.
- 인식·계획 실패 후에는 다음 명시적 지시를 받는다. 다른 대상을 자동 선택하지 않는다.
- 가상 카메라는 이상적인 좌우 RGB 모델이다. 실물 336L의 IR·별도 컬러 정렬은
  [실물 준비 절차](m1013-hardware-preparation.md)를 따른다.
- 기본 관측 자세는 접근축이 수직 아래 대비 약15°인 자세다. 파지 후보의 기본 상한3°와는 별개다.
- 가상 카메라 장착·TCP와 시뮬레이션의 구동 상한을 실물 설정으로 사용하지 않는다.
- 동작 중 영상 추적·재계획은 지원하지 않는다. 장면이 바뀌면 새 지시에서 다시 관측한다.
- 이 가이드는 `run_known_pick_place` 경로다. 기존 `tiptop_run`·websocket과 구분한다.

## 같은 작업을 이어서 진단하기

`--active-task path/to/semantic-task.json`은 이전의 지시·라벨·ROI만 재사용한다.
현재 영상·SAM·깊이·CAD pose·계획은 새로 계산하며 Gemini를 호출하지 않는다.
지시가 달라지거나 대상이 ROI 밖으로 움직인 경우에 사용할 수 없다. 영상 추적 기능은 없다.
`--allow-gemini-once`와 동시에 지정할 수 없다.
