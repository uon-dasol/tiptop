# M1013 알려진 물체 집기·박스에 놓기

현재 영상에서 등록된 물체를 인식하고, 파지 DB로 경로를 계획해 집어서 목적지(고정 박스 또는 Gemini가 찾은 면)에 놓는다.
가상 환경은 [고정 카메라](m1013-fixed-camera.md), 실물은 [실물 준비](m1013-hardware-preparation.md)를 따른다.

## 물체 데이터

`tiptop/config/known_objects.json`에 물체와 경로를 등록한다. 한 작업에서 집는 인스턴스는 하나다.

| 데이터 | 경로 |
|---|---|
| CAD | `<cad_root>/<class>/edited/<class>.obj` (mm → m 변환) |
| 파지 DB | `<db_root>/<class>/UON_3finger_gripper/<class>_merged_grasp_zero_pose.json` |

- `cad_root`·`db_root`는 `/nas/ochansol/3d_model/2026_real_world_test_objects`, `/nas/Dataset/Dataset_2026/isaacsim_grasp_data_gen`이다.
- `additional_datasets`의 폴더(현재 `isaacsim_grasp_data_gen2`)는 폴더 이름을 클래스로 쓰며, 기존 등록 물체와 겹치면 기존 경로가 우선한다.
- 자산이 없으면 다른 데이터로 대체하지 않고 중단한다.
- 행의 `grasp_mat`은 object zero pose 기준 4×4 파지 자세다. `single_object_grasp_compact` DB는 이 위치가 파지 원점이고, `dataset_v2_object_grasp_compact` DB는 베이스 자세라 로더가 `target_points`(로컬 +Z 237 mm)로 파지 원점을 맞춘 뒤 DB→TCP 보정을 적용한다.
- `approach_vector`는 `grasp_mat`의 세 번째 열(로컬 Z)과 같다.
- `target_orientation`은 오일러각(°, scipy `xyz`)이며 `grasp_mat`의 회전에 로컬 X축 180°를 곱한 값이다. 파지 자세는 `grasp_mat`만 쓴다.
- `target_width`는 `0.058`·`0.087`·`0.1015`·`0.116`·`0.145` m 다섯 값 중 하나라 측정된 닫힘 폭이 아니다. 계획에는 쓰지 않고 선택한 행 요약(`selected_grasp.db`)에만 남긴다.
- `score`는 후보를 정렬하는 시뮬레이션 점수이며 선택한 행 요약(`selected_grasp.db`)에도 남는다. 성공 확률로 해석하지 않으며 파지 라벨은 모두 시뮬레이션에서 나왔다.
- 물체 파일·로더를 바꾸면 FoundationPose 서버와 상주 worker를 재시작한다.

물체별 닫기 전류는 [그리퍼 설정](m1013-gripper-check.md#물체별-닫기-전류)을 따른다.

## 실행

SAM3·FoundationPose 서버(설치: [README_CUSTOM](../README_CUSTOM.md))와 카메라 쪽(가상 Isaac 또는 실물 관측 서버)을 먼저 띄운다.
`--lab`과 센서 깊이 경로에는 FoundationStereo 서버가 필요 없다.

```bash
pixi run python -m tiptop.scripts.run_known_pick_place \
  --camera-url http://127.0.0.1:13345 \
  --perception-mode sam3_reference --reference-scene Lab/Uon_1F_Lab/table \
  --task 'car_leather_conditioner_bottle 집고 박스에 놓아' \
  --table-support declared --viewer
```

| 옵션 | 동작 |
|---|---|
| (기본) | 인식·계획만 하고 결과를 저장 |
| `--check` | 연결·reference·자산만 검사. 인식·동작 없음. 실물 준비까지 보려면 `--check --execute`와 실물 옵션을 함께 지정 |
| `--execute` | 계획한 경로를 실행 |
| `--interactive` | 결과 후 다음 지시를 계속 입력 |
| `--perception-mode gemini_er --allow-gemini-once` | Gemini 사용. `--reference-scene`은 뺀다. 지시마다 최대 1회 호출, 인증키 필요 |

- SAM3 모드의 지시문은 `<class_id> 집고 박스에 놓아` 형식이다. UI는 한글 이름을 class_id로 바꿔 준다.
- 실물은 `--camera-url http://127.0.0.1:13348 --hardware-profile tiptop/config/m1013_fixed_hardware.declared.json --gateway-url http://127.0.0.1:8090`을 추가한다.
- 첫 팔 이동 전에 그리퍼 열림을 확인한다. 실패·취소 뒤 자동 복구나 재실행은 없다.
- `execution_success:true`, `success:null`은 제어 명령 완료다. 실제 파지 성공은 따로 확인한다.
- 동작 중 영상 추적·재계획은 없다. 물체가 움직이면 다시 관측한다.

## 작업 프리셋

`--task-preset`으로 놓을 곳을 고른다. 기본 `fixed_box`이며 UI에는 표시하지 않는다. 두 프리셋 모두 접근 → 닫기 → 상승·운반 → 열기 → 홈 순서이고 같은 계획기를 쓴다.

| 프리셋 | 목적지 | 지시문 | 인식 |
|---|---|---|---|
| `fixed_box` (기본) | 설정에 고정된 박스(`tiptop/config/fixed_box.json`) 윗면 | `<물체> 집고 박스에 놓아` (`집고` 뒤 문구는 무시, 인식은 물체만) | SAM3·Gemini |
| `perceived_surface` | Gemini가 영상에서 찾은 면 | `<물체> 집고 박스에 놓아` (목적지까지 Gemini가 해석) | Gemini만 |

- 박스는 하나의 직육면체로 충돌 모델에 들어가고, cuTAMP가 놓을 자리와 경로를 계획한다. `perceived_surface`에서도 고정 박스는 장애물로 등록된다.
- 접근+하강, 상승+운반을 각각 이어 붙여 접점에서 멈추지 않는다. 이어 붙인 경로가 한계를 넘거나 충돌 검사를 통과하지 못하면 두 구간으로 둔다.
- 운반은 놓을 자세의 2 cm 앞(TCP 접근축 방향)에서 끝나고 그 자리에서 연다. 요청 높이(`release_height_m`)는 둘 다 0이며, 실제로는 CAD 바닥이 목적지 윗면 약 2 cm 위에서 열린다. 1.5 cm까지 더 낮아도 허용하고 더 높은 건 허용한다.
- 놓은 물체의 CAD 바닥 높이와 박스 이탈량은 `plan.json`의 `placement_check`에 남는다.

## 인식 모드 차이

물체를 인식하는 단계만 다르고 포즈 추정(FoundationPose), 파지 DB, 계획, 실행은 같다.

| | SAM3 (`sam3_reference`) | Gemini (`gemini_er`) |
|---|---|---|
| 마스크 | SAM3가 reference로 직접 구함 | Gemini bbox를 SAM2가 분할 |
| 후보 검증 | 점수 순 최대 8개를 FoundationPose·CAD IoU로 검증 | bbox 하나를 유일한 후보로 검증 |
| 주변 물체 | `object` reference로 SAM3를 한 번 더 호출 | 같은 호출에서 Gemini가 찾은 다른 물체 |
| 지시문 | `<class_id> 집고 박스에 놓아` | 자연어 |
| 상주 worker | 사용 가능 | 사용 불가 |

**Gemini 프롬프트**는 지시문을 끼운 기본 템플릿(`tiptop/perception/prompts/detect_and_translate.txt`)에 물체 카탈로그(class_id·한글 이름·alias·외형)를 덧붙인 것이다. 현재 위치·시뮬레이터 상태는 넣지 않으며, 고정 박스 위치도 알려 주지 않는다.

| 프리셋 | Gemini가 받는 지시문 | 돌려받는 것 |
|---|---|---|
| `fixed_box` | `집고` 뒤를 잘라낸 `<물체> 집어` | 물체 bbox, `holding(물체)`. 목표는 코드가 `holding`으로 다시 정하고 박스는 `fixed_box.json`에서 계획 단계에 등록 |
| `perceived_surface` | 입력한 전체 문장 + 목적지 규칙 | 물체·목적지 bbox, `on(물체, 목적지)`. 목적지가 박스·평면이 아니면 놓을 수 있는 부분(예: 그릇 안쪽 `<목적지>_inside`)만 하나의 bbox로 잡음 |

**주변 물체 장애물 반영**(`--surrounding-obstacles`, 기본 켜짐)은 두 모드 모두 같은 필터다. 켜면 인식한 모든 물체를 충돌 회피 대상으로 등록하고, 끄면 파지 대상(과 인식한 목적지)만 등록해 나머지는 계획에서 투명해진다. 테이블·고정 구조물·고정 박스·자기충돌 검사는 끄더라도 유지한다. SAM3에서는 이 옵션이 `object` 호출 여부도 정한다.

## 주요 옵션

기본값은 CLI 기준이다. UI 기본값은 `tiptop/config/launcher_presets.json`에 따로 있다. 전체 인자는 `--help`로 확인한다.

| 옵션 | 기본값·의미 |
|---|---|
| `--max-approach-tilt-deg` | 3: 수직 하향 접근축에서 허용할 기울기 |
| `--cad-mask-min-iou` | 0.75: 추정 포즈의 CAD 실루엣과 검출 마스크의 IoU 하한. 실루엣 밖의 그림자 픽셀은 제외하고 계산 |
| `--min-table-clearance-m` | 0.01: 열린 그리퍼 충돌 구와 지지면 사이 최소 간격 |
| `--min-engagement-depth-m` | 0.02: 손끝이 물체 윗면 아래로 들어가는 최소 깊이 |
| `--max-wrist-travel-deg` | 90: 구간 시작 대비 J4·J5 최대 변화량 (J6 제한 아님) |
| `--footprint-overhang-tolerance-mm` | 25: `fixed_box`에서 놓은 물체가 박스 윗면 밖으로 나가도 되는 최대치 |
| `--num-particles` | 32: cuTAMP에 넘기는 IK 통과 파지 수 상한. 점수 순으로 하나씩 입자에 배정 |
| `--table-support` | `ransac-cad`(기본), `ransac-mask`, `declared`(설정의 책상 윗면). `ransac-mask`·`declared`는 CAD 바닥과 지지면 차이가 30 mm를 넘으면 거부 |
| `--surrounding-obstacles` | 기본 켜짐. `--no-surrounding-obstacles`는 파지 대상만 장애물로 등록 (테이블·고정 구조물·자기충돌 검사는 유지) |
| `--minimal-records` | 데모 모드. 카메라 패킷·인식 이미지·후보 크롭·계획 입력 pickle을 저장하지 않음. 계획·실행·시간 기록과 뷰어는 유지 |
| `--worker-socket` | 상주 worker로 실행 (아래) |

- 로봇 충돌 허용치 0.01 m, 개별 모션 탐색 1초는 옵션이 없는 고정값이다.
- 가속도·jerk·종단 속도 허용치는 [로봇 한계](m1013-robot-limits.md)에서 정한다. 모든 구간의 속도 배율은 1.0이다.
- 후보가 없어도 제한값을 자동으로 완화하지 않는다.
- 지지면 방식 비교는 [실험 도구](../experiments/README.md#책상-지지면-방식-비교)를 참고한다.

## 상주 worker

단발 SAM3 명령 사이에 모델·CAD·DB·계획 객체를 유지한다.

```bash
pixi run python -m tiptop.scripts.serve_known_task_worker --preload-objects all
```

작업 명령에 `--worker-socket "/tmp/tiptop-$(id -u)/known-task.sock"`을 추가한다.
worker는 한 번에 한 작업만 받고 매번 새로 관측·인식·계획한다. 코드를 바꾸면 재시작한다. Gemini·`--interactive`는 worker를 쓰지 않는다.
FoundationPose 서버는 `--preload-objects all`로 자산을 미리 준비한다. 작업 명령의 `--preload-objects all --prepare-only`는 준비만 하고 관측·실행을 하지 않는다.

## 결과

출력은 `tiptop_outputs/m1013/tasks/task-NNN`에 자동 증가하며 기존 기록을 덮어쓰지 않는다.

| 파일 | 내용 |
|---|---|
| `task-result.json` | 상태·실패 단계·오류 |
| `observation.npz`, `semantic-task.json` | 관측, 대상 인식 |
| `plan.json`, `planning-result.json` | 경로, 적용한 계획 설정, 선택한 DB 행(`selected_grasp.db`), 장애물 등록 내역 |
| `physical-execution.json` | 실물 명령별 완료·정지 응답 |
| `timings.json` | 인식·계획·실행 시간 |

가상 실행 결과는 runtime의 `executions/<관측 ID>/execution.json`에 있다. 화면 확인은 [뷰어](perception-viewer.md)를 사용한다.
