# TiPToP 프로젝트 메모리

갱신: 2026-11-25. 현재 상태·지속할 결정·미완료 작업만 유지한다.
기존 항목을 갱신하고 중복·대체된 내용을 제거한다. 대화 이력·PID·과거 staging·검사 수치를 계속 덧붙이지 않는다.
설정은 실제 코드/JSON, 커밋·staging은 Git을 다시 확인한다. 실행 중인 서비스가 있다고 가정하지 않는다.

## 사용자 선호와 작업 범위

- 커밋은 사용자가 한다. 커밋·브랜치·이력 변경은 명시 요청 범위에서만 수행한다. 부분 staging도 요청된 파일/범위만 처리한다.
- 커밋명은 기존 이력처럼 `[YYMMDD] feat/fix/refactor/chore/docs: 한글 설명`. 정확한 파일 목록을 제시하고 별도 커밋 분할안 파일은 만들지 않는다. `git add .`·`git add -A`를 권하지 않는다.
- 문서·답변은 짧고 구체적으로 작성한다. 현재 설정·실행 명령·입출력·확인 기준만 남기고 실험 과정·완료 보고·임시 출력 경로를 누적하지 않는다.
- README 수정은 요청 범위에서 수행하고 상세 사용법은 docs로 연결한다. 불필요한 일회성 코드·테스트·추적 문서·대규모 재설계를 추가하지 않는다.
- 실물 로봇·그리퍼 연결/상태 조회/제어권 획득/bringup/동작은 사용자 명시 허락 범위에서만 수행한다. 사용자 실험은 사용자가 진행한다.
- 서비스 기동·재시작을 임의로 하지 않는다. 설정 수정 때 Isaac을 켜서 보여줄 필요는 없고, 화면·가상 실행을 요청했을 때 실행한다.
- Gemini 신규 식별은 필요할 때만 최대 1회씩 사용하는 지속 승인이 있다. 자동 재시도·불필요한 호출은 하지 않는다. 실제 호출 시 CLI의 명시적 허용 요건도 유지한다.
- 외부 레포·NAS 원본·다른 프로젝트는 요청 없이 수정하지 않는다. 기존 관련 없는 변경을 보존하고, 삭제 전 잃는 기능·자료를 설명한다.
- 실패 시 핵심 로그·영향·추정 원인을 먼저 알린다. 같은 실험을 오래 반복하거나 범위를 임의 확대하지 않는다.

## 프로젝트와 환경

- 목표: M1013 + UON 3지 그리퍼 + Orbbec Gemini 336L로 현재 영상의 알려진 물체를 집어 고정 박스(또는 Gemini가 찾은 면)에 놓기. TiPToP 인식→계획→실행 구조를 장치 어댑터로 공유한다.
- 루트 `/home/uon/workspace/tiptop`. 본체는 Pixi, Isaac은 `/home/uon/isaac-sim/6.2.0/python.sh --no-ros-env`, FP·SAM3는 별도 conda 환경이다. Isaac 의존성을 본체 환경에 섞지 않는다.
- 서버 스크립트의 기본 conda 경로는 `~/miniforge3/envs/`; 실제 설치가 miniconda이면 `FOUNDATIONPOSE_PYTHON`·`SAM3_PYTHON`을 지정한다. 실행 상태·포트는 매번 확인한다.

| 서비스 | 포트/주소 |
|---|---|
| FoundationStereo | 13344, 선택 사용 |
| Isaac 가상 관측·실행 | 13345 |
| FoundationPose | 13346 |
| SAM3 | 13347 |
| 실물 Orbbec 관측 | 13348 |
| 실물 gateway / 뷰어 | 8090 / 8091 |
| SAM3 작업 worker | `/tmp/tiptop-<uid>/known-task.sock` |

- 실물 제어기 기본값: `192.168.1.30:12345`, namespace `dsr01`. 동일 로봇의 ROS 터미널은 같은 domain을 사용한다. Docker는 CLI 기준으로 안내한다.
- CAD·DB는 `tiptop/config/known_objects.json`: CAD `/nas/ochansol/3d_model/2026_real_world_test_objects`, DB `/nas/Dataset/Dataset_2026/isaacsim_grasp_data_gen`. OBJ 축척 0.001, DB 좌표계 object_zero_pose. 데이터 누락 시 로컬 사본으로 자동 대체하지 않는다.
- SAM3 reference root `/nas/dasolbyun/sanjabu/test_v2`, 실험실 scene `Lab/Uon_1F_Lab/table`. 서버의 기본 scene은 Home이므로 작업에 명시한다. CSV Class_name의 obj ID와 행 번호 No를 구분한다.
- data_gen의 `reference-gen`은 `reference/<물체명>/0000.png` 한 장만 저장한다. `--sources`에서만 같은 scene의 `reference_sources/<물체명>/`에 번호를 이어 여러 장 저장한다. `background`는 이 모드 전용 예약 이름으로 CSV 없이 선택하며 `reference_sources/background/0000.png` 한 장만 보존한다. YAML의 `directories.reference`는 `reference`로 유지한다.
- data_gen 합성은 `python scripts/compose_reference.py <scene>`로 reference_sources 전체를 처리한다(background 제외, --object-name 선택). 공통 background PNG가 필수다. **BBox가 있는 reference는 승인된 데이터로 보존**하고, 미승인은 다시 합성한다. 첫 결과는 `reference/<물체명>/0000.png`, 기존 이미지가 있으면 `0000_2.png`, `0000_3.png`처럼 비교본을 추가한다. 라벨링 도구는 첫 PNG를 읽으므로 채택한 비교본을 0000.png로 선택한 뒤 사용자가 직접 모든 박스를 그린다. 기존 라벨·index는 수정하지 않는다.
- 합성은 현재 배경의 닫힌 빨간 테두리 안을 자동 작업 영역으로 삼아 바깥 사람·케이블 변화를 제외한다. auto 촬영을 유지하며 밝기 보정·GrabCut 마스크는 내부 분리에만 사용한다. 모든 물체에 동일한 자동 합성 규칙을 적용하며 물체명별 예외나 매번 수동 튜닝을 요구하지 않는다. 원래 그림자 범위와 인형 신발 같은 어두운 부분을 보존한다. 배경 명암을 통째로 복사하지 않고, 그림자의 어두워지는 정도를 합성 위치의 배경에 적용해 밝은 반사 패치 전이를 막는다. 실제 마스크의 겹침과 간격을 검사하고 배치 순서를 자동 재탐색한다. 작은 분리된 배경 조각은 제외한다. 물체와 테이블 안 배경 변화의 구분이 어려우면 SAM을 검토한다. 자동 박스·마스크 파일·JSON·index 생성이나 ROI 선택 UI는 추가하지 않는다.

## 현재 운영 설정과 좌표

- 카메라 `tiptop/config/m1013_fixed_camera.declared.json`, 하드웨어 `tiptop/config/m1013_fixed_hardware.declared.json`, 그리퍼 `ros2_ws/src/execution/tiptop_execution/config/uon_native.measured.json`을 사용한다.
- declared/calibrated 플래그는 사용자가 채택한 운영 설정이며 정밀 실측 완료를 뜻하지 않는다. JSON 행렬을 기준으로 읽는다. 하드웨어 note의 예전 +45° 설명은 실제 −60° 행렬과 다르다. note 변경도 계획의 프로필 지문을 바꾸므로 무관한 수정에 섞지 않는다.
- 월드 기준은 ROS base_link, 단위 m/rad다. 실물 DR_BASE(0)와 대조하며 제어기 DR_WORLD(2)를 같은 프레임이라고 가정하지 않는다. GetCurrentPosj는 degree다.
- 몸체 장착 Z 회전 −90°, body→TCP Z +0.2368508m, spinner 원점 Z +0.1m/회전 180°, 손가락 각도는 spinner의 −2배다.
- DB→TCP 보정은 **−60°(=300°), 이동 0**. `world_cad @ cad_db @ grasp_mat @ db_grasp_from_tcp`로 목표 TCP를 만든다. DB 원본이나 모델 TCP 높이를 DB 행렬에 다시 더하지 않는다.
- 180° 손가락 모델 보정과 −60° TCP 보정을 합치면 120°여서 기존 DB 손가락 배치와 맞는다. 60°·180°·300°는 같은 손가락 배치를 만들지만 몸체·케이블 방향은 다르다.
- 300° 선택 근거: 11월 3일 세 물체의 오프라인 접근 계획 비교 후 사용자가 “실제 실험하면서 정하고 일단 300으로” 채택했다. 항상 최소 회전하는 값이나 자동 최적화 결과가 아니다. 시험했던 Z −45.8mm 보정은 제거됐다.
- 카메라 본체 전면 중앙 `(-0.01,0.76,1.02)m`, 수직 하향, RGB·Depth 180° 회전. 몸체 전면 위치와 SDK 오프셋을 포함한 컬러 광학 원점을 구분한다.
- 상부 프레임 아랫면 1.175m, 카메라 전면까지 간격 0.155m. 책상 X 0.78m/Y 2.5m/두께 0.03m, 상판 Z=0. Right 작업 ROI `[[-0.365,0.470],[0.365,1.225]]m`; Left keepout 유지, Center는 작업 후보만 제외하고 팔 통행은 허용한다.
- 열림 3400/닫힘 0(11-18 변경), 개폐 전류 상한 50, 모터 전류 상한 300, 접촉 추정 문턱 40, 위치 허용 오차 10 count. 피드백 상한 6331과 드라이버 명령 상한 3600은 다른 값이다. 물체별 닫기 전류는 `tiptop/config/grasp_params.json`의 `effort_counts`이며 [그리퍼 설정](docs/m1013-gripper-check.md#물체별-닫기-전류)을 따른다.
- 모터 위치·전류 count를 m·N이나 모델 각도로 해석하지 않는다. count→각도/개구폭 대응은 미보정이며 빈 끝점·기구 걸림도 전류를 발생시킨다. 개폐 완료가 물체 파지 성공은 아니다. 빈 손 닫힘은 175~315 count에서 멈추며, 뷰어의 `파지 성공/실패`는 닫은 위치 315 count 이하 여부로 추정한 값이다.

## 실행·계획·뷰어의 유지할 동작

- 진입점 `tiptop/scripts/run_known_pick_place.py`. SAM3 reference는 class ID 지시, Gemini는 자연어 지시를 사용한다.
- 작업 프리셋은 `fixed_box`(기본, UI 미표시; 목적지는 `tiptop/config/fixed_box.json`의 고정 박스, SAM3·Gemini)와 `perceived_surface`(Gemini가 찾은 면, Gemini만) 둘이다. 기본 지시문은 둘 다 `<물체> 집고 박스에 놓아`다. `home_drop`(초기 위치에서 박스에 넣기)은 2026-11-25 SAM3·Gemini 모두에서 제거했다. Gemini는 `fixed_box`에서 `<물체> 집어`만 받고 박스 위치를 모른다.
- 순서: 접근+하강(이어 붙임) → 닫기 → 상승+운반(이어 붙임) → 놓을 자세의 2cm 앞에서 열기 → 홈 복귀. 이어 붙임은 블렌드 반경 3cm의 5차 다항식이며 한계나 충돌 검사를 넘으면 두 구간으로 둔다. 요청 `release_height_m`은 0이고 허용 하한은 1.5cm 아래다. 두 프리셋 모두 `place_task.run_standard_place`를 쓴다.
- CLI 기본값(접근각 3°, 테이블 여유 0.01m, 입자 32)과 운영 UI 기본값(접근각 25°, 테이블 여유 0.005m, 입자 8, 데모 모드 켜짐)을 구분한다. 공통: IoU 0.75, 파지 깊이 0.02m, J4/J5 변화량 90°, table-support=declared. 설정은 `tiptop/config/launcher_presets.json`에서 확인한다.
- 모든 구간 속도 배율 1.0(`SPEED_FACTOR`). 가속도·jerk·종단 속도 허용치는 `tiptop/config/robot_limits.json`(J1~J5 8.0/256, J6 12.0/384, 종단 0.025)이며 코드 기본값(5.0/160, J6 7.0/224, 0.015)보다 우선한다. 경로 위치·속도·시간을 그대로 전달하고 gateway 최소 10초 조정·추가 0.1초 종점 유지 샘플은 없다.
- `--max-wrist-travel-deg`는 J4/J5만 검사한다. J6 케이블 범위 제한·60/180/300° 자동 선택·이동 시간 최적화는 미구현이다. 초기 자세를 케이블 중립각으로 매번 재정의하지 않는다.
- UI `주변 물체 장애물 반영`과 CLI `--surrounding-obstacles`는 기본 ON이며 SAM3·Gemini 모두 같은 필터다(SAM3에서는 `object` reference 호출 여부도 정한다). `--no-surrounding-obstacles`는 선택된 파지 대상(과 인식한 목적지)만 물체 장애물로 등록하고, 인식 마스크·후보 순위·테이블·고정 구조물·자기충돌 검사는 유지한다. 3D Obstacles는 해당 실행에서 등록한 물체만 표시한다. 열린 그리퍼 충돌 구의 테이블 간격과 TCP 파지 깊이는 별개다.
- CAD 통과 후보 최대 48개에서 IK 성공 후보를 점수 순으로 `--num-particles`개까지 입자에 하나씩 배정한다(cuTAMP 패치 `ordered_grasps`). 유효 후보를 첫 완성 경로까지 검사하며, 최종 후보가 단순 최고 DB 점수는 아니다. 실패했다고 제약을 자동 완화하지 않는다.
- SAM3 단발 작업은 선택적 `serve_known_task_worker`/`--worker-socket`으로 초기화를 재사용한다. 매번 새 관측·인식·계획을 수행하고 동시 요청은 거절한다. Gemini·interactive는 worker 미지원이다.
- 같은 작업의 선별 IK를 조건 확인 후 재사용한다. 작업 간 계획 객체는 모델 키 확인과 world/mesh/seed 초기화 후 재사용하며, 첫 완성 경로 시도 실패 시 일반 IK/기존 최적화로 복귀한다.
- PNG/NPZ는 무손실 level 1, 관측 원본 패킷을 저장한다. planning-input.pkl은 입력 복사 후 백그라운드 저장하고 실패·취소에도 drain한다. timings.json의 겹치는 세부 시간을 중복 합산하지 않는다.
- 작업 출력은 `tiptop_outputs/m1013/tasks/task-NNN` 자동 증가. 기존 결과·실패·Gemini 호출 마커를 덮어쓰지 않는다. TASK_OUTPUT/SESSION_OUTPUT의 실제 경로를 사용한다.
- 뷰어 기본은 완성 계획 Pick.descend 끝점 FK인 최종 파지다. DB 후보 조회는 별도 보관하고, 계획이 없을 때 후보를 최종 파지로 대체하지 않는다. 실측 도달 자세·연속 영상이 아니다.
- DB 범위 표시는 DB→TCP 보정을 역변환해 놓고 임의 회전/이동으로 CAD에 맞추지 않는다. 새 plan.json의 db_grasp_from_tcp를 저장 뷰어에서 우선 사용한다. 측면은 Right 끝에서 로봇을 보는 방향(화면 오른쪽 −X, 위 +Z)이다.
- 뷰어 서비스 칩은 두 줄이다(1줄: ros2-dev·FoundationPose·SAM3·작업 worker, 2줄: bringup·컨트롤러·그리퍼·gateway·관측 서버). `launcher_presets.json`의 `only_in`으로 SAM3 서버·worker는 Gemini 모드에서도 보이되 `SAM3 모드에서만 사용` 상자로 묶고 SAM3 모드에서는 상자가 없다. ▶/■는 `bash -ic`로 실행해 `~/.bashrc`의 `fp`·`sam` alias를 쓰며 `rosdev`는 쓰지 않는다. 종료는 시작의 역순이고 관측 서버 → gateway → bringup·그리퍼 → ros2-dev 순서만 지키면 된다. 뷰어 하단 안내 문구는 제거했다(OVERLAY 패널이 맨 아래). 코드·프리셋 JSON을 바꾸면 뷰어 서버와 작업 worker를 재시작한다.
- UI 실행 버튼은 --execute를 포함한다. 계획만 볼 때는 CLI, 표시만 허용하려면 뷰어 --read-only를 사용한다. UI 중단은 스크립트 Ctrl+C이며 로봇 비상정지가 아니다.
- 실물 실행 전 그리퍼 열림을 확인한다. 실패 시 그리퍼 유지·경로 취소 실패와 무관하게 팔 정지를 시도하며, 성공 응답이 확인된 중복 정지만 생략한다. 정지 응답은 물리적 정지 증명이 아니다.
- `execution_success:true`, `success:null`은 실물 제어 완료다. DB 점수·가상 접촉·RViz 부착·SDK 명령 수락을 실물 파지 성공으로 확대하지 않는다. 동작 중 영상 추적/재계획·자동 실패 복구는 없다.

## 가상 환경과 외부 저장소

- 현재 고정 카메라 환경은 `live_perception.py --lab`이다. 실물 하드웨어 설정을 직접 읽고 USD의 광학 자세·책상·구조물 15개·ROI를 대조하며 불일치 시 거부한다. 별도 wrist 예제는 유지한다.
- 예전 `config/fixed_camera.json`·lab_box_review 래퍼·lab_table_draft/view·preview_lab_box는 제거됐다. 기본 `lab_table_gripper_view.usda → keepout → camera → camera_optics` 4개 레이어는 역할별 참조를 유지한다.
- live --lab은 외부 USD 원본을 바꾸지 않고 손가락/관절 기준에 URDF의 180° 보정을 적용한다. USD를 직접 열거나 m1013_uon.py 단독 경로가 동일하다고 가정하지 않는다.
- 홈은 `config/lab_home.json`의 10월 29일 저장 관절값, 물체는 lab_objects.json의 예제 3개다. 현재 실물 자세·물체 배치와 자동 동기화되지 않는다.
- 가상 RGB/Depth는 저장 SDK 광학·정렬·180° 회전을 적용한 이상적인 기하 렌더다. 실물은 현재 sensor 깊이 경로를 사용하며 FS는 선택 경로다. 서버를 끈다고 깊이 소스를 자동 전환하지 않는다.
- 가상 링크 중력 off·그리퍼 5Nm·PhysX 마찰/접촉/자기충돌은 실제 제어기·모터 count/전류와 다르다. 계획용 카메라는 여유 2mm 상자 2개, 물리 카메라는 본체+브래킷 3부분이다.
- Doosan USD의 단위 메타데이터(.01)만 보고 숫자를 재축척하지 않는다. 실제 관절/메시는 m 기반 URDF와 맞으며 Metrics Assembler 자동 변환을 끈다. 사용자 armature 값을 덮어쓰지 않는다.
- 시점은 Save current view로 lab_pick_view.usda에 저장한다. --view는 시점만 복원한다. 일반 Save As가 물리 핸들을 무효화하면 runtime을 재시작한다.
- FoundationPose·SAM3는 data_gen도 사용하는 공유 서브모듈이다. 변경 전 그쪽 README·호출 경로와 업데이트 영향을 확인한다. Utils.py의 직경 최적화는 공용 함수 안에 두고 서버 모듈에 의존시키지 않는다. 수치·dtype·난수 동작 호환은 검증했다.
- UON 외부 드라이버는 ros2_ws의 `patches/uon_3finger_gripper.patch`·`scripts/apply_gripper_patch.sh`로 관리하고 ros2.repos의 고정 revision과 맞춘다. 외부 원격 권한을 전제로 하지 않는다.
- 두산 최소 명령 변화량은 코드 상수 `1e-9 rad`, 시작 로그로 확인한다. DSR_MIN_COMMAND_CHANGE_RAD/DSR_TRACE_FILE 환경변수와 RT CSV 기능은 제거됐다. 전송 문턱은 최종 관절 오차 허용값과 다르다.

## 현재 정리 상태와 다음 작업

- 확인한 루트 HEAD는 `3b800ed`(home_drop 제거, 뷰어 화면 개선, 2026-11-25). 이후 상태는 git status/log로 재확인한다.
- 확인한 gitlink: FoundationPose `24964c3`, SAM3 `29c8db3`, ros2_ws `80a6cb8`(이후 변경은 미확인). ros2_ws/src/drivers는 vcs로 가져온 별도 외부 저장소이며 루트 변경에 통째로 추가하지 않는다. 그리퍼의 패치 적용 상태는 정상 확인했다.
- 문서(README_CUSTOM·`docs/`의 M1013 문서 15개)는 2026-11-25에 현재 코드 기준으로 다시 정리했고, 신규는 perception-viewer·m1013-real-connection·m1013-gripper-check·m1013-robot-limits·m1013-arm-trial이다. 이 정리는 사용자가 staging했고 커밋은 사용자가 한다. MEMORY 자체를 포함한 추가 staging/commit은 요청 없이 하지 않는다.
- 유지·커밋된 도구: 지지면 비교(experiments perception table-support), 실물 RGB 확인(preview_orbbec_rgb), 가상 광학 확인(preview_lab_camera). 예전 RT CSV 분석기 analyze_m1013_trace와 중복 재생 도구는 현재 없다.
- 미완료 정리: 로컬 .vscode/·MEMORY.md·외부 FoundationStereo/·저장 시점 lab_pick_view.usda의 커밋 제외 방식. 삭제/이동/ignore 반영이 완료됐다고 가정하지 않는다.
- FS 실행 스크립트의 추론 후 CUDA 캐시 해제 변경은 미커밋 상태다. 구문·반환/예외 전달은 확인했지만 실제 GPU 메모리/지연 효과는 미측정이다.
- 검증 범위: 단위 테스트 670개 통과(1개 skip, `integration` 제외)와 문서 링크·앵커 검사. 뷰어 HTML은 headless DOM으로만 확인했고 실제 화면 렌더링·Isaac 접촉 파지·실물 동작은 이번 정리에서 검증하지 않았다. pytest는 ROS를 source한 셸에서 `env -u PYTHONPATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH -u LD_LIBRARY_PATH pixi run python -m pytest`로 실행한다.
- 깊이 정확도 비교·정밀 카메라 외부 보정은 사용자 요청으로 보류다. 재개 시 같은 RGB/IR/센서 깊이·마스크·독립 실측값으로 비교한다. 서로 다른 관측의 CAD 자세나 깊이를 정답으로 삼지 않는다.
- 남은 과제: 케이블 중립각/허용 범위와 후보 선택, count↔손가락 각도 및 접촉 판정 보정, 실물·가상 전체 동작 검증. 카메라 공통 회전/검증을 Doosan 모듈에서 분리하는 작업은 검토만 했으며 미구현이다.

## 근거 자료와 문서

- 실행은 docs/known-object-pick-place.md, 가상은 docs/m1013-fixed-camera.md, 실물은 docs/m1013-real-connection.md·m1013-hardware-preparation.md, 그리퍼는 docs/m1013-gripper-check.md, 로봇 한계는 docs/m1013-robot-limits.md, 뷰어·서비스 칩은 docs/perception-viewer.md, 진단은 experiments/README.md를 따른다.
- 과거 상세 자료는 `tiptop_outputs/m1013/model-adjustments/`, `candidate-policy-review-20261104/`, `latency-runtime-20261103/`, `planning-input-save-20261104/`, `ros2_ws/diagnostics/`에 있다. 현재 설정이나 실행 상태로 취급하지 않는다.
- 옛 실험 원자료는 `tiptop_outputs/experiments/grasping/20261020/`·`diagnostics/`에 보존됐다. /tmp의 오래된 실행기·압축 소스를 본체 대신 사용하지 않는다. Git 커밋만으로 NAS·출력 자료가 보존되지는 않는다.
