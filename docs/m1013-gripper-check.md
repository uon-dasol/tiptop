# M1013 그리퍼 설정

## 좌표와 모델

| 항목 | 값 |
|---|---|
| 플랜지 → 몸체 | 이동 0, Z 회전 −90° |
| 몸체 → spinner | Z +0.1 m, 기준 회전 180° |
| 몸체 → TCP | Z +0.2368508 m, 회전 0° |
| DB grasp → TCP | Z 회전 −60°(=300°), 이동 0 |
| 손가락 연동 | 손가락 각도 = −2 × spinner 각도 |

실물 운영값은 `tiptop/config/m1013_fixed_hardware.declared.json`, ROS 장착 모델은 `ros2_ws/src/description/m1013_uon_3f_description/config/mount.yaml`에서 관리한다.

```text
world_from_tcp = world_from_cad × cad_from_db × grasp_mat × db_grasp_from_tcp
```

- DB 원본은 바꾸지 않고 TCP 목표를 계산할 때 보정한다. TCP 높이를 DB 행렬에 다시 더하지 않는다.
- 손가락은 120° 간격이라 `180° − 60° = 120°`로 DB 손가락 배치와 맞는다. 60°·180°·300°는 손가락 배치가 같지만 몸체·케이블 방향이 다르며, 최소 회전 자동 선택은 없다.
- 이 값들은 정밀 실측 보정이 끝난 값이 아니다.

## 실물 개폐

`ros2_ws/src/execution/tiptop_execution/config/uon_native.measured.json`을 쓴다.

| 항목 | 값 |
|---|---:|
| 열림·닫힘 목표 | 3400 / 0 count |
| 열기·닫기 전류 상한 | 50 / 50 count |
| 접촉 추정 전류 문턱 | 40 count |
| 위치 허용 오차 | 10 count |
| 모터 전류 상한 | 300 count |

- 닫기 목표를 0(기계적 끝보다 아래)으로 두어, 물체가 어디서 멈춰도 목표와의 거리가 커서 전류가 닫기 전류 상한까지 오른다.
- 빈 손으로 닫으면 닫기 전류에 따라 175~315 count에서 멈춘다 (상한 2에서 315, 5에서 299, 10에서 219, 50에서 약 200, 300에서 175).
- 위치·전류 값을 바꾸면 gateway를 재시작한다.

### 물체별 닫기 전류

`tiptop/config/grasp_params.json`의 `effort_counts`(`default`와 `class_id`별)가 닫기 전류 상한이다.
값은 1~300의 정수이며 계획 시 `plan.json`의 `Pick.close` 단계에 들어가 gateway가 그 값으로 닫는다.
접촉 판정 문턱은 그 값에 프로파일 비율(40/50)을 곱해 따라가고, 정지할 때는 마지막 닫기에 쓴 값으로 위치를 유지한다.

열기·닫기 결과(`physical-execution.json`)에는 `trace`로 최대 40개의 `[경과 초, 위치 count, 전류 count]` 샘플이 들어간다. 정지 위치와 전류 문턱을 정하는 근거로 쓴다.

## 해석 주의

- `GripperCommand`의 position·max_effort는 native 위치·전류 count다. m·N 단위가 아니며 count → 관절각 대응은 보정되지 않았다.
- 전류·정지 여부만으로 물체 접촉과 기구 걸림을 구분할 수 없다.
- RViz의 개폐 표시는 실제 모터 피드백이나 파지 성공을 증명하지 않는다.
- 고정 카메라의 로봇 깊이 제거는 관측 관절값과 모델 충돌 구를 쓰므로 장착·보정 오차의 영향을 받는다.
