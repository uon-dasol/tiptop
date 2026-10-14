# M1013 계획 경로 실행

팔 드라이버·게이트웨이·통합 모델 준비는 [실행 안내](m1013-execution.md)를 따른다.

## 이번에 확인하는 것

```text
현재 관절값 조회 → cuRobo 경로 계산 → 계획 파일 저장
  → 기존 run 명령 → ROS2 → 가상 M1013
```

`demo`는 지정 경로, `plan`은 M1013 모델로 계산한 경로를 사용한다.
이번 명령은 열린 그리퍼 형상과 장애물 없는 시험 공간을 가정한다.
카메라·실제 작업대·물체 파지·cuTAMP 전체 작업 계획은 포함하지 않는다.

## 1. 계획 생성

호스트의 최상위 TiPToP 폴더에서 실행한다.

```bash
pixi run python -m tiptop.doosan plan
```

현재 가상 팔 관절값을 읽고, 아래 왕복 동작의 두 팔 경로를 모두 계산한다.

```text
그리퍼 닫기 → 열기 → 여러 관절 이동 → 닫기 → 열기 → 출발 자세로 복귀
```

이동 목표는 출발 자세에서 1~6번 관절 각각 **+20°, -20°, +20°, +15°, -15°, +25°**다.
그리퍼는 팔이 멈춘 상태에서 개폐하며, 팔은 열린 그리퍼 형상으로 계획하므로 이동 전에 다시 연다.
`success: true`, `sent_to_robot: false`, `plan_file`을 확인한다.
`steps: 6`, `returns_to_start: true`가 표시된다.
`waypoints`와 `requested_duration_s`는 왕복 경로점 수와 팔 실행 요청 시간의 합이다.
그리퍼 개폐 시간은 별도로 더 걸린다.

계획 실패 시 이전 JSON이 남아 있을 수 있으므로, `success: true`를 확인한 뒤 실행한다.
현재 충돌 모델과 외부 패치의 범위는 [충돌 검사 안내](m1013-collision.md)를 참고한다.

## 2. 생성한 경로 실행

계획 생성 이후 팔을 다른 명령으로 움직이지 않은 상태에서 실행한다.

```bash
pixi run python -m tiptop.doosan run tiptop_outputs/m1013/showcase.json
```

RViz에서 그리퍼 두 번 개폐, 팔의 큰 이동과 출발 자세 복귀를 확인한다. 최종 `completed_steps: 6`을 확인한다.
팔 완료 로그의 `motion_status: 0`과 `max_joint_error_rad`도 확인한다.

## 계획 JSON은 무엇인가

JSON에는 출발 관절값(`q_init`, rad), 순서대로 실행할 동작(`steps`), 각 경로의
관절 위치(`positions`, rad)·속도(`velocities`, rad/s)·간격(`dt`, 초)이 들어 있다.

중단 명령과 완료 기준은 [실행 안내](m1013-execution.md#중단종료)를 따른다.
