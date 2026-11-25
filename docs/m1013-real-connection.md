# M1013 실물 기동

설치·빌드는 [ROS 워크스페이스 안내](../ros2_ws/README_CUSTOM.md)를 따른다.
ROS 명령은 `ros2-dev` 컨테이너에서 실행하고, 같은 로봇의 노드는 같은 ROS domain을 쓴다.
Bringup은 제어권·서보·RT 통신을 설정하므로 단순 상태 조회가 아니다. 기존 드라이버와 중복 실행하지 않는다.

[뷰어](perception-viewer.md#서비스-시작종료)의 서비스 칩으로도 아래 순서대로 시작·종료할 수 있다.

## 팔과 컨트롤러

```bash
ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py \
  name:=dsr01 mode:=real host:=192.168.1.30 rt_host:=192.168.1.30 \
  port:=12345 model:=m1013 gui:=false
```

별도 ROS 터미널에서 컨트롤러를 확인하고, 비활성이면 시작한다.

```bash
ros2 control list_controllers -c /dsr01/controller_manager
# dsr_moveit_controller가 active가 아닐 때만 실행
ros2 run controller_manager spawner dsr_moveit_controller -c /dsr01/controller_manager
```

상태 확인 (`robot_system=0`이 실물):

```bash
ros2 service call /dsr01/dsr_controller2/system/get_robot_system dsr_msgs2/srv/GetRobotSystem '{}'
ros2 service call /dsr01/dsr_controller2/aux_control/get_current_posj dsr_msgs2/srv/GetCurrentPosj '{}'
```

`GetCurrentPosj`는 degree, ROS JointState·TiPToP 관절값은 rad다.

## 그리퍼와 gateway

각각 별도 ROS 터미널에서 실행한다. 그리퍼 드라이버는 워크스페이스의 패치를 적용한 빌드를 쓴다.

```bash
ros2 launch uon_3f_gripper uon_3f_gripper_node.launch.py
```

```bash
ros2 run tiptop_execution gateway --namespace dsr01 --robot-system real \
  --gripper-config /workspaces/ros2_ws/src/execution/tiptop_execution/config/uon_native.measured.json
```

- gateway는 `127.0.0.1:8090`에서 요청을 받는다. 상태 전용은 `--read-only`를 쓴다.
- 관측 서버·작업 명령은 [실물 준비](m1013-hardware-preparation.md), 위치·전류 단위는 [그리퍼 설정](m1013-gripper-check.md)을 따른다.

## 변경 적용과 로그

- ROS 코드를 바꾸면 해당 패키지를 빌드하고 기존 노드를 재시작한다.
- 두산 드라이버의 최소 명령 변화량은 코드 상수 `1e-9 rad`이며 환경 변수는 없다.
- 실패·정지 응답은 ROS 터미널 로그와 작업 폴더의 `physical-execution.json`에서 확인한다.
- 단계별 제어 완료는 실제 물체의 파지·방출 성공과 구분한다.
