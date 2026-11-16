"""Execute a newly observed physical plan through the existing ROS gateway."""

import logging
import json
from pathlib import Path
import threading
import time

from tiptop.doosan.client import describe_trajectory_error, validate_plan_backend
from tiptop.doosan.readiness import check_gateway_state
from tiptop.execute_plan import ExecutionFailure, execute_cutamp_plan

_log = logging.getLogger(__name__)


def execute_measured_plan(plan, client, *, cancelled=None, report_path=None):
    """Validate mode, measured model, complete path and current joints first.

    Controller completion does not prove physical object retention or delivery.
    This function neither recovers a failed grasp nor retries a command.
    """
    from tiptop.doosan.tamp import verify_gateway_contract

    cancelled = cancelled if cancelled is not None else threading.Event()
    report = dict(observation_id=plan.get("observation_id"), finished=False,
                  execution_success=False, physical_grasp_verified=False, events=[])
    acknowledged_stop = None

    def save_report():
        if report_path is not None:
            path = Path(report_path)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(path)
    preflight_started = time.perf_counter()
    if plan.get("version") != "1.0.0" or plan.get("robot_system") != 0:
        raise ValueError("Physical execution requires a newly observed physical version 1.0.0 plan")
    verify_gateway_contract(plan)  # In-process recorder only; no hardware calls.
    state = client.get_state()
    validate_plan_backend(plan, state)
    check_gateway_state(state, execute=True, observed_q=plan["q_init"])
    report["execution_timings"] = dict(preflight_s=time.perf_counter() - preflight_started)

    class CancellableClient:
        step_index = 0
        preparing = True

        def __getattr__(self, name):
            def command(*args, **kwargs):
                nonlocal acknowledged_stop
                if cancelled.is_set():
                    raise ExecutionFailure("Task cancelled; no further commands will be sent")
                if self.preparing:
                    stage = dict(step=0, type="gripper", action="open", label="initial opening")
                else:
                    self.step_index += 1
                    step = plan["steps"][self.step_index - 1]
                    stage = {k: step[k] for k in ("type", "label", "action") if k in step}
                    stage["step"] = self.step_index
                command_started = time.perf_counter()
                event = dict(**stage, method=name, started_unix_s=time.time(), status="running")
                report["events"].append(event)
                save_report()
                _log.info("Physical execution stage: %s", stage)
                try:
                    result = getattr(client, name)(*args, **kwargs)
                    explanation = describe_trajectory_error(result.get("error")) if result else None
                    if explanation:
                        print(f"{explanation[1]}{explanation[0]}\033[0m", flush=True)
                    stopped = result.get("stop_result") if result else None
                    if stopped and stopped.get("success") is True and stopped.get("arm_stop_acknowledged") is True:
                        acknowledged_stop = stopped
                    event.update(result=result, status="completed" if result and result.get("success") else "failed")
                    return result
                except BaseException as error:
                    event.update(status="failed", error=str(error))
                    raise
                finally:
                    event["finished_unix_s"] = time.time()
                    event["duration_s"] = time.perf_counter() - command_started
                    save_report()

            return command

    try:
        commands = CancellableClient()
        _log.info("Preparing physical task: opening the gripper before arm approach")
        preparation_started = time.perf_counter()
        preparation = commands.open_gripper()
        if not preparation or preparation.get("success") is not True or preparation.get("endpoint_reached") is not True:
            raise ExecutionFailure("Initial gripper opening failed; arm approach blocked: " + str(preparation))
        opened_state = client.get_state()
        check_gateway_state(opened_state, execute=True, observed_q=plan["q_init"], require_open=True)
        report["execution_timings"]["initial_gripper_open_s"] = time.perf_counter() - preparation_started
        _log.info("Initial gripper opening verified: %s", opened_state["gripper"])
        commands.preparing = False
        execution_started = time.perf_counter()
        execute_cutamp_plan(plan["steps"], commands)
        report["execution_duration_s"] = time.perf_counter() - execution_started
        report["execution_timings"]["planned_steps_s"] = report["execution_duration_s"]
        postflight_started = time.perf_counter()
        final = client.get_state()
        if not final.get("success") or final.get("motion_status") != 0:
            raise ExecutionFailure("No idle robot feedback after execution")
        report["execution_timings"]["postflight_s"] = time.perf_counter() - postflight_started
    except BaseException as error:
        report.update(finished=True, error=str(error), finished_unix_s=time.time())
        try:
            report["stop_result"] = acknowledged_stop if acknowledged_stop is not None else client.stop()
        except Exception as stop_error:
            report["stop_error"] = str(stop_error)
        save_report()
        raise
    report.update(finished=True, execution_success=True, finished_unix_s=time.time())
    save_report()
    return dict(
        observation_id=plan["observation_id"],
        finished=True,
        success=None,
        execution_success=True,
        execution_duration_s=report["execution_duration_s"],
        execution_timings=report["execution_timings"],
        status="execution_completed_unverified",
        completed_steps=len(plan["steps"]),
        gripper_preparation=dict(command=preparation, verified_state=opened_state["gripper"]),
        physical_grasp_verified=False,
        physical_placement_verified=False,
        final_state=final,
        verification_source="controller_feedback_only",
    )
