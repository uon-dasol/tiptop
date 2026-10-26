"""Execute a newly observed physical plan through the existing ROS gateway."""

import threading

from tiptop.doosan.client import validate_plan_backend
from tiptop.doosan.readiness import check_gateway_state
from tiptop.execute_plan import ExecutionFailure, execute_cutamp_plan


def execute_measured_plan(plan, client, *, cancelled=None):
    """Validate mode, measured model, complete path and current joints first.

    Controller completion does not prove physical object retention or delivery.
    This function neither recovers a failed grasp nor retries a command.
    """
    from tiptop.doosan.tamp import verify_gateway_contract

    cancelled = cancelled if cancelled is not None else threading.Event()
    if plan.get("version") != "1.0.0" or plan.get("robot_system") != 0:
        raise ValueError("Physical execution requires a newly observed physical version 1.0.0 plan")
    verify_gateway_contract(plan)  # In-process recorder only; no hardware calls.
    state = client.get_state()
    validate_plan_backend(plan, state)
    check_gateway_state(state, execute=True, observed_q=plan["q_init"])

    class CancellableClient:
        def __getattr__(self, name):
            def command(*args, **kwargs):
                if cancelled.is_set():
                    raise ExecutionFailure("Task cancelled; no further commands will be sent")
                return getattr(client, name)(*args, **kwargs)

            return command

    try:
        execute_cutamp_plan(plan["steps"], CancellableClient())
        final = client.get_state()
        if not final.get("success") or final.get("motion_status") != 0:
            raise ExecutionFailure("No idle robot feedback after execution")
    except BaseException:
        try:
            client.stop()
        except Exception:
            pass  # Preserve the action failure even if the gateway is unavailable.
        raise
    return dict(
        observation_id=plan["observation_id"],
        finished=True,
        success=None,
        execution_success=True,
        status="execution_completed_unverified",
        completed_steps=len(plan["steps"]),
        physical_grasp_verified=False,
        physical_placement_verified=False,
        final_state=final,
        verification_source="controller_feedback_only",
    )
