"""Attempt a feasible grasp's complete path before spending the full opt budget."""

import logging
import re
from dataclasses import replace

from tiptop.console import log_tag


def failure_text(failure):
    """Korean one-liner for cuTAMP's failure strings; anything unrecognised is shown unchanged."""
    text = str(failure)
    if "No satisfying particles" in text:
        return "제약을 만족한 입자가 없음 (입자 전부가 충돌 등 제약을 어김)"
    motion = re.search(r"Motion planning failed for (\d+)/(\d+)", text)
    if motion:
        return f"만족한 입자 {motion[2]}개 중 {motion[1]}개 모두 모션 계획 실패"
    return text


def log_pass(label, plan, seconds, failure):
    done = plan is not None
    log_tag("cuTAMP", f"{label} {'성공' if done else '실패'} · {seconds:.1f}s" + ("" if done else f" · {failure_text(failure)}"),
            "ok" if done else "fail")


def run_known_planning(run, env, config, q_init, ik, grasps, motion_gen, surfaces,
                       *, screened_ik=None, try_feasible_first=True):
    """Keep upstream constraint/ranking/path checks and restore full search on failure.

    Only the first pass gets a satisfying-particle stop threshold. A failed pass
    retries the original config and original IK solver, with the same supplied
    grasps. It does not widen candidates or change any motion parameters.
    """
    first_config = replace(config, prop_satisfying_break=1/config.num_particles) if try_feasible_first else config
    first_label = "1차 (만족 입자가 나오면 바로 모션 계획)" if try_feasible_first else "1차 (전체 최적화)"
    log_tag("cuTAMP", f"{first_label} 시작 · 입자 {config.num_particles}개", "info")
    plan, elapsed, failure = run(
        env, first_config, q_init, screened_ik if screened_ik is not None else ik, grasps, motion_gen, surfaces,
    )
    log_pass(first_label, plan, elapsed, failure)
    attempts = [dict(feasible_first=try_feasible_first, screened_ik_enabled=screened_ik is not None,
                     seconds=elapsed, failure=failure, complete=plan is not None)]
    used_cached_ik = screened_ik is not None and screened_ik.stats["reused_batches"] > 0
    if plan is None and (try_feasible_first or used_cached_ik):
        logging.getLogger(__name__).info("No complete path on first pass; restoring full optimization and ordinary IK")
        log_tag("cuTAMP", "2차 (전체 최적화 + 일반 IK로 재시도) 시작", "info")
        plan, fallback_elapsed, failure = run(env, config, q_init, ik, grasps, motion_gen, surfaces)
        log_pass("2차 (전체 최적화 + 일반 IK)", plan, fallback_elapsed, failure)
        elapsed += fallback_elapsed
        attempts.append(dict(feasible_first=False, screened_ik_enabled=False, seconds=fallback_elapsed,
                             failure=failure, complete=plan is not None))
    return plan, elapsed, failure, dict(feasible_first_enabled=try_feasible_first, attempts=attempts)
