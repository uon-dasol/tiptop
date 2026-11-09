"""Attempt a feasible grasp's complete path before spending the full opt budget."""

import logging
from dataclasses import replace


def run_known_planning(run, env, config, q_init, ik, grasps, motion_gen, surfaces,
                       *, screened_ik=None, try_feasible_first=True):
    """Keep upstream constraint/ranking/path checks and restore full search on failure.

    Only the first pass gets a satisfying-particle stop threshold. A failed pass
    retries the original config and original IK solver, with the same supplied
    grasps. It does not widen candidates or change any motion parameters.
    """
    first_config = replace(config, prop_satisfying_break=1/config.num_particles) if try_feasible_first else config
    plan, elapsed, failure = run(
        env, first_config, q_init, screened_ik if screened_ik is not None else ik, grasps, motion_gen, surfaces,
    )
    attempts = [dict(feasible_first=try_feasible_first, screened_ik_enabled=screened_ik is not None,
                     seconds=elapsed, failure=failure, complete=plan is not None)]
    used_cached_ik = screened_ik is not None and screened_ik.stats["reused_batches"] > 0
    if plan is None and (try_feasible_first or used_cached_ik):
        logging.getLogger(__name__).info("No complete path on first pass; restoring full optimization and ordinary IK")
        plan, fallback_elapsed, failure = run(env, config, q_init, ik, grasps, motion_gen, surfaces)
        elapsed += fallback_elapsed
        attempts.append(dict(feasible_first=False, screened_ik_enabled=False, seconds=fallback_elapsed,
                             failure=failure, complete=plan is not None))
    return plan, elapsed, failure, dict(feasible_first_enabled=try_feasible_first, attempts=attempts)
