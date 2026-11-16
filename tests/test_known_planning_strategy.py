"""Search continuation guards without models, devices or robot calls."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from tiptop.doosan.known_planning import run_known_planning


@dataclass
class Config:
    num_particles: int = 16
    num_opt_steps: int = 100
    prop_satisfying_break: float | None = None


@pytest.mark.parametrize("first_success", [True, False])
@pytest.mark.parametrize("fallback_success", [True, False])
def test_complete_path_required_and_failure_restores_original_solver_and_budget(first_success, fallback_success):
    cfg, ordinary, grasps = Config(), object(), object()
    cached = SimpleNamespace(stats=dict(reused_batches=1))
    calls = []
    plan = [dict(label="complete path")]

    def run(env, config, q, ik, supplied, mg, surfaces):
        calls.append((config, ik))
        assert supplied is grasps
        if len(calls) == 1:
            assert config.prop_satisfying_break == 1/16 and config.num_opt_steps == 100
            assert ik is cached
            return (plan, .1, None) if first_success else (None, .1, "initial path failed")
        assert config is cfg and config.prop_satisfying_break is None and config.num_opt_steps == 100
        assert ik is ordinary
        return (plan, .8, None) if fallback_success else (None, .8, "full search failed")

    result, elapsed, failure, report = run_known_planning(
        run, None, cfg, None, ordinary, grasps, None, [], screened_ik=cached,
    )
    assert len(calls) == (1 if first_success else 2)
    assert result is (plan if first_success or fallback_success else None)
    assert elapsed == pytest.approx(.1 if first_success else .9)
    assert failure == (None if first_success or fallback_success else "full search failed")
    assert report["attempts"][0]["complete"] is first_success
    assert cfg.prop_satisfying_break is None


@pytest.mark.parametrize("cache_hit", [False, True])
def test_reuse_only_fallback_requires_an_actual_hit(cache_hit):
    calls = []
    cfg, ordinary = Config(), object()
    cached = SimpleNamespace(stats=dict(reused_batches=int(cache_hit)))

    def run(env, config, q, ik, grasps, mg, surfaces):
        calls.append(ik)
        assert config is cfg
        return None, 1., "no path"

    result, elapsed, failure, report = run_known_planning(
        run, None, cfg, None, ordinary, None, None, [], screened_ik=cached, try_feasible_first=False,
    )
    assert calls == ([cached, ordinary] if cache_hit else [cached])
    assert result is None and failure == "no path" and elapsed == len(calls)


def test_ordinary_mode_is_one_unmodified_attempt():
    config, ordinary, calls = Config(), object(), []

    def run(*args):
        calls.append(args)
        return None, 2., "no path"

    run_known_planning(run, None, config, None, ordinary, None, None, [], try_feasible_first=False)
    assert len(calls) == 1 and calls[0][1] is config and calls[0][3] is ordinary


def test_unexpected_planning_exception_is_not_retried():
    calls = []

    def run(*args):
        calls.append(args)
        raise RuntimeError("model error")

    with pytest.raises(RuntimeError, match="model error"):
        run_known_planning(run, None, Config(), None, None, None, None, [])
    assert len(calls) == 1


def test_each_search_pass_is_logged_with_its_result_time_and_a_readable_failure(capsys):
    from tiptop.doosan.known_planning import failure_text

    outcomes = iter([(None, .5, "No satisfying particles found after optimizing all 1 plan(s)"), ([dict(label="path")], 1.5, None)])
    run_known_planning(lambda *args: next(outcomes), None, Config(), None, object(), None, None, None, try_feasible_first=True)
    lines = capsys.readouterr().out.splitlines()
    text = "\n".join(lines)
    assert "1차 (만족 입자가 나오면 바로 모션 계획) 시작 · 입자 16개" in text
    assert "1차 (만족 입자가 나오면 바로 모션 계획) 실패 · 0.5s · 제약을 만족한 입자가 없음" in text
    assert "2차 (전체 최적화 + 일반 IK) 성공 · 1.5s" in text
    assert failure_text("Motion planning failed for 6/6 satisfying particle(s)") == "만족한 입자 6개 중 6개 모두 모션 계획 실패"
    assert failure_text("something else") == "something else"
