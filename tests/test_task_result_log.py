import json

from tiptop import task_result_log as log
from tiptop.object_catalog import grasp_effort_counts


def make_task(tmp_path, number, *, effort=50, position=1272, with_execution=True):
    output = tmp_path / f"task-{number}"
    output.mkdir()
    (output / "plan.json").write_text(json.dumps(dict(
        class_id="disney_mickey_handled_dipper",
        steps=[dict(label="Pick.approach"), dict(label="Pick.close", effort_counts=effort)])))
    if with_execution:
        (output / "physical-execution.json").write_text(json.dumps(dict(events=[
            dict(method="open_gripper", result=dict(position_counts=3400)),
            dict(method="close_gripper", result=dict(position_counts=position, current_counts=-(effort - 1), current_limit_counts=effort))])))
    return output


def test_entry_and_line_show_the_object_effort_close_position_and_an_unconfirmed_result(tmp_path):
    entry = log.task_log_entry(make_task(tmp_path, 877), dict(status="execution_completed_unverified", executed=True), 18.43)
    assert entry["task"] == 877 and entry["name_ko"] == "바가지"
    assert log.log_line(entry) == "TASK 877 · 바가지 · effort 50 · 전류 -49 · 위치 1272 · 18.4초 · 파지 성공 · 작업 완료"


def test_a_close_that_ended_at_or_below_the_empty_close_limit_is_a_failed_grasp(tmp_path):
    for number, position, outcome in ((879, 315, "실패"), (880, 263, "실패"), (881, 316, "성공"), (882, 1212, "성공")):
        entry = log.task_log_entry(make_task(tmp_path, number, position=position), dict(status="done"), 10)
        assert entry["grasp"] == outcome and f"파지 {outcome}" in log.log_line(entry)


def test_a_task_that_stopped_before_closing_has_no_position_and_names_its_failed_stage(tmp_path):
    output = make_task(tmp_path, 878, with_execution=False)
    (output / "plan.json").unlink()  # planning failed: only the registered pose exists
    (output / "registered-pose.json").write_text(json.dumps(dict(class_id="disney_mickey_handled_dipper")))
    entry = log.task_log_entry(output, dict(status="failed", failed_stage="planning"), 3.64)
    effort = grasp_effort_counts("disney_mickey_handled_dipper")  # read from the config, not from a plan
    assert log.log_line(entry) == f"TASK 878 · 바가지 · effort {effort} · 전류 - · 위치 - · 3.6초 · 작업 중단(planning)"


def test_the_log_appends_to_one_file_and_shows_the_newest_hundred_first(tmp_path):
    path = tmp_path / "task-log.jsonl"
    for number in range(1, 106):
        log.append_task_log(make_task(tmp_path, number), dict(status="execution_completed_unverified"), 10, path)
    shown = log.read_task_log(path)
    assert len(path.read_text().splitlines()) == 105  # nothing is dropped from the file
    assert [entry["task"] for entry in shown][:2] == [105, 104] and len(shown) == 100 and shown[-1]["task"] == 6


def test_a_missing_file_or_a_broken_line_gives_what_can_be_read_and_writing_never_raises(tmp_path):
    path = tmp_path / "task-log.jsonl"
    assert log.read_task_log(path) == []
    path.write_text('{"task": 1}\nnot json\n{"task": 2}\n')
    assert [entry["task"] for entry in log.read_task_log(path)] == [2, 1]
    log.append_task_log(tmp_path / "missing" / "task-9", {}, 1.0, tmp_path)  # a directory instead of a file


def test_a_person_can_enter_and_remove_what_they_saw_only_for_logged_tasks_and_listed_values(tmp_path):
    path = tmp_path / "task-log.jsonl"
    log.append_task_log(make_task(tmp_path, 7), dict(status="execution_completed_unverified"), 10, path)
    log.TASK_LOG_PATH.write_text(path.read_text())
    log.set_task_mark(7, "result", "이송 중 놓침")
    log.set_task_mark(7, "perception", "오인식")
    assert log.read_task_marks() == {"7": dict(result="이송 중 놓침", perception="오인식")}
    row = log.task_log_rows()[0]
    assert (row["result"], row["perception"]) == ("이송 중 놓침", "오인식")
    log.set_task_mark(7, "result", "")
    assert log.read_task_marks() == {"7": dict(perception="오인식")} and log.task_log_rows()[0]["result"] == ""
    log.set_task_mark(7, "perception", "")
    assert log.read_task_marks() == {}
    for task, field, value in ((8, "result", "성공"), (7, "result", "실패"), (7, "perception", "성공"), (7, "name", "성공"),
                               (7, "result", None)):
        try:
            log.set_task_mark(task, field, value)
        except ValueError:
            continue
        raise AssertionError(f"accepted task {task} {field} {value!r}")


def test_a_broken_marks_file_reads_as_no_results(tmp_path):
    log.TASK_MARKS_PATH.write_text("{broken")
    assert log.read_task_marks() == {}


def test_a_task_run_in_demo_mode_says_so_after_its_outcome(tmp_path):
    output = make_task(tmp_path, 9)
    result = dict(status="execution_completed_unverified", executed=True)
    assert log.log_line(log.task_log_entry(output, result, 10)).endswith("작업 완료")
    assert log.task_log_entry(output, result, 10)["demo"] is False
    entry = log.task_log_entry(output, result, 10, demo=True)
    assert log.log_cells(entry)[-2:] == ["작업 완료", "데모 모드"] and log.log_line(entry).endswith("작업 완료 · 데모 모드")


def test_the_recognition_mode_is_the_first_column_and_names_match_the_mode_buttons(tmp_path):
    for mode, name in (("gemini_er", "Gemini"), ("sam3_reference", "SAM3")):
        entry = log.task_log_entry(make_task(tmp_path, 900 + len(name)), dict(status="done", perception_mode=mode), 10)
        assert entry["perception_mode"] == mode
        assert log.log_cells(entry)[0] == name and log.log_cells(entry)[1].startswith("TASK ")
        assert log.log_line(entry).startswith(f"{name} · TASK ")
    unknown = log.task_log_entry(make_task(tmp_path, 905), dict(status="done", perception_mode="other"), 10)
    assert log.log_cells(unknown)[0] == "" and log.log_line(unknown).startswith("TASK ")


def test_older_lines_without_a_mode_take_it_from_the_task_result_file(tmp_path, monkeypatch):
    monkeypatch.setattr(log, "ROOT", tmp_path)
    output = tmp_path / "tasks/task-910"
    output.mkdir(parents=True)
    (output / "task-result.json").write_text(json.dumps(dict(perception_mode="gemini_er")))
    entry = dict(task=910, output="tasks/task-910", name_ko="바가지")  # a line written before the mode was logged
    assert log.perception_mode(entry) == "gemini_er" and log.log_cells(entry)[0] == "Gemini"
    assert log.log_cells(dict(task=911, output="tasks/task-911"))[0] == ""  # no result file: blank, not an error
    assert log.log_cells(dict(task=912))[0] == ""
