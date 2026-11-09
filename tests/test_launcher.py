"""Launcher: command construction from the presets, run management and the token-protected viewer API."""

import http.client
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import pytest

from tiptop.launcher import FIELDS, KINDS, MODES, PRESETS_PATH, build_command, class_id_task, load_presets, save_preset
from tiptop.run_manager import RunManager
from tiptop.scripts.serve_perception_viewer import make_server

ROOT = Path(__file__).resolve().parents[1]
TASK = "car_leather_conditioner_bottle 집고 제자리에 놓아"
COMMON = [
    "--task", TASK, "--max-approach-tilt-deg", "20", "--cad-mask-min-iou", "0.6", "--min-table-clearance-m", "0.01",
    "--max-wrist-travel-deg", "90", "--min-engagement-depth-m", "0.02", "--table-support", "declared",
]


def arguments(mode, kind):
    return build_command(mode, load_presets()["presets"][mode], kind)[3:]


def test_gemini_execute_matches_the_hand_written_command():
    assert arguments("gemini_er", "execute") == [
        "--camera-url", "http://127.0.0.1:13348", "--hardware-profile", "tiptop/config/m1013_fixed_hardware.declared.json",
        "--gateway-url", "http://127.0.0.1:8090", "--perception-mode", "gemini_er", "--allow-gemini-once",
        *COMMON, "--execute", "--viewer",
    ]


def test_sam3_execute_matches_the_hand_written_command():
    assert arguments("sam3_reference", "execute") == [
        "--worker-socket", f"/tmp/tiptop-{os.getuid()}/known-task.sock", "--camera-url", "http://127.0.0.1:13348",
        "--hardware-profile", "tiptop/config/m1013_fixed_hardware.declared.json",
        "--gateway-url", "http://127.0.0.1:8090", "--perception-mode", "sam3_reference",
        "--reference-scene", "Lab/Uon_1F_Lab/table", *COMMON, "--execute", "--viewer",
    ]


def test_options_of_the_other_mode_and_gateway_are_left_out_of_planning_runs():
    gemini, sam3 = arguments("gemini_er", "plan"), arguments("sam3_reference", "plan")
    assert "--gateway-url" not in gemini + sam3 and "--execute" not in gemini + sam3
    assert "--allow-gemini-once" in gemini and "--worker-socket" not in gemini and "--reference-scene" not in gemini
    assert "--allow-gemini-once" not in sam3 and "--worker-socket" in sam3 and "--reference-scene" in sam3


def test_connection_check_uses_the_gateway_without_inference_options():
    for mode in MODES:
        args = arguments(mode, "check")
        assert "--check" in args and "--gateway-url" in args
        assert not {"--allow-gemini-once", "--worker-socket", "--viewer", "--execute"} & set(args)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", KINDS)
def test_every_generated_command_is_accepted_by_the_runner_parser(mode, kind):
    from tiptop.scripts.run_known_pick_place import parse_args

    args = parse_args(arguments(mode, kind))
    assert args.perception_mode == mode and args.execute == (kind == "execute") and args.check == (kind == "check")


def test_invalid_input_is_rejected_with_the_field_name():
    values = dict(load_presets()["presets"]["gemini_er"])
    with pytest.raises(ValueError, match="gateway_url is required"):
        build_command("gemini_er", dict(values, gateway_url=""), "execute")
    with pytest.raises(ValueError, match="declared"):
        build_command("gemini_er", dict(values, hardware_profile=""), "plan")
    with pytest.raises(ValueError, match="max_approach_tilt_deg must be a number"):
        build_command("gemini_er", dict(values, max_approach_tilt_deg="abc"), "plan")
    with pytest.raises(ValueError, match="finite"):
        build_command("gemini_er", dict(values, cad_mask_min_iou="nan"), "plan")
    with pytest.raises(ValueError, match="Unknown mode"):
        build_command("other", values, "plan")


def test_saved_defaults_keep_common_and_mode_values_apart(tmp_path):
    path = tmp_path / "presets.json"
    shutil.copy(PRESETS_PATH, path)
    values = dict(load_presets(path)["presets"]["sam3_reference"], max_approach_tilt_deg=15,
                  reference_scene="Lab/other", worker_socket=f"/tmp/tiptop-{os.getuid()}/other.sock")
    save_preset("sam3_reference", values, path)
    document = json.loads(path.read_text())
    assert document["common"]["max_approach_tilt_deg"] == 15
    assert document["sam3_reference"]["reference_scene"] == "Lab/other"
    assert document["sam3_reference"]["worker_socket"] == "/tmp/tiptop-{uid}/other.sock"
    assert load_presets(path)["presets"]["gemini_er"]["max_approach_tilt_deg"] == 15
    assert "reference_scene" not in document["common"] and document["gemini_er"] == {}
    with pytest.raises(ValueError, match="Unknown launcher fields"):
        save_preset("gemini_er", {"bogus": 1}, path)


def python_command(code):
    return lambda mode, values, kind: [sys.executable, "-c", code]


def wait_until(condition, timeout=10):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.02)


def test_run_manager_keeps_console_output_with_its_colour_codes_and_the_exit_code():
    manager = RunManager(ROOT, build=python_command("print('\\033[32mhello\\033[0m'); print('second')"))
    manager.start("gemini_er", {}, "plan")
    wait_until(lambda: not manager.running() and manager.log(0)["lines"][-1].startswith("[종료 코드"))
    assert manager.log(0)["lines"] == ["\x1b[32mhello\x1b[0m", "second", "[종료 코드 0]"]
    assert manager.log(2)["lines"] == ["[종료 코드 0]"] and manager.state()["returncode"] == 0


def test_run_manager_allows_one_run_and_stops_it_with_ctrl_c():
    manager = RunManager(ROOT, build=python_command("import time; print('up', flush=True); time.sleep(60)"))
    manager.start("gemini_er", {}, "plan")
    wait_until(lambda: manager.log(0)["lines"] == ["up"])
    with pytest.raises(RuntimeError, match="already in progress"):
        manager.start("gemini_er", {}, "plan")
    assert manager.stop() is True
    wait_until(lambda: not manager.running())
    assert manager.stop() is False


@pytest.fixture
def viewer(tmp_path):
    servers = []

    def start(launcher):
        server = make_server(tmp_path, port=0, launcher=launcher)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def request(server, method, path, *, body=None, token=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    send = dict(headers or {})
    if token:
        send["X-Launcher-Token"] = token
    connection.request(method, path, body=None if body is None else json.dumps(body), headers=send)
    response = connection.getresponse()
    data = response.read().decode()
    connection.close()
    return response.status, data


def test_page_carries_the_token_only_for_a_local_host(viewer):
    server = viewer(RunManager(ROOT, build=python_command("pass")))
    _, page = request(server, "GET", "/")
    assert server.launcher_token in page and "__LAUNCHER_TOKEN__" not in page
    _, foreign = request(server, "GET", "/", headers={"Host": "evil.example"})
    assert server.launcher_token not in foreign


def test_api_needs_the_token_a_same_origin_request_and_a_launcher(viewer, tmp_path):
    server = viewer(RunManager(ROOT, build=python_command("pass")))
    token = server.launcher_token
    assert request(server, "GET", "/api/presets")[0] == 403
    assert request(server, "GET", "/api/presets", token="wrong")[0] == 403
    assert request(server, "GET", "/api/presets", token=token, headers={"Origin": "http://evil.example"})[0] == 403
    assert request(server, "GET", "/api/presets", token=token, headers={"Host": "evil.example"})[0] == 403
    status, data = request(server, "GET", "/api/presets", token=token)
    assert status == 200 and {field["key"] for field in json.loads(data)["fields"]} == {field["key"] for field in FIELDS}
    read_only = viewer(None)
    assert request(read_only, "GET", "/api/presets", token=read_only.launcher_token)[0] == 403
    assert read_only.launcher_token == ""


def test_object_list_follows_the_catalog_order(viewer):
    server = viewer(RunManager(ROOT, build=python_command("pass")))
    status, data = request(server, "GET", "/api/objects", token=server.launcher_token)
    catalog = json.loads((ROOT / "tiptop/config/known_objects.json").read_text())["objects"]
    objects = json.loads(data)["objects"]
    assert status == 200 and [item["class_id"] for item in objects] == [item["class_id"] for item in catalog]
    assert objects[0]["class_id"] == "black_pepper_shaker" and len(objects) == 28
    assert objects[0]["name_ko"] == "후추통" and all(item["name_ko"] for item in objects)


def test_korean_display_names_reach_the_runner_as_class_ids():
    catalog = json.loads((ROOT / "tiptop/config/known_objects.json").read_text())["objects"]
    names = [item["name_ko"] for item in catalog]
    assert len(set(names)) == len(names)
    for item in catalog:
        assert class_id_task(f"{item['name_ko']} 집고 제자리에 놓아") == f"{item['class_id']} 집고 제자리에 놓아"
    assert class_id_task("3구 멀티탭 집고 제자리에 놓아") == "t_shaped_3_outlet_adapter 집고 제자리에 놓아"
    assert class_id_task("소스통 집고 제자리에 놓아").startswith("small_squeeze_bottle ")
    assert class_id_task("불닭소스통 집고 제자리에 놓아").startswith("samyang_buldak_sauce_bottle ")
    assert class_id_task(TASK) == TASK
    assert class_id_task("후추통을 집어") == "후추통을 집어"
    values = dict(load_presets()["presets"]["gemini_er"], task="후추통 집고 제자리에 놓아")
    assert build_command("gemini_er", values, "execute")[build_command("gemini_er", values, "execute").index("--task") + 1] == (
        "black_pepper_shaker 집고 제자리에 놓아"
    )


def test_run_api_starts_one_run_at_a_time_and_stops_it(viewer):
    server = viewer(RunManager(ROOT, build=python_command("import time; time.sleep(60)")))
    token = server.launcher_token
    values = load_presets()["presets"]["gemini_er"]
    body = dict(mode="gemini_er", values=values, kind="execute")
    assert request(server, "POST", "/api/run", body=body, token=token)[0] == 200
    assert request(server, "POST", "/api/run", body=body, token=token)[0] == 409
    assert json.loads(request(server, "GET", "/api/status", token=token)[1])["running"] is True
    assert json.loads(request(server, "POST", "/api/stop", body={}, token=token)[1])["stopped"] is True
    assert request(server, "POST", "/api/run", body={}, token=token)[0] == 400
