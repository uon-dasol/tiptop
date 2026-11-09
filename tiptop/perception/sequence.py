"""Prepare catalog assets and reference bundles without observing or moving a robot."""

import asyncio
import time

import aiohttp

from tiptop.object_catalog import sequence_objects


class SequencePreparationError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def prepare_local_assets(objects):
    from tiptop.doosan.grasp_database import ensure_database_cache_capacity, warm_database_grasps
    from tiptop.perception.cad_cache import ensure_cad_cache_capacity, load_cad_mesh

    capacities = dict(cad=ensure_cad_cache_capacity(len(objects)), db=ensure_database_cache_capacity(len(objects)))
    rows = []
    print(f"\033[33mLOCAL_PREPARING objects={len(objects)} CAD・파지 DB 준비 중\033[0m", flush=True)
    for index, item in enumerate(objects, 1):
        started = time.perf_counter()
        try:
            load_cad_mesh(item["cad"], item["cad_scale_to_m"])
            warm_database_grasps(item["db"], class_id=item["class_id"])
        except Exception as error:
            print(f"\033[31mLOCAL_PREPARE_FAILED {index}/{len(objects)} {item['class_id']} | {error}\033[0m", flush=True)
            raise
        row = dict(class_id=item["class_id"], elapsed_s=time.perf_counter()-started)
        rows.append(row)
        print(f"\033[33mLOCAL_PREPARE {index}/{len(objects)} {row['class_id']} "
              f"준비 완료 ({row['elapsed_s']:.2f}s)\033[0m", flush=True)
    return dict(cache_capacity=capacities, objects=rows)


async def _prepare_references(session, objects, server_url, scene, report):
    if not scene:
        raise ValueError("SAM3 sequence preparation requires a reference scene")
    report["stage"] = "sam3_references"
    report["references"] = []
    for item in objects:
        async with session.get(server_url.rstrip("/")+"/reference", params=dict(class_id=item["class_id"], scene=scene),
                               timeout=aiohttp.ClientTimeout(total=30)) as response:
            response.raise_for_status()
            result = await response.json()
        if (result.get("success") is not True or result.get("class_id") != item["class_id"]
                or result.get("reference_scene") != scene or result.get("inference_performed") is not False):
            raise ValueError("SAM3 reference preparation returned a different class/scene or ran inference")
        report["references"].append(result)


async def prepare_sequence(session, class_ids, *, pose_url, sam3_url=None, reference_scene=None):
    """Finish preparation before any task can capture an observation.

    Old servers must explicitly report that preparation is supported. No dummy
    /estimate or /segment-reference calls are issued to simulate preparation.
    """
    report = dict(success=False, inference_performed=False, class_ids=[])
    started = time.perf_counter()
    try:
        objects = sequence_objects(class_ids)
        report["class_ids"] = [item["class_id"] for item in objects]
        report["stage"] = "pose_server_capability"
        async with session.get(pose_url.rstrip("/")+"/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
            response.raise_for_status()
            health = await response.json()
        if health.get("preparation_supported") is not True:
            raise ValueError("FoundationPose server lacks /prepare; restart it with the updated code before preparing")
        if sam3_url is not None:
            # Fail on a missing reference before expensive CAD preparation.
            await _prepare_references(session, objects, sam3_url, reference_scene, report)
        report["stage"] = "local_assets"
        report["local_assets"] = await asyncio.to_thread(prepare_local_assets, objects)
        report["stage"] = "foundation_pose"
        payload = dict(objects=[dict(class_id=item["class_id"], mesh_path=str(item["cad"]),
                                     mesh_scale=item["cad_scale_to_m"]) for item in objects])
        print(f"FOUNDATIONPOSE_PREPARING objects={len(objects)} (no inference)", flush=True)
        async with session.post(pose_url.rstrip("/")+"/prepare", json=payload,
                                timeout=aiohttp.ClientTimeout(total=600)) as response:
            result = await response.json()
            report["foundation_pose"] = result
            response.raise_for_status()
        prepared = result.get("objects", [])
        if (result.get("success") is not True or result.get("inference_performed") is not False
                or [item.get("class_id") for item in prepared] != report["class_ids"]
                or not all(item.get("success") is True for item in prepared)):
            raise ValueError("FoundationPose did not prepare every requested class")
        report.update(success=True, stage="ready")
        return report
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise SequencePreparationError(str(error), report) from error
    finally:
        report["elapsed_s"] = time.perf_counter()-started
