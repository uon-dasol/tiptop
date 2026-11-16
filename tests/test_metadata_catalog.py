"""CAD attribute row numbers and optional legacy names preserve object IDs."""

import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture(params=["sam3/reference_catalog.py", "examples/real_object_catalog.py"])
def catalog_module(request):
    path = Path(__file__).resolve().parents[1] / "SAM3" / request.param
    name = "test_catalog_" + request.param.replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_cad_numbering_with_blank_old_name_preserves_class_id(tmp_path, catalog_module):
    path = tmp_path / "attributes.csv"
    path.write_text("No,Object_name,Class_name,Old_name,물체명 한글, Year\n"
                    "1,black_pepper_shaker,obj_291,,순후추 통,2026\n"
                    "29,chair_leg_caps,obj_309,,,2026\n")
    identity = catalog_module.resolve_real_object("chair_leg_caps", path)
    assert identity.object_id == "obj_309"
    assert identity.applied_key == "chair_leg_caps"
    assert identity.old_name is None
    assert identity.key_namespace == "Object_name"
    assert catalog_module.resolve_real_object("obj_291", path).applied_key == "black_pepper_shaker"


def test_legacy_catalog_still_requires_no_and_class_to_agree(tmp_path, catalog_module):
    path = tmp_path / "legacy.csv"
    path.write_text("No,Object_name,Class_name,Old_name,Year\n1,pepper,obj_291,,2026\n")
    with pytest.raises(catalog_module.RealObjectCatalogError, match="expected 'obj_001'"):
        catalog_module.resolve_real_object("pepper", path)


def test_cad_class_id_still_requires_obj_number(tmp_path, catalog_module):
    path = tmp_path / "attributes.csv"
    path.write_text("No,Object_name,Class_name,Old_name,물체명 한글,Year\n1,pepper,invalid,,,2026\n")
    with pytest.raises(catalog_module.RealObjectCatalogError, match="Invalid CAD Class_name"):
        catalog_module.resolve_real_object("pepper", path)
