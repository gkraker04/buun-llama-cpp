#!/usr/bin/env python3

import pathlib
import sys


root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "tools" / "server" / "bench"))

from cache_plan_common import (  # noqa: E402
    SUPPORTED_SCHEMAS,
    UnsupportedSchemaError,
    iter_cache_plan_records,
)


fixture = root / "tools" / "server" / "bench" / "fixtures" / "cache-plan-golden.jsonl"

records = list(iter_cache_plan_records([fixture]))
schema4 = [rec for rec in records if rec["schema_version"] == 4]
schema5 = [rec for rec in records if rec["schema_version"] == 5]
assert SUPPORTED_SCHEMAS == (1, 2, 3, 4, 5)
assert len(schema4) == 2
assert {rec["yield"]["plan_state"] for rec in schema4} == {
    "not_required",
    "planned",
}
assert all(rec["yield"]["actual_state"] == "not_observed" for rec in schema4)
assert any(rec["yield"]["projected_domains"] for rec in schema4)
assert len(schema5) == 2
assert {rec["authority"]["state"] for rec in schema5} == {
    "shadow",
    "fallback_legacy",
}
assert all("target_slot_id" in candidate and "origin_tier" in candidate
           for rec in schema5 for candidate in rec["candidates"])
assert any(rec["authority"]["disagreed"] for rec in schema5)
assert all(rec["accounting"]["schema_version"] == 2 for rec in schema5)

try:
    list(iter_cache_plan_records([fixture], supported_schemas=(1, 2, 3, 4)))
except UnsupportedSchemaError:
    pass
else:
    raise AssertionError("frozen schema-4 reader accepted a schema-5 record")

print("cache-plan schema compatibility checks passed")
