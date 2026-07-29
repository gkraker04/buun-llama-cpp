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
assert SUPPORTED_SCHEMAS == (1, 2, 3, 4)
assert len(schema4) == 2
assert {rec["yield"]["plan_state"] for rec in schema4} == {
    "not_required",
    "planned",
}
assert all(rec["yield"]["actual_state"] == "not_observed" for rec in schema4)
assert any(rec["yield"]["projected_domains"] for rec in schema4)

try:
    list(iter_cache_plan_records([fixture], supported_schemas=(1, 2, 3)))
except UnsupportedSchemaError:
    pass
else:
    raise AssertionError("frozen schema-3 reader accepted a schema-4 record")

print("cache-plan schema compatibility checks passed")
