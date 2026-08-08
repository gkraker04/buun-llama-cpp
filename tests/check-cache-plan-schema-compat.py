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
fixture_v7 = root / "tools" / "server" / "bench" / "fixtures" / "cache-plan-v7-golden.jsonl"

records = list(iter_cache_plan_records([fixture, fixture_v7]))
schema4 = [rec for rec in records if rec["schema_version"] == 4]
schema5 = [rec for rec in records if rec["schema_version"] == 5]
schema6 = [rec for rec in records if rec["schema_version"] == 6]
schema7 = [rec for rec in records if rec["schema_version"] == 7]
assert SUPPORTED_SCHEMAS == (1, 2, 3, 4, 5, 6, 7)
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
assert len(schema6) == 1
assert schema6[0]["destruction"]["reason"] == "effect_drift"
assert schema6[0]["destruction"]["effects"] == [
    {
        "effect": "same_target_cold_replacement",
        "action_class": "slot_drop",
        "physical_reason": "slot_rebind",
    },
    {
        "effect": "different_host_source_consumption",
        "action_class": "host_artifact_drop",
        "physical_reason": "cache_update",
    },
    {
        "effect": "checkpoint_member_drop",
        "action_class": "checkpoint_drop",
        "physical_reason": "checkpoint_replace",
    },
]
assert schema6[0]["destruction"]["admission_sequence"] > 0
assert schema6[0]["destruction"]["quote_duration_us"] >= 0
assert schema6[0]["destruction"]["quote_accounting_serial"] == \
       schema6[0]["accounting"]["serial"]
assert schema6[0]["destruction"]["manifest_digest"] != "unavailable"
assert schema6[0]["destruction"]["union_effect_digest"] != "unavailable"
assert schema6[0]["destruction"]["recovery_citation"] == "resolved"
assert schema6[0]["destruction"]["recovery_source"] == {
    "artifact_id": 21,
    "manifest_digest":
        "0300000000000000000000000000000000000000000000000000000000000000",
}
assert schema6[0]["destruction"]["selected"] == {
    "attention": [],
    "recurrent": [],
}
assert "projected_domains" not in schema6[0]["destruction"]
assert schema6[0]["seq_cp_capability"] is True
assert schema6[0]["yield"]["actual_state"] == "not_observed"
assert schema6[0]["accounting"]["schema_version"] == 2

assert len(schema7) == 1
assert schema7[0]["accounting"]["schema_version"] == 2
assert schema7[0]["optimizer"]["policy_version"] == 1
assert schema7[0]["optimizer"]["mode"] == "off"
assert schema7[0]["optimizer"]["retention_policy"] == "historical"
assert schema7[0]["optimizer"]["retention_summary"]["state"] == "not_attempted"
assert schema7[0]["optimizer"]["retention_summary"]["outcome_counts"] == {
    "executed": 0,
    "deferred": 0,
    "publication_skipped": 0,
    "blocked": 0,
}

try:
    list(iter_cache_plan_records([fixture_v7], supported_schemas=(1, 2, 3, 4, 5, 6)))
except UnsupportedSchemaError:
    pass
else:
    raise AssertionError("frozen schema-6 reader accepted a schema-7 record")

print("cache-plan schema compatibility checks passed")
