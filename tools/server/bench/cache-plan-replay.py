#!/usr/bin/env python3
# cache-plan-replay.py — B-6 offline replay/eval over CACHE_PLAN records (schema v1/v2).
#
# Consumes server log files (grepping the 'CACHE_PLAN {json}' lines) or raw JSONL, and
# recomputes every report from the JSON alone (no in-process state):
#   - decision mix: outcomes, shipped chosen provider histogram, selection histogram
#   - agreement rate (v2, shadow computed): shipped chosen candidate IN shadow tie set
#   - disagreement inventory: records where shipped is outside the tie set, with the
#     exact predicted saving (shipped predicted total - shadow choice predicted total)
#   - calibration residuals: predicted total of the shipped choice vs measured ttft_us
#     (end-to-end residual — ttft includes work outside the modeled terms; B-5)
#   - invalidation/fault summaries: restore_failed outcomes, overflowed/truncated
#     inventories, observer shadow_unavailable growth
#
# Agreement semantics (pins v2/v3): a tie is NOT a disagreement — agreement means the
# shipped choice is a member of the shadow tie set. Records without shadow output (v1, or
# v2 with shadow unavailable) count toward the decision mix only.

import argparse
import json
import sys
from collections import Counter

from cache_plan_common import UnsupportedSchemaError, iter_cache_plan_records


def known(v):
    return v if isinstance(v, (int, float)) else None


def shipped_candidate_id(rec):
    # the COMPLETE shipped plan ordinal (chain on composed deliveries); fall back to the
    # terminal provider's row for records predating shipped_plan_candidate
    for key in ("shipped_plan_candidate", "chosen_candidate"):
        cid = rec.get(key)
        if isinstance(cid, int):
            return cid
    return None


def candidate_by_id(rec, cid):
    for c in rec.get("candidates", []):
        if c.get("id") == cid:
            return c
    return None


def main():
    ap = argparse.ArgumentParser(description="offline replay/eval over CACHE_PLAN records")
    ap.add_argument("logs", nargs="+", help="server log files or JSONL record files")
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    ap.add_argument("--max-disagreements", type=int, default=20,
                    help="cap on itemized disagreements in the report (counts are never capped)")
    args = ap.parse_args()

    n = 0
    by_schema   = Counter()
    outcomes    = Counter()
    chosen      = Counter()
    selections  = Counter()
    inv_states  = Counter()
    shadow_missing = 0
    agreements     = 0
    disagreements  = []
    residuals      = []
    restore_failed = 0

    try:
        records = list(iter_cache_plan_records(args.logs))
    except UnsupportedSchemaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for rec in records:
        n += 1
        by_schema[rec["schema_version"]] += 1
        outcomes[rec["outcome"]] += 1
        chosen[rec.get("chosen", "unknown")] += 1
        selections[rec.get("selection", "none")] += 1
        if rec["outcome"] == "restore_failed_fell_back_cold":
            restore_failed += 1
        for prov, st in rec.get("inventory_states", {}).items():
            inv_states[f"{prov}:{st}"] += 1

        shadow = rec.get("shadow")
        cid    = shipped_candidate_id(rec)
        if not isinstance(shadow, dict) or cid is None:
            shadow_missing += 1
            continue

        tie_set = shadow.get("tie_set", [])
        sc      = candidate_by_id(rec, shadow.get("choice"))
        sh      = candidate_by_id(rec, cid)
        if cid in tie_set:
            agreements += 1
        else:
            saving = None
            if sc and sh:
                pt_shadow  = known(sc.get("predicted_total_us"))
                pt_shipped = known(sh.get("predicted_total_us"))
                if pt_shadow is not None and pt_shipped is not None:
                    saving = pt_shipped - pt_shadow
            disagreements.append({
                "id_task":            rec.get("id_task"),
                "shipped_provider":   rec.get("chosen"),
                "shipped_candidate":  cid,
                "shadow_choice":      shadow.get("choice"),
                "shadow_provider":    sc.get("provider") if sc else None,
                "predicted_saving_us": saving,
            })
        # end-to-end residual where the shipped choice carries a predicted total
        if sh:
            pt   = known(sh.get("predicted_total_us"))
            ttft = known(rec.get("ttft_us"))
            if pt is not None and ttft is not None:
                residuals.append(pt - ttft)

    # SAMPLED-PREFIX clusters (see the server-side comment): a 64-token-window FNV hash,
    # so distinct workloads sharing a long leading boilerplate merge and collisions are
    # possible. Good enough to separate workload variants offline; never an identity.
    families = {}
    for rec in records:
        fam = rec.get("identity", {}).get("prefix_tokens")
        if not isinstance(fam, int):
            continue
        f = families.setdefault(fam, {"n": 0, "outcomes": Counter(), "reused": 0, "replayed": 0,
                                      "ttft_us": 0, "ttft_n": 0})
        f["n"] += 1
        f["outcomes"][rec["outcome"]] += 1
        for k, key in (("reused", "n_reused_tokens"), ("replayed", "n_replayed_tokens")):
            v = rec.get(key)
            if isinstance(v, int):
                f[k] += v
        t = rec.get("ttft_us")
        if isinstance(t, (int, float)):
            f["ttft_us"] += t
            f["ttft_n"] += 1
    family_rows = sorted((
        {"family": f"{fam:016x}", "requests": f["n"], "outcomes": dict(f["outcomes"]),
         "reused_tokens": f["reused"], "replayed_tokens": f["replayed"],
         "mean_ttft_us": (f["ttft_us"] / f["ttft_n"]) if f["ttft_n"] else None}
        for fam, f in families.items()), key=lambda r: -r["requests"])

    evaluated = agreements + len(disagreements)
    report = {
        "records":            n,
        "by_schema":          dict(by_schema),
        "outcomes":           dict(outcomes),
        "chosen":             dict(chosen),
        "selections":         dict(selections),
        "inventory_states":   dict(inv_states),
        "restore_failed":     restore_failed,
        "shadow_evaluated":   evaluated,
        "shadow_unavailable": shadow_missing,
        "agreement_rate":     (agreements / evaluated) if evaluated else None,
        "disagreements":      len(disagreements),
        "disagreement_items": disagreements[: args.max_disagreements],
        "prefix_families":    family_rows,
        "residual_us": {
            "n":    len(residuals),
            "mean": (sum(residuals) / len(residuals)) if residuals else None,
            "min":  min(residuals) if residuals else None,
            "max":  max(residuals) if residuals else None,
        },
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"records={n} schemas={dict(by_schema)}")
        print(f"outcomes={dict(outcomes)}")
        print(f"chosen={dict(chosen)} selections={dict(selections)}")
        print(f"inventory_states={dict(inv_states)}")
        print(f"shadow: evaluated={evaluated} unavailable={shadow_missing} "
              f"agreement_rate={report['agreement_rate']}")
        print(f"disagreements={len(disagreements)}")
        for d in report["disagreement_items"]:
            print(f"  task={d['id_task']} shipped={d['shipped_provider']}#{d['shipped_candidate']} "
                  f"shadow={d['shadow_provider']}#{d['shadow_choice']} "
                  f"saving_us={d['predicted_saving_us']}")
        r = report["residual_us"]
        print(f"residual_us: n={r['n']} mean={r['mean']} min={r['min']} max={r['max']}")
        print(f"prefix_families={len(family_rows)}")
        for f in family_rows[:8]:
            print(f"  family={f['family']} n={f['requests']} outcomes={f['outcomes']} "
                  f"reused={f['reused_tokens']} replayed={f['replayed_tokens']} "
                  f"mean_ttft_ms={(f['mean_ttft_us'] or 0)/1000:.0f}")

    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
