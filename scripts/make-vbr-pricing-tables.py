#!/usr/bin/env python3
# DEFINITIVE per-model pricing tables + baked runtime degrade orders (2026-07-05).
# Methodology applied (knowledge/measurement-methods.md §2/§8):
#   - statistic per model chosen by bench verdict + split-half reliability (mean excluded);
#   - fp16->t8 band forced to the frac lens (the only statistic that resolves that rung);
#   - orders = fp16->t8 band first (runtime contract), then price-per-bit water-fill over
#     t8->t4->t3tcq->t2tcq->t1tcq in the model's recommended lens;
#   - reliability published next to every price column.
# Outputs: pricing_tables/<m>_prices.tsv, pricing_tables/README.md,
#          llama-vbr-degrade-orders.inc (arch-keyed registry for the runtime).
import csv, os
from collections import defaultdict

D = os.path.dirname(os.path.abspath(__file__))
BITS = {"t8": 8.125, "t4": 4.125, "t3tcq": 3.25, "t2tcq": 2.25, "t1tcq": 1.25}
CHAIN = [("t8", "t4"), ("t4", "t3tcq"), ("t3tcq", "t2tcq"), ("t2tcq", "t1tcq")]
TIER_ENUM = {"t8": "VBR_TIER_T8", "t4": "VBR_TIER_T4", "t3tcq": "VBR_TIER_T3_TCQ",
             "t2tcq": "VBR_TIER_T2_TCQ", "t1tcq": "VBR_TIER_T1_TCQ"}
# recommended lens per model: bench verdict where decisive, else best split-half reliability
# (measurement-methods.md §8d). Evidence: q27 flip-bench winner=trim1 (margins top-2);
# g12 bench frac dominant; g26 bench median wins b2.5+b3.5; g31 reliability .87-.93 median
# (bench partial: median won b2.5); moe bench tie -> frac = highest reliability (.93-.99).
RECOMMENDED = {"q27": "trim1", "g12": "frac", "g26": "median", "g31": "median", "moe": "frac"}
LENS_COL = {"median": "excess_median", "trim1": "excess_trim0.01", "frac": "frac_gt_0.001",
            "mean": "excess_mean"}
ARCH = {  # (llm_arch enum, n_layer) registry keys
    "q27": ("LLM_ARCH_QWEN35", 64), "moe": ("LLM_ARCH_QWEN35MOE", 40),
    "g12": ("LLM_ARCH_GEMMA4", 48), "g26": ("LLM_ARCH_GEMMA4", 30), "g31": ("LLM_ARCH_GEMMA4", 60),
}

def load(tag):
    prices = defaultdict(dict); rel = {}
    layers = set()
    for r in csv.DictReader(open(f"{D}/matrix_v3_cells.tsv"), delimiter="\t"):
        if r["tag"] != tag or r["group"] != "all" or r["side"] == "kv": continue
        key = (int(r["layer"]), r["side"], r["transition"])
        layers.add(int(r["layer"]))
        for lens, col in LENS_COL.items():
            prices[lens][key] = float(r[col])
    hp = f"{D}/hazard_{tag}_cells.tsv"
    if os.path.exists(hp):
        for r in csv.DictReader(open(hp), delimiter="\t"):
            if r["group"] != "all" or r["side"] == "kv": continue
            prices["flip"][(int(r["layer"]), r["side"], r["transition"])] = float(r["flip_excess"])
    for r in csv.DictReader(open(f"{D}/matrix_v3_reliability.tsv"), delimiter="\t"):
        if r["tag"] != tag: continue
        rel[(r["transition"], r["side"], r["stat"])] = float(r["rho_half"])
    return prices, sorted(layers), rel

def band_order(prices, layers, lens, tr):
    # one transition: cheapest (least protected) first, both sides
    items = []
    for l in layers:
        for s in ("k", "v"):
            p = prices[lens].get((l, s, tr))
            if p is not None: items.append((max(p, 0.0), l, s))
    items.sort()
    return [(l, s) for _, l, s in items]

def waterfill(prices, layers, lens):
    state = {(l, s): "t8" for l in layers for s in ("k", "v")}
    order = []
    while True:
        best, bkey = None, None
        for (l, s), cur in state.items():
            nxt = dict(CHAIN).get(cur)
            if nxt is None: continue
            p = prices[lens].get((l, s, f"{cur}-{nxt}"))
            if p is None: continue
            key = max(p, 0.0) / (BITS[cur] - BITS[nxt])
            if best is None or key < best: best, bkey = key, (l, s, cur, nxt)
        if bkey is None: break
        l, s, cur, nxt = bkey
        order.append((l, s, nxt))
        state[(l, s)] = nxt
    return order

os.makedirs(f"{D}/pricing_tables", exist_ok=True)
inc = ["// GENERATED 2026-07-05 by make_pricing_tables.py from matrix v3 (box-2, 3169 cells,",
       "// deployment-true tap config, paired-anchor dumps, reliability-gated statistics).",
       "// Per-model lens: bench-validated or best split-half reliability (see pricing_tables/).",
       "// fp16->t8 band = frac lens (the only statistic that resolves that rung).",
       "// Runtime override: VBR_DEGRADE_ORDER=<file>, tokens \"<il><k|v>:<t8|t4|t3|t2|t1>\"."]
registry = []
readme = ["# Pricing tables (matrix v3, 2026-07-05)\n",
          "Per model: all-lens prices per (transition, side, layer) with split-half reliability;",
          "chosen lens per the methodology doc §8d. Floors: see margin_findings.md.\n"]

for tag in ("q27", "moe", "g12", "g26", "g31"):
    prices, layers, rel = load(tag)
    lens = RECOMMENDED[tag]
    # 1. prices tsv
    with open(f"{D}/pricing_tables/{tag}_prices.tsv", "w") as f:
        f.write("transition\tside\tlayer\tmedian\ttrim1\tfrac\tflip\tmean\trho_half_median\trho_half_frac\tchosen_lens\n")
        for tr in ["fp16-t8"] + [f"{a}-{b}" for a, b in CHAIN]:
            chosen = "frac" if tr == "fp16-t8" else lens
            for s in ("k", "v"):
                for l in layers:
                    row = [prices[ln].get((l, s, tr)) for ln in ("median", "trim1", "frac", "flip", "mean")]
                    if row[0] is None and row[2] is None: continue
                    f.write(f"{tr}\t{s}\t{l}\t" + "\t".join("" if v is None else f"{v:.9g}" for v in row)
                            + f"\t{rel.get((tr, s, 'excess_median'), float('nan')):.2f}"
                            + f"\t{rel.get((tr, s, 'frac_gt_0.001'), float('nan')):.2f}\t{chosen}\n")
    # 2. runtime order: fp16->t8 band (frac) + water-fill (recommended lens)
    steps = [(l, s, "t8") for l, s in band_order(prices, layers, "frac", "fp16-t8")]
    steps += waterfill(prices, layers, lens)
    arr = f"vbr_order_{tag}"
    inc.append(f"\n// {tag}: lens={lens} (fp16->t8 band: frac), {len(steps)} steps, layers={len(layers)}")
    inc.append(f"static const vbr_degrade_step {arr}[] = {{")
    for i in range(0, len(steps), 6):
        inc.append("    " + " ".join(f"{{{l:2d}, {1 if s == 'v' else 0}, {TIER_ENUM[t]}}},"
                                     for l, s, t in steps[i:i+6]))
    inc.append("};")
    arch, nl = ARCH[tag]
    registry.append(f"    {{ {arch}, {nl}, {arr}, sizeof({arr})/sizeof({arr}[0]) }},  // {tag}")
    readme.append(f"- **{tag}**: lens={lens}, {len(steps)} baked steps, {len(layers)} KV layers "
                  f"(arch {arch}/n_layer {nl})")

inc.append("\nstruct vbr_baked_orders_entry { llm_arch arch; uint32_t n_layer; const vbr_degrade_step * steps; size_t n; };")
inc.append("static const vbr_baked_orders_entry vbr_baked_orders[] = {")
inc += registry
inc.append("};")
open(f"{D}/llama-vbr-degrade-orders.inc", "w").write("\n".join(inc) + "\n")
open(f"{D}/pricing_tables/README.md", "w").write("\n".join(readme) + "\n")
print("wrote pricing_tables/ + llama-vbr-degrade-orders.inc")
for tag in RECOMMENDED: print(f"  {tag}: lens={RECOMMENDED[tag]}")
