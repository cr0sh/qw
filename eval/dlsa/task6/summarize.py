#!/usr/bin/env python3
"""Summarize the task-6 `real_dlsa_bench` reports (docs/dlsa.md section 15) into markdown tables.

usage: summarize.py [directory of t6_*.json reports]   (default: this file's directory)

Main runs are t6_<T>_off<offset>_r<n>_*.json; *_hyb_* (dense_below 12288) and *_diag_*
(dense, mlx, metal on the divergent documents) get their own tables.
"""
import glob, json, statistics, sys, os

R = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
files = sorted(f for f in glob.glob(f"{R}/t6_*.json") if "_smoke_" not in f and "_hyb_" not in f and "_diag_" not in f)
runs = []  # (context, offset, tag, mode dict, report)
for f in files:
    d = json.load(open(f))
    for res in d["results"]:
        for m in res["modes"]:
            runs.append((res["context"], d["prompt_offset"], os.path.basename(f), m, d))

def by(ctx, mode):
    return [r for r in runs if r[0] == ctx and r[3]["mode"] == mode]

def med(v):
    return statistics.median(v) if v else float("nan")

def spread(v):
    return f"{min(v):.0f}–{max(v):.0f}" if len(v) > 1 else "—"

ctxs = sorted({r[0] for r in runs})
GiB = 1 << 30
out = []
P = out.append

P("### Prefill\n")
P("| context | prompt | reps | dense tok/s median (range) | metal tok/s median (range) | metal / dense | TTFT dense / metal s | peak GiB dense / metal | index B/token |")
P("|---|---|---|---|---|---|---|---|---|")
for c in ctxs:
    d, m = by(c, "dense"), by(c, "metal")
    if not d or not m:
        continue
    dt = [r[3]["prefill"]["tok_s"] for r in d]
    mt = [r[3]["prefill"]["tok_s"] for r in m]
    # paired ratio per rep (same file)
    pairs = {}
    for r in d + m:
        pairs.setdefault(r[2], {})[r[3]["mode"]] = r[3]["prefill"]["tok_s"]
    ratios = [p["metal"] / p["dense"] for p in pairs.values() if "metal" in p and "dense" in p]
    P(f"| {c} | {d[0][3]['prompt_tokens']} | {len(d)} | {med(dt):.1f} ({spread(dt)}) | {med(mt):.1f} ({spread(mt)}) | "
      f"{med(ratios):.3f} ({min(ratios):.3f}–{max(ratios):.3f}) | "
      f"{med([r[3]['prefill']['wall_s'] for r in d]):.1f} / {med([r[3]['prefill']['wall_s'] for r in m]):.1f} | "
      f"{med([r[3]['prefill']['peak_memory_bytes'] for r in d])/GiB:.1f} / {med([r[3]['prefill']['peak_memory_bytes'] for r in m])/GiB:.1f} | "
      f"{med([r[3]['prefill']['index_bytes_per_token'] for r in m]):.0f} |")

P("\n### Raw prefill tok/s per run\n")
P("| context | file | dense | metal | metal-pca | swap before→after (metal) |")
P("|---|---|---|---|---|---|")
seen = {}
for r in runs:
    seen.setdefault((r[0], r[2]), {})[r[3]["mode"]] = r[3]
for (c, f), ms in sorted(seen.items()):
    def tk(k):
        return f"{ms[k]['prefill']['tok_s']:.1f}" if k in ms else "—"
    sw = ms.get("metal", {})
    swb = (sw.get("swap_before") or "").split("used = ")[-1].split(" ")[0]
    swa = (sw.get("swap_after") or "").split("used = ")[-1].split(" ")[0]
    P(f"| {c} | {f} | {tk('dense')} | {tk('metal')} | {tk('metal-pca')} | {swb}→{swa} |")

P("\n### Decode (baseline greedy, ms/token median over 32 steps; median of reps)\n")
P("| context | dense ms | metal ms | metal / dense time | tokens equal (of 33) |")
P("|---|---|---|---|---|")
for c in ctxs:
    d, m = by(c, "dense"), by(c, "metal")
    if not d or not m:
        continue
    dd = med([r[3]["decode"]["median_ms"] for r in d])
    mm = med([r[3]["decode"]["median_ms"] for r in m])
    eq = [r[3]["decode"]["tokens_equal_reference_prefix"] for r in m]
    P(f"| {c} | {dd:.1f} | {mm:.1f} | {mm/dd:.3f} | {', '.join(str(e) for e in eq)} |")

P("\n### Speculative decoders (EOS suppressed, 64 tokens)\n")
P("| context | decoder | prefill s dense / metal | decode ms/token dense / metal | acceptance dense / metal |")
P("|---|---|---|---|---|")
for c in ctxs:
    for dec in ["mtp", "dflash2"]:
        row = {}
        for mode in ["dense", "metal"]:
            for r in by(c, mode):
                for s in r[3]["spec"]:
                    if s["decoder"] == dec:
                        row[mode] = s
        if "dense" in row and "metal" in row:
            a, b = row["dense"], row["metal"]
            P(f"| {c} | {dec} | {a['prefill_s']:.1f} / {b['prefill_s']:.1f} | {a['decode_ms_per_token']:.1f} / {b['decode_ms_per_token']:.1f} | {a['acceptance']:.3f} / {b['acceptance']:.3f} |")

P("\n### Quality: teacher-forced tail (255 scored rows per run) vs dense\n")
P("| context | mode | runs | ppl dense | ppl ratio | mean KL | p90 KL | max KL | top-1 agreement |")
P("|---|---|---|---|---|---|---|---|---|")
for c in ctxs:
    for mode in ["metal", "metal-pca"]:
        rs = [r for r in by(c, mode) if r[3]["tail"] and "vs_reference" in r[3]["tail"]]
        if not rs:
            continue
        v = [r[3]["tail"]["vs_reference"] for r in rs]
        P(f"| {c} | {mode} | {len(rs)} | {', '.join(f'{x['reference_ppl']:.3f}' for x in v)} | "
          f"{', '.join(f'{x['ppl_ratio']:.4f}' for x in v)} | {', '.join(f'{x['mean_kl']:.4f}' for x in v)} | "
          f"{max(x['p90_kl'] for x in v):.4f} | {max(x['max_kl'] for x in v):.3f} | {', '.join(f'{x['top1_agreement']:.3f}' for x in v)} |")

P("\n### Needles (exact match) and prefix-hit TTFT\n")
P("| context | mode | found | same answer as dense | hit TTFT ms median | restore ms median |")
P("|---|---|---|---|---|---|")
for c in ctxs:
    for mode in ["dense", "metal", "metal-pca"]:
        ns = [n for r in by(c, mode) for n in r[3]["needles"]]
        if not ns:
            continue
        found = sum(n["exact_match"] for n in ns)
        same = [n["same_answer_as_reference"] for n in ns if n["same_answer_as_reference"] is not None]
        P(f"| {c} | {mode} | {found}/{len(ns)} | {sum(same)}/{len(same) if same else '—'} | "
          f"{med([n['hit_ttft_ms'] for n in ns]):.0f} | {med([n['restore_ms'] for n in ns]):.0f} |")
    missed = [(r[3]["mode"], n["vault"], n["answer"]) for r in runs if r[0] == c for n in r[3]["needles"] if not n["exact_match"]]
    if missed:
        P(f"| {c} | missed: {missed} | | | | |")

P("\n### Stage profile (metal, fenced last 2 prefill chunks, ms per full-attention layer; first run)\n")
groups = {
    "indexing/clustering": ["index_key_projection", "key_readback", "ward", "index_query_projection", "centroid_sync"],
    "scoring/selection": ["score_matmul", "device_select"],
    "attention read (K2)": ["read"],
    "qkv proj + rope + KV fetch + finish": ["input", "qkv_projection", "rotate", "kv_fetch", "finish"],
}
P("| context | " + " | ".join(groups) + " | attn stages sum | full-attn layer (incl. MLP) | dense full-attn layer | linear layers % fwd (metal) | fenced fwd s dense / metal |")
P("|---|" + "---|" * (len(groups) + 5))
for c in ctxs:
    m, d = by(c, "metal"), by(c, "dense")
    if not m or not d:
        continue
    p = m[0][3]["profile"]["prefill"]
    pd = d[0][3]["profile"]["prefill"]
    st = p["stages"]
    cells = [sum(st.get(k, {}).get("ms_per_layer", 0) for k in ks) for ks in groups.values()]
    P(f"| {c} | " + " | ".join(f"{x:.1f}" for x in cells) +
      f" | {p['dlsa_stage_sum']['ms_per_layer']:.1f} | {p['full_attention_layers']['ms_per_layer']:.1f} | {pd['full_attention_layers']['ms_per_layer']:.1f} | "
      f"{p['linear_layers']['pct_of_forward']:.0f}% | {pd['ms_per_forward']/1e3:.2f} / {p['ms_per_forward']/1e3:.2f} |")

P("\n### Snapshot bytes per token (needle runs)\n")
for c in ctxs:
    d, m = by(c, "dense"), by(c, "metal")
    ds = [r[3]["snapshot"]["bytes_per_token"] for r in d if r[3]["snapshot"]]
    ms_ = [r[3]["snapshot"]["bytes_per_token"] for r in m if r[3]["snapshot"]]
    if ds and ms_:
        P(f"- {c}: dense {med(ds):.0f} B/token, metal {med(ms_):.0f} B/token (+{med(ms_)-med(ds):.0f})")

therm = {r[3].get("thermal_before") for r in runs} | {r[3].get("thermal_after") for r in runs}
P(f"\nthermal states seen: {therm}")
P("\n### dense_below = 12288 (hyb runs)\n")
P("| context | dense tok/s | metal tok/s | ratio | decode dense / metal ms | tail mean KL | needles |")
P("|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(f"{R}/t6_*_hyb_*.json"), key=lambda f: json.load(open(f))["results"][0]["context"]):
    res = json.load(open(f))["results"][0]
    ms = {m["mode"]: m for m in res["modes"]}
    d, m = ms["dense"], ms["metal"]
    kl = (m["tail"].get("vs_reference") or {}).get("mean_kl", float("nan"))
    P(f"| {res['context']} | {d['prefill']['tok_s']:.1f} | {m['prefill']['tok_s']:.1f} | {m['prefill']['tok_s']/d['prefill']['tok_s']:.3f} | "
      f"{d['decode']['median_ms']:.1f} / {m['decode']['median_ms']:.1f} | {kl:.4f} | {sum(n['exact_match'] for n in m['needles'])}/{len(m['needles'])} |")

P("\n### Diagnostics (diag runs)\n")
P("| context | offset | mode | prefill tok/s | ppl | mean KL | max KL | top-1 | rows KL > 1 | needles |")
P("|---|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(f"{R}/t6_*_diag_*.json")):
    d = json.load(open(f))
    res = d["results"][0]
    for m in res["modes"]:
        v = m["tail"].get("vs_reference") or {}
        big = sum(1 for x in v.get("row_kl", []) if x > 1)
        P(f"| {res['context']} | {d['prompt_offset']} | {m['mode']} | {m['prefill']['tok_s']:.1f} | {m['tail']['ppl']:.3f} | "
          f"{v.get('mean_kl', 0):.4f} | {v.get('max_kl', 0):.2f} | {v.get('top1_agreement', 1):.3f} | {big if v else '—'} | "
          f"{sum(n['exact_match'] for n in m['needles'])}/{len(m['needles'])} |")
print("\n".join(out))
