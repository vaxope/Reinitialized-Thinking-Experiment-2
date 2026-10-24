import json, random, statistics as st, sys
from collections import defaultdict
from pathlib import Path

rows = [json.loads(l) for l in (Path(sys.argv[1]).expanduser() / "raw.jsonl").open(encoding="utf-8")]


def boot_ci(xs, n=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


by = defaultdict(list)
for r in rows:
    by[(r["model"], r["arm"])].append(r)

print(f"{'model':12s} {'arm':8s} {'n':>4s} {'acc':>6s} {'95% CI':>15s} {'tok mean':>9s} "
      f"{'tok med':>8s} {'tok p90':>8s} {'trunc':>6s} {'noparse':>8s} {'tok/s':>6s}")
for (m, arm), rs in sorted(by.items()):
    acc = [1.0 if r["correct"] else 0.0 for r in rs]
    toks = sorted(r["eval_count"] or 0 for r in rs)
    lo, hi = boot_ci(acc)
    tps = [r["tok_per_s"] for r in rs if r["tok_per_s"]]
    print(f"{m:12s} {arm:8s} {len(rs):4d} {st.mean(acc):6.3f} [{lo:.3f}, {hi:.3f}] "
          f"{st.mean(toks):9.0f} {st.median(toks):8.0f} {toks[int(0.9 * (len(toks) - 1))]:8.0f} "
          f"{sum(r['truncated'] for r in rs):6d} {sum(r['pred'] is None for r in rs):8d} "
          f"{(st.mean(tps) if tps else 0):6.1f}")

# Paired comparison per model: same question and sample, think vs nothink.
print("\nPaired (same question + shuffle):")
for m in sorted({r["model"] for r in rows}):
    t = {(r["qid"], r["sample"]): r["correct"] for r in rows if r["model"] == m and r["arm"] == "think"}
    n = {(r["qid"], r["sample"]): r["correct"] for r in rows if r["model"] == m and r["arm"] == "nothink"}
    keys = t.keys() & n.keys()
    only_t = sum(t[k] and not n[k] for k in keys)
    only_n = sum(n[k] and not t[k] for k in keys)
    both = sum(t[k] and n[k] for k in keys)
    print(f"  {m}: pairs={len(keys)}  both right={both}  only think={only_t}  only no-think={only_n}")

# Accuracy by domain.
print("\nBy domain:")
dom = defaultdict(list)
for r in rows:
    dom[(r["model"], r["arm"], r["domain"])].append(r["correct"])
for k, v in sorted(dom.items()):
    print(f"  {k[0]:12s} {k[1]:8s} {k[2]:10s} n={len(v):3d} acc={sum(v) / len(v):.3f}")