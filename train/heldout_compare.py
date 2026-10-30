"""Paired comparison of two heldout_eval.py tags. CPU only, stdlib only: runs on a login node.
Prints aggregates and problem ids, never problem text.

Usage: python train/heldout_compare.py <out>/base <out>/stageA-2b-gar [--formats gar nothink think]
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

BUCKETS = ((0, 1000), (1000, 2000), (2000, 4000), (4000, 10**9))


def load(d, fmt):
    by = defaultdict(list)
    for line in open(Path(d) / f"{fmt}.jsonl", encoding="utf-8"):
        r = json.loads(line); by[r["id"]].append(r)
    return by


def acc(rows):
    return f"{sum(r['correct'] for r in rows) / len(rows):.3f} (n={len(rows)})" if rows else "-"


def compare(A, B):
    assert A.keys() == B.keys(), "the two tags were run on different problems"
    fa, fb = [r for v in A.values() for r in v], [r for v in B.values() for r in v]
    lines = [f"  all: {acc(fa)} -> {acc(fb)}"]
    for key in ("category", "answer_type"):
        for v in sorted({r[key] for r in fa}):
            lines.append(f"  {key}={v}: {acc([r for r in fa if r[key] == v])} -> {acc([r for r in fb if r[key] == v])}")
    lines.append(f"  not truncated: {acc([r for r in fa if not r['truncated']])} -> {acc([r for r in fb if not r['truncated']])}")
    lines.append(f"  answered (pred not None): {acc([r for r in fa if r['pred'] is not None])} -> {acc([r for r in fb if r['pred'] is not None])}")
    for lo, hi in BUCKETS:
        pick = lambda rs: [r for r in rs if lo <= r["tokens"] < hi]
        lines.append(f"  tokens {lo}-{hi if hi < 10**9 else 'max'}: {acc(pick(fa))} -> {acc(pick(fb))}")
    pa = {i: sum(r["correct"] for r in A[i]) / len(A[i]) for i in A}
    pb = {i: sum(r["correct"] for r in B[i]) / len(B[i]) for i in B}
    d = {i: pb[i] - pa[i] for i in A}
    lines.append(f"  problems better / same / worse: {sum(x > 0 for x in d.values())} / "
                 f"{sum(x == 0 for x in d.values())} / {sum(x < 0 for x in d.values())}")
    lines.append(f"  base solved >=3/4 but trained <=1/4: {sum(pa[i] >= .75 and pb[i] <= .25 for i in A)}; "
                 f"reverse: {sum(pb[i] >= .75 and pa[i] <= .25 for i in A)}")
    lines.append(f"  largest drops (ids): {sorted(A, key=lambda i: d[i])[:8]}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--formats", nargs="+", default=["gar", "nothink", "think"])
    a = ap.parse_args()
    print(f"{Path(a.a).name} -> {Path(a.b).name}")
    for fmt in a.formats:
        print(f"== {fmt}"); print("\n".join(compare(load(a.a, fmt), load(a.b, fmt))))


if __name__ == "__main__":
    main()
