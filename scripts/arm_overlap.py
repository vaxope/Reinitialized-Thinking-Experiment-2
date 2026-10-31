"""Cross-arm analysis of GPQA runs: which arms solve which questions, answer agreement, and how
similar the trajectories are. Prints aggregates and question ids only (no GPQA text), as markdown.

Usage: python scripts/arm_overlap.py runs/RUN-002-full runs/RUN-008-gar-pilot > runs/arm_overlap.md
"""
import json, re, statistics as st, sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SHORT = {"qwen3.5:2b": "2B", "qwen3.5:4b-q8_0": "4B", "gar-2b-trained": "2B"}  # pass one gar run: untrained (RUN-008) or trained (RUN-010)
GROUPS = {"2B (4 arms)": ["2B think", "2B nothink", "2B gar", "2B direct"],
          "2B reasoning (3 arms)": ["2B think", "2B nothink", "2B gar"],
          "4B (3 arms)": ["4B think", "4B nothink", "4B direct"],
          "All 7 arms": ["2B think", "2B nothink", "2B gar", "2B direct", "4B think", "4B nothink", "4B direct"]}
REASONING = ["2B think", "2B nothink", "2B gar", "4B think", "4B nothink"]
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
# phrases where the model commits to a letter mid-reasoning ("the answer is B", "option C is correct")
COMMIT_RE = re.compile(r"(?:answer is|answer would be|correct (?:answer|option|choice) is|choose|select|go with)\s*"
                       r"(?:option\s*)?\**\(?([ABCD])\)?\b|(?:option|choice)\s*\(?([ABCD])\)?\s*is\s*(?:the\s*)?correct", re.I)
WAIT_RE = re.compile(r"\bwait\b", re.I)
# appeals to a remembered source ("the answer key says B"); the model cannot actually look these up
SOURCE_RE = re.compile(r"answer key|official (?:answer|solution)|known (?:problem|question)|this (?:specific|exact|particular) (?:question|problem)"
                       r"|past paper|olympiad|\bJEE\b|\bGATE\b|standard answer|accepted answer|textbook answer", re.I)
GUESS_RE = re.compile(r"I'?ll go with|best guess|educated guess|I (?:will|would) guess|have to (?:choose|pick|guess)|most likely intended", re.I)
VERIFY_RE = re.compile(r"double[- ]check|re-?check|verify|let me re-?(?:calculate|compute|evaluate|read|consider)|re-?examine", re.I)


def load(dirs):
    A = defaultdict(dict)
    for d in dirs:
        for line in open(Path(d) / "raw.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["sample"] == 0:
                A[f"{SHORT[r['model']]} {r['arm']}"][r["qid"]] = r
    return A


def reasoning(r):
    return (r["thinking"] or "") + "\n" + (r["content"] or "")


def commits(text):
    return [a or b for a, b in COMMIT_RE.findall(text)]


def pct(x, n):
    return f"{x}/{n} ({100 * x / n:.0f}%)" if n else "-"


def main():
    A = load(sys.argv[1:])
    Q = sorted(set.intersection(*(set(v) for v in A.values())))
    dom = {q: A["2B think"][q]["domain"] for q in Q}
    gold = {q: A["2B think"][q]["gold"] for q in Q}
    out = [f"# Cross-arm analysis ({len(Q)} questions, sample 0)\n", "## Accuracy\n",
           "| Arm | Accuracy | Median tokens |", "|---|---|---|"]
    for k in GROUPS["All 7 arms"]:
        out.append(f"| {k} | {sum(A[k][q]['correct'] for q in Q) / len(Q):.3f} | {st.median(A[k][q]['eval_count'] for q in Q):.0f} |")

    for name, arms in GROUPS.items():
        pat = {q: tuple(A[k][q]["correct"] for k in arms) for q in Q}
        n_ok = Counter(sum(p) for p in pat.values())
        out += [f"\n## {name}: {', '.join(arms)}\n", "| Arms correct | Questions | By domain (Chem / Phys / Bio) |", "|---|---|---|"]
        for c in range(len(arms) + 1):
            qs = [q for q in Q if sum(pat[q]) == c]
            dc = Counter(dom[q] for q in qs)
            out.append(f"| {c} of {len(arms)} | {len(qs)} | {dc['Chemistry']} / {dc['Physics']} / {dc['Biology']} |")
        out.append(f"\nAny arm correct (oracle): {pct(sum(any(p) for p in pat.values()), len(Q))}")
        plural = 0
        for q in Q:
            votes = Counter(A[k][q]["pred"] for k in arms if A[k][q]["pred"])
            top = votes.most_common(2)
            plural += bool(top) and (len(top) == 1 or top[0][1] > top[1][1]) and top[0][0] == gold[q]
        out.append(f"Plurality vote (ties count wrong): {pct(plural, len(Q))}\n")
        out += ["| Pattern (" + ", ".join(arms) + ") | Questions | Ids |", "|---|---|---|"]
        for p, n in sorted(Counter(pat.values()).items(), key=lambda x: (-sum(x[0]), x[0])):
            ids = [q for q in Q if pat[q] == p]
            out.append(f"| {''.join('✓' if x else '·' for x in p)} | {n} | {', '.join(ids) if n <= 12 else '(' + str(n) + ' ids)'} |")
        wrong = [q for q in Q if not any(pat[q])]
        if wrong:
            letters = [Counter(A[k][q]["pred"] for k in arms) for q in wrong]
            same = sum(len(c) == 1 and None not in c for c in letters)
            top_share = st.mean(c.most_common(1)[0][1] / len(arms) for c in letters)
            out.append(f"\nAll-wrong questions: {len(wrong)}. Every arm picked the same wrong letter on {pct(same, len(wrong))}; "
                       f"mean share of arms on the most common wrong letter {top_share:.2f}.")

    out += ["\n## Pairwise answer agreement\n",
            "Same letter on all questions; same wrong letter given both wrong (uniform chance would be 0.33); correctness agreement.\n",
            "| Pair | Same letter | Both wrong → same wrong letter | Both right | Only first | Only second | Both wrong |", "|---|---|---|---|---|---|---|"]
    for a, b in combinations(GROUPS["All 7 arms"], 2):
        same = sum(A[a][q]["pred"] == A[b][q]["pred"] for q in Q)
        bw = [q for q in Q if not A[a][q]["correct"] and not A[b][q]["correct"]]
        sw = sum(A[a][q]["pred"] == A[b][q]["pred"] and A[a][q]["pred"] is not None for q in bw)
        c = Counter((A[a][q]["correct"], A[b][q]["correct"]) for q in Q)
        out.append(f"| {a} vs {b} | {same / len(Q):.2f} | {pct(sw, len(bw))} | {c[(True, True)]} | {c[(True, False)]} | {c[(False, True)]} | {c[(False, False)]} |")

    out += ["\n## Trajectory features (reasoning = thinking + reply)\n",
            "| Arm | Median tokens, correct / wrong | 'wait' per reply, median | Verify phrases, median | Replies committing to ≥2 different letters | First committed letter = final |",
            "|---|---|---|---|---|---|"]
    for k in REASONING:
        rs = [A[k][q] for q in Q]
        ok = [r["eval_count"] for r in rs if r["correct"]]; bad = [r["eval_count"] for r in rs if not r["correct"]]
        cm = [commits(reasoning(r)) for r in rs]
        multi = sum(len(set(c)) >= 2 for c in cm)
        first = [c[0] == r["pred"] for c, r in zip(cm, rs) if c and r["pred"]]
        out.append(f"| {k} | {st.median(ok):.0f} / {st.median(bad):.0f} | {st.median(len(WAIT_RE.findall(reasoning(r))) for r in rs):.0f} | "
                   f"{st.median(len(VERIFY_RE.findall(reasoning(r))) for r in rs):.0f} | {pct(multi, len(rs))} | {pct(sum(first), len(first))} |")
    out.append("\nAccuracy by whether the reasoning commits to ≥2 different letters (changes its mind):\n")
    out += ["| Arm | Changes mind: acc | Doesn't: acc |", "|---|---|---|"]
    for k in REASONING:
        m = [A[k][q] for q in Q if len(set(commits(reasoning(A[k][q])))) >= 2]
        s = [A[k][q] for q in Q if len(set(commits(reasoning(A[k][q])))) < 2]
        f = lambda rs: f"{sum(r['correct'] for r in rs) / len(rs):.2f} (n={len(rs)})" if rs else "-"
        out.append(f"| {k} | {f(m)} | {f(s)} |")

    out += ["\nAppeals to a remembered source (answer key, exam name, 'this specific question') and explicit guesses:\n",
            "| Arm | Cites a source: share | acc with / without | Guesses: share | acc with / without |", "|---|---|---|---|---|"]
    for k in REASONING:
        f = lambda rs: f"{sum(r['correct'] for r in rs) / len(rs):.2f}" if rs else "-"
        row = [f"| {k}"]
        for rx in (SOURCE_RE, GUESS_RE):
            w = [A[k][q] for q in Q if rx.search(reasoning(A[k][q]))]; wo = [A[k][q] for q in Q if not rx.search(reasoning(A[k][q]))]
            row.append(f"{pct(len(w), len(Q))} | {f(w)} / {f(wo)}")
        out.append(" | ".join(row) + " |")
    pat3 = {q: tuple(A[k][q]["correct"] for k in GROUPS["2B reasoning (3 arms)"]) for q in Q}
    for p, k in (((False, False, True), "2B gar"), ((True, False, False), "2B think"), ((False, True, False), "2B nothink")):
        qs = [q for q in Q if pat3[q] == p]
        out.append(f"\nOnly {k} correct among 2B reasoning arms ({len(qs)} questions): it guessed explicitly in "
                   f"{sum(bool(GUESS_RE.search(reasoning(A[k][q]))) for q in qs)}, cited a source in "
                   f"{sum(bool(SOURCE_RE.search(reasoning(A[k][q]))) for q in qs)}, committed to ≥2 letters in "
                   f"{sum(len(set(commits(reasoning(A[k][q])))) >= 2 for q in qs)}.")
    allw = [q for q in Q if not any(A[k][q]["correct"] for k in REASONING)]
    one = sum(len({A[k][q]["pred"] for k in REASONING}) == 1 for q in allw)
    out.append(f"\nAll five reasoning arms (2B and 4B) wrong: {len(allw)} questions; all five on the same wrong letter: {one}.")

    # similarity of reasoning between arms, per question
    docs = {(k, q): reasoning(A[k][q]) for k in REASONING for q in Q}
    keys = list(docs)
    tf = TfidfVectorizer(sublinear_tf=True, min_df=2, token_pattern=r"(?u)\b[A-Za-z][A-Za-z]+\b|\d+(?:\.\d+)?").fit_transform(docs.values())
    idx = {k: i for i, k in enumerate(keys)}
    nums = {key: set(NUM_RE.findall(t)) for key, t in docs.items()}
    out += ["\n## Trajectory similarity between arms (same question)\n",
            "TF-IDF cosine of the reasoning text, and Jaccard overlap of the numbers written in it (a proxy for computing the same quantities). Medians.\n",
            "| Pair | Cosine: all | same answer | different answer | Numbers Jaccard: same answer | different answer |", "|---|---|---|---|---|---|"]
    for a, b in combinations(REASONING, 2):
        cos = {q: float(cosine_similarity(tf[idx[(a, q)]], tf[idx[(b, q)]])[0, 0]) for q in Q}
        jac = {q: len(nums[(a, q)] & nums[(b, q)]) / max(1, len(nums[(a, q)] | nums[(b, q)])) for q in Q}
        sa = [q for q in Q if A[a][q]["pred"] == A[b][q]["pred"]]; da = [q for q in Q if q not in set(sa)]
        m = lambda d, qs: f"{st.median(d[q] for q in qs):.2f}" if qs else "-"
        out.append(f"| {a} vs {b} | {m(cos, Q)} | {m(cos, sa)} (n={len(sa)}) | {m(cos, da)} (n={len(da)}) | {m(jac, sa)} | {m(jac, da)} |")
    # cross-question baseline: same arm pair, different questions
    base = [float(cosine_similarity(tf[idx[("2B gar", Q[i])]], tf[idx[("2B nothink", Q[(i + 1) % len(Q)])]])[0, 0]) for i in range(len(Q))]
    out.append(f"\nBaseline, 2B gar vs 2B nothink on *different* questions: median cosine {st.median(base):.2f}.")

    out += ["\n## Lengths by 2B reasoning pattern (think, nothink, gar)\n", "| Pattern | n | Median tokens think / nothink / gar |", "|---|---|---|"]
    arms = GROUPS["2B reasoning (3 arms)"]
    for p, n in sorted(Counter(tuple(A[k][q]["correct"] for k in arms) for q in Q).items(), key=lambda x: (-sum(x[0]), x[0])):
        qs = [q for q in Q if tuple(A[k][q]["correct"] for k in arms) == p]
        out.append(f"| {''.join('✓' if x else '·' for x in p)} | {n} | " + " / ".join(f"{st.median(A[k][q]['eval_count'] for q in qs):.0f}" for k in arms) + " |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
