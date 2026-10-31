"""Trajectory features of trained gar vs 2B think and nothink, overall and on the questions gar gets
wrong while think and nothink both get right. Prints aggregates and ids only (no GPQA text).

Usage: python scripts/gar_failures.py runs/RUN-002-full runs/RUN-010-gar-trained
"""
import re, statistics as st, sys

from arm_overlap import commits, load, reasoning

ARMS = ["2B think", "2B nothink", "2B gar"]
OPTION_RE = re.compile(r"\b(?:option|choice)\s*\(?([ABCD])\)?|\(([ABCD])\)|^\W*([ABCD])[\).:]", re.I | re.M)
HEUR_RE = re.compile(r"most logical (?:choice|answer|option)|best fit|intended answer|(?:MC|multiple[- ]choice) questions?"
                     r"|most likely (?:answer|choice|option|intended)|most plausible (?:answer|choice|option)|by elimination", re.I)
HEDGE_RE = re.compile(r"\b(?:likely|probably|perhaps|assume|assuming|presumably|seems|might be)\b", re.I)
RECALL_RE = re.compile(r"\b(?:recall|remember|known (?:as|that)|well[- ]known|standard result|typically|usually)\b", re.I)
# the derivation disagrees with the options and the reply explains the mismatch away instead of re-deriving
MISMATCH_RE = re.compile(r"\bflaw|\btypo|discrepanc|closest (?:option|match|answer|value|is|to)|none of the options|not an option"
                         r"|(?:doesn't|does not|don't|do not) match|if I (?:must|have to) (?:choose|pick)|assume the (?:intended )?answer is", re.I)
SANITY_RE = re.compile(r"\bdimension(?:al|s)?\b|\bunits? of\b|order of magnitude|faster than light|v\s*<\s*c\b|exceeds? (?:the speed of light|c\b)", re.I)
# a line pairing an option letter with a verdict, e.g. "B) ... (Incorrect)" or "Option C matches"
VERDICT_RE = re.compile(r"^\W*(?:option\s*)?\(?[ABCD]\)?[\s).:].*\b(?:incorrect|correct|matches|wrong|false|true|eliminat\w*|ruled out)\b", re.I | re.M)
VERIFY_RE = re.compile(r"double[- ]check|re-?check|verify|re-?calculat|re-?comput|re-?evaluat|let me re-?|\bwait\b", re.I)


def feats(r, gold):
    t = reasoning(r)
    c = commits(t)
    k = max(1, len(t) / 1000)  # per 1,000 characters
    return {"tokens": r["eval_count"], "options named": len({a or b or d for a, b, d in OPTION_RE.findall(t)}),
            "equations /1k ch": t.count("=") / k, "hedges /1k ch": len(HEDGE_RE.findall(t)) / k,
            "recall /1k ch": len(RECALL_RE.findall(t)) / k, "verify+wait /1k ch": len(VERIFY_RE.findall(t)) / k,
            "option verdict lines": len(VERDICT_RE.findall(t)),
            "explains a mismatch away": bool(MISMATCH_RE.search(t)), "sanity check (units, magnitude, v<c)": bool(SANITY_RE.search(t)),
            "MC heuristic used": bool(HEUR_RE.search(t)), "changes letter": len(set(c)) >= 2,
            "commits to gold at some point": gold in c}


def table(A, qs, gold, title):
    rows = {k: [feats(A[k][q], gold[q]) for q in qs] for k in ARMS}
    out = [f"\n## {title} (n={len(qs)})\n", "| Feature | " + " | ".join(ARMS) + " |", "|---|" + "---|" * len(ARMS)]
    for f in rows[ARMS[0]][0]:
        cells = []
        for k in ARMS:
            v = [x[f] for x in rows[k]]
            cells.append(f"{sum(v)}/{len(v)} ({100 * sum(v) / len(v):.0f}%)" if isinstance(v[0], bool) else f"{st.median(v):.2f}")
        out.append(f"| {f} | " + " | ".join(cells) + " |")
    out.append("| accuracy | " + " | ".join(f"{sum(A[k][q]['correct'] for q in qs) / len(qs):.2f}" for k in ARMS) + " |")
    return out


def main():
    A = load(sys.argv[1:])
    Q = sorted(set.intersection(*(set(A[k]) for k in ARMS)))
    gold = {q: A["2B think"][q]["gold"] for q in Q}
    ok = lambda k, q: A[k][q]["correct"]
    lost = [q for q in Q if not ok("2B gar", q) and ok("2B think", q) and ok("2B nothink", q)]
    out = ["# Trained gar failure analysis"]
    out += table(A, Q, gold, "All questions")
    out += table(A, lost, gold, "gar wrong, think and nothink right")
    out += table(A, [q for q in Q if all(ok(k, q) for k in ARMS)], gold, "All three right")
    out.append(f"\nids (gar wrong, think and nothink right): {', '.join(lost)}")
    rejected = sum(A["2B gar"][q]["pred"] in commits(reasoning(A["2B think"][q])) + commits(reasoning(A["2B nothink"][q])) for q in lost)
    out.append(f"\nOn these, gar's wrong letter is one that think or nothink committed to at some point (then rejected) in {rejected} of {len(lost)}.")
    print("\n".join(out))


if __name__ == "__main__":
    main()
