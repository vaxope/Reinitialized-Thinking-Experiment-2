"""Rewards for solver-only GRPO (Stage A). Pure Python so tests run anywhere.

r_k = 1[correct_k] * (1 - lam * len_score_k) + fmt_bonus * 1[well-formed <analysis>]
len_score_k = (l_k - l_min) / (l_max - l_min) over the correct rollouts of one problem; 0 if < 2 correct.
"""
import re
from fractions import Fraction

ANSWER_RE = re.compile(r"^\W*Answer\s*[:：]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
NUM_RE = re.compile(
    r"[-−]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?"           # 1,234.5 or 16/7
    r"(?:\s*(?:[eE]|[x×*]\s*10\s*\^?\s*\{?)\s*[-−+]?\d+\}?)?"       # e-3, x10^3, ×10^{-4}
    r"|[-−]?\.\d+"
)
MC_RE = re.compile(r"^\(?([a-eA-E])\)?\.?$")
MC_PRED_RE = re.compile(r"^\(?([a-eA-E])\)?(?![A-Za-z])")  # "b", "(b) 4.0 L", "B. text"
NUMERIC = {"Float", "Integer", "Fraction", "Percentage"}
REL_TOL = 1e-2  # WebInstruct golds are rounded to ~3 significant figures


def extract_answer(text):
    """Text after the last 'Answer:' line, stripped of markdown; None if absent."""
    m = ANSWER_RE.findall(text or "")
    return m[-1].strip("*_ .$") if m else None


SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")


def _num_str(s):
    """First number in s, normalised to Python float syntax (or 'a/b'); None if none."""
    s = (s or "").replace("−", "-").replace("\\times", "×").replace("\\cdot", "×")
    s = re.sub(r"10([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)", lambda m: "10^" + m.group(1).translate(SUPERSCRIPT), s)
    m = NUM_RE.search(s)
    if not m:
        return None
    t = re.sub(r"\s+", "", m.group(0)).replace(",", "").replace("{", "").replace("}", "")
    return re.sub(r"[x×*]10\^?", "e", t)


def half_ulp(s):
    """Half a unit in the last significant digit of s: 1.3e3 -> 50, 0.174 -> 0.0005, 16/7 -> 0."""
    t = _num_str(s)
    if t is None or "/" in t:
        return 0.0
    mant, _, exp = t.lower().partition("e")
    decimals = len(mant.partition(".")[2])
    return 0.5 * 10 ** (int(exp or 0) - decimals)


def parse_number(s):
    """First number in s as float (handles 1,234 / 16/7 / 3.2e-4 / 3.2 x 10^-4); None if none."""
    t = _num_str(s)
    if t is None:
        return None
    try:
        if "/" in t:
            a, b = t.split("/")
            return float(Fraction(a) / Fraction(b)) if float(b) else None
        return float(t)
    except (ValueError, ZeroDivisionError):
        return None


def gold_ok(gold, answer_type):
    """True if this gold answer can be checked automatically (used by data_prep)."""
    if answer_type == "Multiple Choice":
        return bool(MC_RE.match(gold.strip()))
    # one number only: drops golds like "0.377 or 37.7%" or "72% dark energy, 23% ..."
    return answer_type in NUMERIC and len(NUM_RE.findall(gold)) == 1 and parse_number(gold) is not None


def _close(p, g, tol=0.0):
    """Within 1%, or within the gold's own rounding (tol = half its last digit)."""
    return abs(p - g) <= max(REL_TOL * abs(g), tol) or abs(p - g) < 1e-12


def is_correct(pred, gold, answer_type):
    if pred is None:
        return False
    if answer_type == "Multiple Choice":
        p, g = MC_PRED_RE.match(pred.strip()), MC_RE.match(gold.strip())
        return bool(p and g and p.group(1).lower() == g.group(1).lower())
    p, g = parse_number(pred), parse_number(gold)
    if p is None or g is None:
        return False
    tol = 0.0 if answer_type == "Integer" else half_ulp(gold)  # integers are exact counts, not rounded
    if answer_type == "Percentage":  # 37.7% and 0.377 are both accepted
        return _close(p, g, tol) or _close(p * 100, g, tol)
    return _close(p, g, tol)


def well_formed_analysis(text):
    """Exactly one non-empty <analysis>...</analysis>, with the Answer line after it."""
    text = text or ""
    if text.count("<analysis>") != 1 or text.count("</analysis>") != 1:
        return False
    body, _, after = text.partition("<analysis>")[2].partition("</analysis>")
    return bool(body.strip()) and extract_answer(after) is not None


def len_scores(lengths, correct):
    ok = [l for l, c in zip(lengths, correct) if c]
    if len(ok) < 2 or max(ok) == min(ok):
        return [0.0] * len(lengths)
    lo, hi = min(ok), max(ok)
    return [(l - lo) / (hi - lo) if c else 0.0 for l, c in zip(lengths, correct)]


def trl_correct_reward(lam=0.0):
    """TRL reward function: correctness (+ optional length penalty), grouped by problem id."""
    def correct(completions, completion_ids, id, answer, answer_type, **_):
        texts = [c[0]["content"] for c in completions]
        groups = {}
        for i, pid in enumerate(id):
            groups.setdefault(pid, []).append(i)
        r = [0.0] * len(texts)
        for idx in groups.values():
            vals = group_rewards([texts[i] for i in idx], [len(completion_ids[i]) for i in idx],
                                 answer[idx[0]], answer_type[idx[0]], lam=lam)
            for i, v in zip(idx, vals):
                r[i] = v
        return r
    return correct


def analysis_format(completions, **_):
    """TRL reward function: 1 for a well-formed <analysis> block."""
    return [float(well_formed_analysis(c[0]["content"])) for c in completions]


def group_rewards(texts, lengths, gold, answer_type, lam=0.0, fmt_bonus=0.0):
    """Rewards for all rollouts of one problem."""
    correct = [is_correct(extract_answer(t), gold, answer_type) for t in texts]
    ls = len_scores(lengths, correct)
    return [float(c) * (1 - lam * s) + (fmt_bonus if fmt_bonus and well_formed_analysis(t) else 0.0)
            for t, c, s in zip(texts, correct, ls)]
