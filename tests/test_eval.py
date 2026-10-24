import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gpqa_eval import build, extract

ITEM = {"qid": "recX", "question": "Q?", "correct": "right", "wrong": ["w1", "w2", "w3"]}


def test_extract_last_answer_wins():
    assert extract("Answer: A\n...reconsider...\nAnswer: C") == "C"


def test_extract_formats():
    assert extract("**Answer: (b)**") == "B"
    assert extract("Answer：D") == "D"


def test_extract_missing():
    assert extract("I think it is B.") is None
    assert extract("") is None and extract(None) is None


def test_shuffle_deterministic_and_gold_correct():
    p1, g1, o1 = build(ITEM, 0)
    p2, g2, o2 = build(ITEM, 0)
    assert (p1, g1, o1) == (p2, g2, o2)
    assert f"{g1}) right" in p1
    assert sorted(o1) == [0, 1, 2, 3] and o1["ABCD".index(g1)] == 0
