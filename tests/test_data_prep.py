import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from data_prep import ngrams

SPAN = "a b c d e f g h i j k l m"  # 13 words


def test_ngram_overlap_detected_through_case_and_punctuation():
    assert ngrams(f"Intro. {SPAN.upper()}, end") & ngrams(f"other {SPAN} words")


def test_short_overlap_ignored():
    assert not ngrams("x " + SPAN[:-2]) & ngrams(SPAN[:-2] + " y")
