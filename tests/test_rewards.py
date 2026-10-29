import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from rewards import (analysis_format, extract_answer, gold_ok, group_rewards, half_ulp, is_correct, len_scores,
                     parse_number, trl_correct_reward, well_formed_analysis)

GAR_OK = "<analysis>\n1. F = ma\n2. 2*3\n3. units ok\n</analysis>\nAnswer: 6 N"


def test_extract_last_answer_line():
    assert extract_answer("Answer: 5\nwait\nAnswer: 7 m") == "7 m"
    assert extract_answer("**Answer: b**") == "b"
    assert extract_answer("the answer is 5") is None and extract_answer(None) is None


def test_parse_number_forms():
    assert parse_number("264,600N") == 264600
    assert parse_number("16/7 meters") == 16 / 7
    assert abs(parse_number("38.0*10^3 g") - 38000) < 1e-9
    assert abs(parse_number("4.5 × 10^{-4} Wb") - 4.5e-4) < 1e-15
    assert parse_number("3.2e-4") == 3.2e-4 and parse_number("−2.5") == -2.5
    assert parse_number("no number") is None
    assert parse_number("1.3 \\times 10^3") == 1300 and parse_number("2.0 \\cdot 10^{-2}") == 0.02
    assert abs(parse_number("2.27 × 10⁻⁵ mol*L⁻¹") - 2.27e-5) < 1e-18


def test_half_ulp():
    assert half_ulp("1.3e+3 g") == 50 and abs(half_ulp("0.174 m") - 0.0005) < 1e-12
    assert half_ulp("264,600N") == 0.5 and half_ulp("16/7") == 0.0


def test_gold_ok():
    assert gold_ok("(c)", "Multiple Choice") and gold_ok("D", "Multiple Choice")
    assert not gold_ok("a, b, c", "Multiple Choice")
    assert gold_ok("569.3 J", "Float") and not gold_ok("0.377 or 37.7%", "Percentage")
    assert not gold_ok("y = 3x", "Expression")


def test_is_correct():
    assert is_correct("0.1745 m", "0.174 m", "Float")       # within 1%
    assert not is_correct("0.19 m", "0.174 m", "Float")
    assert is_correct("0.746", "74.60%", "Percentage") and is_correct("74.6%", "74.60%", "Percentage")
    assert is_correct("C", "(c)", "Multiple Choice") and not is_correct("b", "(c)", "Multiple Choice")
    assert is_correct("(c) 6.0 x 10^-4 Wb", "c", "Multiple Choice") and is_correct("C. Mitosis", "c", "Multiple Choice")
    assert not is_correct("all of the above", "a", "Multiple Choice")
    assert not is_correct(None, "5", "Integer")
    assert is_correct("1277 g", "1.3e+3 g", "Float")          # gold rounded to 2 sig figs
    assert not is_correct("1200 g", "1.3e+3 g", "Float")
    assert not is_correct("1.4", "1", "Integer")


def test_well_formed_analysis():
    assert well_formed_analysis(GAR_OK)
    assert not well_formed_analysis("Answer: 6")                                    # no tags
    assert not well_formed_analysis("<analysis>\n</analysis>\nAnswer: 6")           # empty
    assert not well_formed_analysis("<analysis>x</analysis><analysis>y</analysis>\nAnswer: 6")
    assert not well_formed_analysis("<analysis>x\nAnswer: 6\n</analysis>")          # answer inside


def test_all_wrong():
    assert group_rewards(["Answer: 1", "Answer: 2"], [5, 9], "7", "Integer", lam=0.1) == [0.0, 0.0]


def test_one_correct_gets_no_length_penalty():
    assert group_rewards(["Answer: 7", "Answer: 2"], [50, 9], "7", "Integer", lam=0.1) == [1.0, 0.0]


def test_length_penalty_correct_only():
    r = group_rewards(["Answer: 7", "Answer: 7", "Answer: 3"], [10, 30, 1], "7", "Integer", lam=0.1)
    assert r == [1.0, 0.9, 0.0]


def test_equal_lengths():
    assert len_scores([10, 10, 10], [True, True, False]) == [0.0, 0.0, 0.0]


def test_trl_reward_functions_group_by_problem():
    # Shape GRPOTrainer passes: completions as chat messages, dataset columns as lists, extra kwargs.
    comp = [[{"role": "assistant", "content": t}] for t in ("Answer: 7", "Answer: 7", "Answer: 1", GAR_OK)]
    kw = {"completion_ids": [[0] * 10, [0] * 30, [0] * 5, [0] * 5], "id": ["p1", "p1", "p2", "p2"],
          "answer": ["7", "7", "6", "6"], "answer_type": ["Integer", "Integer", "Float", "Float"],
          "prompts": None, "trainer_state": None, "question": ["q1", "q1", "q2", "q2"]}
    assert trl_correct_reward(lam=0.1)(comp, **kw) == [1.0, 0.9, 0.0, 1.0]
    assert analysis_format(comp, **kw) == [0.0, 0.0, 0.0, 1.0]


def test_format_bonus_only_when_well_formed():
    bad = "<analysis></analysis>\nAnswer: 6 N"
    assert group_rewards([GAR_OK, bad], [5, 5], "6", "Float", fmt_bonus=0.1) == [1.1, 1.0]
