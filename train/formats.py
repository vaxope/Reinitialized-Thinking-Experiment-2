"""Training formats. Each differs only in the instruction text and whether native thinking is on.
Changing any text here is a new format: give it a new name, never edit an existing one."""

GAR_STEPS = """1. Identify the principles and facts the question depends on.
2. Work through the problem step by step, and evaluate each option against them.
3. Check the chosen answer: consistency, units, limiting cases."""

ANSWER_LINE = 'Give your final answer on the last line in the form "Answer: X".'

FORMATS = {
    # Reinitialized thinking (solver version): empty <think>, reasoning restarted inside <analysis>.
    "gar": {"enable_thinking": False,
            "suffix": "Reason inside <analysis></analysis>, following these steps:\n"
                      f"{GAR_STEPS}\nAfter </analysis>, {ANSWER_LINE[0].lower()}{ANSWER_LINE[1:]}"},
    "nothink": {"enable_thinking": False, "suffix": ANSWER_LINE},
    "think": {"enable_thinking": True, "suffix": ANSWER_LINE},
}


def user_message(question, fmt):
    return f"{question.strip()}\n\n{FORMATS[fmt]['suffix']}"
