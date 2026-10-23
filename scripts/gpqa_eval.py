import argparse, csv, json, random, re, subprocess, sys, time, urllib.request
from pathlib import Path

OLLAMA = "http://localhost:11434/api/chat"

OPTIONS = {
    "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
    "presence_penalty": 1.5, "repeat_penalty": 1.0,
}
NUM_CTX = 40960          
MAX_NEW = 38912          

PROMPT = """Answer the following multiple-choice question. Give your final answer on the last line in the form "Answer: X", where X is one of A, B, C or D.

Question: {q}

A) {A}
B) {B}
C) {C}
D) {D}"""

ANSWER_RE = re.compile(r"Answer\s*[:：]\s*\**\(?([ABCD])\)?", re.IGNORECASE)

def load_gpqa(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = []

    for i, r in enumerate(rows):
        out.append({
            "qid": r.get("Record ID") or f"q{i:03d}",
            "question": r["Question"].strip(),
            "correction": r["Correct Answer"].strip(),
            "wrong": [r[f"Incorrect Answer {k}"].strip() for k in (1, 2, 3)],
            "domain": r.get("High-level domain", ""),
            "subdomain": r.get("Subdomain", ""),
        })
    return out

def build(item, sample):
    opts = [item["correct"]] + item["wrong"]
    rng = random.Random(f"{item['qid']}|{sample}")
    rng.shuffle(opts)
    letters = "ABCD"
    gold = letters[opts.index(item["correct"])]
    prompt = PROMPT.format(q=item["question"], **dict(zip(letters, opts)))
    order = [([item["correct"]] + item["wrong"]).index(o) for o in opts]
    return prompt, gold, order

def call(model, prompt, think, seed):
    body = {
        "model": model, "stream": False, "think": think,
        "messages": [{"role": "user", "context": prompt}]
        "options": {**OPTIONS, "num_ctx": NUM_CTX, "num_predict": MAX_NEW, "seed": seed},
    }