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
            "correct": r["Correct Answer"].strip(),
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
        "messages": [{"role": "user", "content": prompt}],
        "options": {**OPTIONS, "num_ctx": NUM_CTX, "num_predict": MAX_NEW, "seed": seed},
    }

    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=None) as r:
        resp = json.load(r)
    resp["_wall_s"] = time.time() - t0
    return resp

def extract(text):
    m = ANSWER_RE.findall(text or "")
    return m[-1].upper() if m else None

def model_info(model):
    try:
        req = urllib.request.Request("http://localhost:11434/api/show",
                                             data=json.dumps({"model": model}).encode(),
                                             headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            d = json.load(r)
        return {"quant": d.get("details", {}). get("quantization_level"),
                "params": d.get("parameters"), "capabilities": d.get("capabilities")}
    except Exception as e:
        return {"error": str(e)}

def ollama_version():
    try:
        with urllib.request.urlopen("http://localhost:11434/api/version") as r:
            return json.load(r).get("version")
    except Exception:
        return "unknown"

def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=["qwen3.5:4b"])
    ap.add_argument("--arms", nargs="+", default=["think", "nothink"], choices=["think", "nothink"])
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="first N questions (pilot)")
    a = ap.parse_args()

    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    items = load_gpqa(a.csv)[: a.limit]
    (out / "config.json").write_text(json.dumps({
        "models": a.models, "arms": a.arms, "samples": a.samples, "n_questions": len(items),
        "options": OPTIONS, "num_ctx": NUM_CTX, "max_new": MAX_NEW,
        "prompt": PROMPT, "git": git_hash(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ollama_version": ollama_version(), "model_info": {m: model_info(m) for m in a.models},
    }, indent=2))

    raw = out / "raw.jsonl"
    done = set()
    if raw.exists():
        for line in raw.open(encoding="utf-8"):
            d = json.loads(line); done.add((d["model"], d["arm"], d["qid"], d["sample"]))

    quant = {m: model_info(m).get("quant") for m in a.models}
    # Interleave arms per question so a stopped run still has paired data.
    with raw.open("a", encoding="utf-8") as f:
        for model in a.models:
            for s in range(a.samples):
                for item in items:
                    prompt, gold, order = build(item, s)
                    for arm in a.arms:
                        key = (model, arm, item["qid"], s)
                        if key in done:
                            continue
                        seed = random.Random(f"{item['qid']}|{s}|{arm}").randrange(2**31)
                        r = call(model, prompt, arm == "think", seed)
                        msg = r.get("message", {})
                        content, thinking = msg.get("content", ""), msg.get("thinking", "")
                        pred = extract(content)
                        rec = {
                            "model": model, "quant": quant[model], "arm": arm,
                            "qid": item["qid"], "sample": s,
                            "domain": item["domain"], "subdomain": item["subdomain"],
                            "option_order": order, "gold": gold, "pred": pred,
                            "correct": pred == gold, "parse_ok": pred is not None,
                            "truncated": r.get("done_reason") == "length",
                            "done_reason": r.get("done_reason"),
                            "thinking_tokens": None, "answer_tokens": None,  # filled by count_tokens.py
                            "eval_count": r.get("eval_count"),           # all generated tokens
                            "prompt_eval_count": r.get("prompt_eval_count"),
                            "tok_per_s": (r["eval_count"] / (r["eval_duration"] / 1e9))
                                          if r.get("eval_duration") else None,
                            "wall_s": round(r["_wall_s"], 1),
                            "thinking": thinking, "content": content, "seed": seed,
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                        print(f"{model} {arm:7s} {item['qid']} s{s} gold={gold} pred={pred} "
                              f"tok={rec['eval_count']} {rec['tok_per_s'] or 0:.1f}tok/s "
                              f"{'TRUNC' if rec['truncated'] else ''}", flush=True)

if __name__ == "__main__":
    sys.exit(main())