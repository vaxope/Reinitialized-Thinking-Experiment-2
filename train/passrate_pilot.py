"""Local pilot (Ollama): k attempts per training problem, to estimate how many problems GRPO can learn
from (solved sometimes, not always) and how long answers run (sets max_completion_length).
Usage: python train/passrate_pilot.py --data runs/data-prep-local/train.jsonl --out runs/passrate-pilot
"""
import argparse, json, statistics as st, sys, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from formats import FORMATS, user_message
from rewards import extract_answer, is_correct, well_formed_analysis
from gpqa_eval import OLLAMA

# Training sampler (grpo.py): no presence penalty, so samples stay on-policy for the GRPO update.
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3.5:2b")
    ap.add_argument("--format", default="gar", choices=list(FORMATS))
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=6144)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**vars(a), "sampling": SAMPLING}, indent=2))
    probs = [json.loads(l) for l in open(a.data, encoding="utf-8")][: a.n]

    raw = out / "raw.jsonl"
    done = {(d["id"], d["k"]) for d in map(json.loads, raw.open(encoding="utf-8"))} if raw.exists() else set()
    with raw.open("a", encoding="utf-8") as f:
        for p in probs:
            for k in range(a.k):
                if (p["id"], k) in done:
                    continue
                body = {"model": a.model, "stream": False, "think": FORMATS[a.format]["enable_thinking"],
                        "messages": [{"role": "user", "content": user_message(p["question"], a.format)}],
                        "options": {**SAMPLING, "num_ctx": 8192, "num_predict": a.max_new, "seed": k}}
                req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=None) as r:
                    resp = json.load(r)
                text = resp["message"].get("content", "")
                pred = extract_answer(text)
                rec = {"id": p["id"], "k": k, "answer_type": p["answer_type"], "gold": p["answer"], "pred": pred,
                       "correct": is_correct(pred, p["answer"], p["answer_type"]),
                       "well_formed": well_formed_analysis(text), "tokens": resp.get("eval_count"),
                       "truncated": resp.get("done_reason") == "length", "content": text}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                print(p["id"], k, rec["correct"], rec["tokens"], flush=True)

    rows = [json.loads(l) for l in raw.open(encoding="utf-8")]
    by = {}
    for r in rows:
        by.setdefault(r["id"], []).append(r["correct"])
    rates = [sum(v) / len(v) for v in by.values()]
    toks = sorted(r["tokens"] or 0 for r in rows)
    print(f"problems={len(by)} attempts={len(rows)} acc={st.mean(r['correct'] for r in rows):.2f}")
    print(f"never solved={sum(x == 0 for x in rates)}  always={sum(x == 1 for x in rates)}  "
          f"sometimes={sum(0 < x < 1 for x in rates)}  GAR filter (0, 0.5]={sum(0 < x <= 0.5 for x in rates)}")
    print(f"tokens median={st.median(toks):.0f} p90={toks[int(0.9 * (len(toks) - 1))]} "
          f"p95={toks[int(0.95 * (len(toks) - 1))]} max={toks[-1]}  truncated={sum(r['truncated'] for r in rows)}  "
          f"well-formed={sum(r['well_formed'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
