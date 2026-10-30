"""Held-out WebInstruct check on DeltaAI (vLLM, bf16): untrained vs trained on the same 300 problems,
sampler, seeds and scoring. Uses the Phase 1 eval sampler (Qwen's), not the training sampler.

Usage: python train/heldout_eval.py --model <hf id or checkpoint dir> --tag <name> --data heldout.jsonl --out <dir>
Writes <out>/<tag>/<format>.jsonl (one row per attempt, full text) and <out>/<tag>/summary.json.
A format whose .jsonl already exists is skipped, so a rerun resumes.
"""
import argparse, json, statistics as st, sys, time
from pathlib import Path

from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
from formats import FORMATS, user_message
from rewards import extract_answer, is_correct, well_formed_analysis

SAMPLER = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5)  # = scripts/gpqa_eval.py


def answer_text(text, fmt):
    """For think, only the reply after </think> is scored (as in Phase 1); no </think> means no answer."""
    if FORMATS[fmt]["enable_thinking"]:
        return text.split("</think>", 1)[1] if "</think>" in text else ""
    return text


def summarize(rows, wall_s):
    toks = sorted(r["tokens"] for r in rows)
    n_ok = sum(r["correct"] for r in rows)
    return {"n": len(rows), "acc": round(n_ok / len(rows), 4), "tok_mean": round(st.mean(toks)),
            "tok_median": st.median(toks), "tok_p90": toks[int(0.9 * (len(toks) - 1))],
            "tok_per_correct": round(sum(toks) / n_ok) if n_ok else None,
            "truncated": sum(r["truncated"] for r in rows), "no_answer": sum(r["pred"] is None for r in rows),
            "well_formed": sum(r["well_formed"] for r in rows), "wall_s": round(wall_s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-2B", help="same tokenizer and chat template for every model")
    ap.add_argument("--formats", nargs="+", default=["gar", "nothink", "think"], choices=list(FORMATS))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=8192)
    a = ap.parse_args()
    out = Path(a.out) / a.tag; out.mkdir(parents=True, exist_ok=True)
    probs = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    assert len({p["id"] for p in probs}) == len(probs), "duplicate ids in --data"

    summ_path = out / "summary.json"
    summary = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    summary["config"] = {"model": a.model, "tokenizer": a.tokenizer, "k": a.k, "max_new": a.max_new,
                         "sampler": SAMPLER, "seed": 0, "n_problems": len(probs), "data": a.data}
    llm = LLM(a.model, tokenizer=a.tokenizer, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=a.max_new + 2048, seed=0)
    for fmt in a.formats:
        raw = out / f"{fmt}.jsonl"
        if raw.exists():
            print(f"{fmt}: {raw} exists, skipping", flush=True)
            continue
        msgs = [[{"role": "user", "content": user_message(p["question"], fmt)}] for p in probs]
        sp = SamplingParams(n=a.k, max_tokens=a.max_new, seed=0, **SAMPLER)
        t0 = time.time()
        outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": FORMATS[fmt]["enable_thinking"]})
        dt = time.time() - t0
        rows = []
        for p, o in zip(probs, outs):
            for k, x in enumerate(o.outputs):
                pred = extract_answer(answer_text(x.text, fmt))
                rows.append({"id": p["id"], "category": p["category"], "answer_type": p["answer_type"], "k": k,
                             "pred": pred, "correct": is_correct(pred, p["answer"], p["answer_type"]),
                             "tokens": len(x.token_ids), "truncated": x.finish_reason == "length",
                             "well_formed": well_formed_analysis(x.text), "text": x.text})
        tmp = raw.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.rename(raw)  # only complete files count as done
        summary[fmt] = summarize(rows, dt)
        summ_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps({"tag": a.tag, "format": fmt, **summary[fmt]}), flush=True)


if __name__ == "__main__":
    main()
