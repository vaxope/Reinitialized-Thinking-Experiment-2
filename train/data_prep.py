"""Build the Stage A training set from WebInstruct-verified, decontaminated against GPQA Diamond.
Runs locally (needs Ollama for embeddings); only train.jsonl / heldout.jsonl go to DeltaAI.

Writes <out>/{train.jsonl, heldout.jsonl, removed.jsonl, prep_config.json}.
removed.jsonl holds only WebInstruct ids, GPQA qids and scores, never question text.
GPQA is used here only to remove overlap; it is never trained on.
"""
import argparse, json, random, re, sys, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rewards import gold_ok
from gpqa_eval import load_gpqa

CATEGORIES = ["Physics", "Chemistry", "Biology", "Other STEM"]
DIFFICULTY = ["Senior High School", "University", "PhD"]
GPQA_MIX = {"Chemistry": 0.47, "Physics": 0.43, "Biology": 0.10}  # GPQA Diamond: 93 / 86 / 19 of 198
IMAGE_RE = r"\b(?:figure|diagram|graph shown|picture|image|shown below|shown above)\b"
N = 13  # word n-gram length for exact-overlap decontamination
OLLAMA_EMBED = "http://localhost:11434/api/embed"


def embed(model, texts, batch=64):
    """Unit-normalised embeddings from the local Ollama server, so GPQA text never leaves this machine."""
    vecs = []
    for i in range(0, len(texts), batch):
        req = urllib.request.Request(OLLAMA_EMBED, data=json.dumps({"model": model, "input": texts[i:i + batch]}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=None) as r:
            vecs += json.load(r)["embeddings"]
        print(f"embedded {min(i + batch, len(texts))}/{len(texts)}", end="\r", flush=True)
    v = np.asarray(vecs, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def ngrams(text):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(w[i:i + N]) for i in range(len(w) - N + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="WebInstruct-verified train parquet")
    ap.add_argument("--gpqa", required=True, help="gpqa_diamond.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=2560)
    ap.add_argument("--n-heldout", type=int, default=300)
    ap.add_argument("--embed-model", default="qwen3-embedding:0.6b",
                    help="Ollama embedding model; '' skips embedding decontamination (testing only)")
    ap.add_argument("--sim-threshold", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    d = pd.read_parquet(a.parquet)
    n_all = len(d)
    d = d[d.category.isin(CATEGORIES) & d.difficulty.isin(DIFFICULTY)]
    d = d[[gold_ok(g, t) for g, t in zip(d.answer, d.answer_type)]]
    d = d[~d.question.str.contains(IMAGE_RE, case=False, regex=True)].reset_index(drop=True)
    n_checkable = len(d)

    # Draw a shuffled pool per domain in GPQA's mix (so training is not mostly physics), with a 1.5x buffer
    # for decontamination losses; embedding all ~54k questions locally would take hours.
    n_tot = a.n_train + a.n_heldout
    want = {cat: round(n_tot * frac) for cat, frac in GPQA_MIX.items()}
    d = pd.concat([d[d.category == c].sample(frac=1, random_state=a.seed).head(int(n * 1.5))
                   for c, n in want.items()]).reset_index(drop=True)

    gpqa = load_gpqa(a.gpqa)
    gtexts = [f"{g['question']} {g['correct']}" for g in gpqa]
    gram_owner = {}
    for g, t in zip(gpqa, gtexts):
        for x in ngrams(t):
            gram_owner[x] = g["qid"]

    removed = {}
    for i, q in enumerate(d.question):
        hit = next((gram_owner[x] for x in ngrams(q) if x in gram_owner), None)
        if hit:
            removed[i] = {"id": d.id[i], "why": "ngram", "gpqa_qid": hit}

    if a.embed_model:
        ge = embed(a.embed_model, gtexts)
        we = embed(a.embed_model, list(d.question))
        sims = we @ ge.T
        best, arg = sims.max(axis=1), sims.argmax(axis=1)
        np.save(out / "embed_best_sim.npy", best)  # for choosing the threshold; no text
        for i in np.nonzero(best >= a.sim_threshold)[0].tolist():
            removed.setdefault(i, {"id": d.id[i], "why": "embed", "gpqa_qid": gpqa[arg[i]]["qid"],
                                   "sim": round(float(best[i]), 3)})
    keep = d.drop(index=list(removed))

    rows = []
    for cat, n in want.items():
        pool = keep[keep.category == cat].to_dict("records")
        if len(pool) < n:
            sys.exit(f"{cat}: only {len(pool)} clean problems for {n}; raise the buffer")
        rows += pool[:n]
    random.Random(a.seed).shuffle(rows)
    cols = ["id", "question", "answer", "answer_type", "category", "difficulty"]
    for name, part in (("train", rows[: a.n_train]), ("heldout", rows[a.n_train: n_tot])):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps({c: r[c] for c in cols}, ensure_ascii=False) + "\n")
    with (out / "removed.jsonl").open("w", encoding="utf-8") as f:
        for r in removed.values():
            f.write(json.dumps(r) + "\n")

    stats = {"n_webinstruct": n_all, "n_checkable": n_checkable, "n_pool_decontaminated": len(d),
             "removed_ngram": sum(r["why"] == "ngram" for r in removed.values()),
             "removed_embed": sum(r["why"] == "embed" for r in removed.values()),
             "n_train": min(a.n_train, len(rows)), "n_heldout": max(0, len(rows) - a.n_train)}
    (out / "prep_config.json").write_text(json.dumps({**vars(a), **stats, "categories": CATEGORIES,
                                                      "difficulty": DIFFICULTY, "mix": GPQA_MIX,
                                                      "ngram_n": N}, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
