import argparse, json
from pathlib import Path
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.tokenizer)
d = Path(a.run_dir).expanduser()
with (d / "raw.jsonl").open(encoding="utf-8") as fin, \
     (d / "raw_tokens.jsonl").open("w", encoding="utf-8") as fout:
    for line in fin:
        r = json.loads(line)
        r["thinking_tokens"] = len(tok.encode(r.get("thinking") or "", add_special_tokens=False))
        r["answer_tokens"] = len(tok.encode(r.get("content") or "", add_special_tokens=False))
        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", d / "raw_tokens.jsonl")
