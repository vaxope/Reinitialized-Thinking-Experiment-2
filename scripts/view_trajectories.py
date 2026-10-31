"""Write one local HTML page for reading trajectories side by side, grouped by question.
The page quotes GPQA: keep it in gitignored runs/ and never share or publish it.

Usage: python scripts/view_trajectories.py runs/RUN-002-full runs/RUN-008-gar-pilot --model qwen3.5:2b \
         --csv gpqa_diamond.csv --out runs/trajectories-2b.html
"""
import argparse, json
from collections import defaultdict
from html import escape
from pathlib import Path

from gpqa_eval import build, load_gpqa

CSS = """body{font:14px system-ui,sans-serif;margin:16px;background:#fff;color:#111}
details{border:1px solid #ccc;border-radius:6px;margin:6px 0;padding:6px 10px}
summary{cursor:pointer}.ok{color:#087f23;font-weight:600}.bad{color:#b00020;font-weight:600}
pre{white-space:pre-wrap;word-wrap:break-word;background:#f6f6f6;padding:8px;border-radius:4px;max-height:70vh;overflow:auto}
.think{background:#eef3fb}.q{background:#fffbe6}
@media (prefers-color-scheme:dark){body{background:#111;color:#ddd}pre{background:#1d1d1d}.think{background:#18202c}.q{background:#262214}
.ok{color:#5fd27a}.bad{color:#ff6b81}details{border-color:#444}}"""
JS = """function f(){const d=document.getElementById('dis').checked,w=document.getElementById('wr').value;
for(const e of document.querySelectorAll('details.question')){const c=JSON.parse(e.dataset.c);
const v=Object.values(c);let show=!d||(v.some(x=>x)&&v.some(x=>!x));if(w&&c[w]!==false)show=false;e.style.display=show?'':'none'}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run folders containing raw.jsonl")
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", help="GPQA csv, to show the question as the model saw it (sample 0 shuffle)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = defaultdict(dict)  # (qid, sample) -> arm -> row
    for d in a.runs:
        for line in open(Path(d) / "raw.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["model"] == a.model:
                rows[(r["qid"], r["sample"])][r["arm"]] = r
    items = {it["qid"]: it for it in load_gpqa(a.csv)} if a.csv else {}
    arms = sorted({arm for v in rows.values() for arm in v})
    parts = [f"<!doctype html><meta charset=utf-8><title>Trajectories {escape(a.model)}</title><style>{CSS}</style>",
             f"<h1>{escape(a.model)}: {len(rows)} question-samples, arms {', '.join(arms)}</h1>",
             "<p><label><input type=checkbox id=dis onchange=f()> only questions where arms disagree</label> &nbsp; "
             "<label>only questions wrong in <select id=wr onchange=f()><option value=''>any</option>"
             + "".join(f"<option>{x}</option>" for x in arms) + "</select></label></p>"]
    for n, ((qid, s), by) in enumerate(sorted(rows.items()), 1):
        gold = next(iter(by.values()))["gold"]
        head = " · ".join(f"{arm} <span class={'ok' if r['correct'] else 'bad'}>{r['pred']}</span> ({r['eval_count']} tok)"
                          for arm, r in sorted(by.items()))
        body = []
        if qid in items:
            body.append(f"<pre class=q>{escape(build(items[qid], s)[0])}</pre>")
        for arm, r in sorted(by.items()):
            th = f"<details><summary>thinking ({len(r['thinking'])} chars)</summary><pre class=think>{escape(r['thinking'])}</pre></details>" if r["thinking"] else ""
            body.append(f"<details><summary><b>{arm}</b>: pred {r['pred']}, {'correct' if r['correct'] else 'wrong'}, "
                        f"{r['eval_count']} tokens{', TRUNCATED' if r['truncated'] else ''}</summary>{th}"
                        f"<pre>{escape(r['content'])}</pre></details>")
        c = json.dumps({arm: r["correct"] for arm, r in by.items()})
        parts.append(f"<details class=question data-c='{escape(c)}'><summary>#{n} {qid} s{s} [{escape(next(iter(by.values()))['domain'])}] "
                     f"gold <b>{gold}</b> — {head}</summary>{''.join(body)}</details>")
    parts.append(f"<script>{JS}</script>")
    Path(a.out).write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {a.out}: {len(rows)} question-samples")


if __name__ == "__main__":
    main()
