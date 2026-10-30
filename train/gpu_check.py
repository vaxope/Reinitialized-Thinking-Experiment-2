"""First DeltaAI GPU job (<30 min): CUDA works, the chat template gives an empty <think> for gar,
rewards parse real outputs, and vLLM throughput / answer lengths for projecting the full run."""
import json, statistics as st, sys, time
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
from formats import FORMATS, user_message
from rewards import extract_answer, is_correct, well_formed_analysis

MODEL, N_PROB, K, MAX_NEW = "Qwen/Qwen3.5-2B", 64, 4, 4096


# vLLM starts its engine with spawn, which re-imports this file: keep work under main().
def main():
    assert torch.cuda.is_available(), "no CUDA: rerun setup_env.sh cu129"
    print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(), flush=True)

    probs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")][:N_PROB]
    llm = LLM(MODEL, gpu_memory_utilization=0.85, max_model_len=MAX_NEW + 2048, seed=0)
    tok = llm.get_tokenizer()
    for fmt in ("gar", "nothink"):
        msgs = [[{"role": "user", "content": user_message(p["question"], fmt)}] for p in probs]
        prompt0 = tok.apply_chat_template(msgs[0], tokenize=False, add_generation_prompt=True,
                                          enable_thinking=FORMATS[fmt]["enable_thinking"])
        assert prompt0.rstrip().endswith("</think>"), f"{fmt}: template did not close <think>:\n{prompt0[-300:]}"

        sp = SamplingParams(n=K, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, max_tokens=MAX_NEW, seed=0)  # = grpo.py
        t0 = time.time()
        outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": FORMATS[fmt]["enable_thinking"]})
        dt = time.time() - t0
        lens, rates, wf, trunc = [], [], 0, 0
        for p, o in zip(probs, outs):
            c = [is_correct(extract_answer(x.text), p["answer"], p["answer_type"]) for x in o.outputs]
            rates.append(sum(c) / K)
            for x in o.outputs:
                lens.append(len(x.token_ids)); wf += well_formed_analysis(x.text); trunc += x.finish_reason == "length"
        lens.sort()
        print(json.dumps({
            "format": fmt, "tok_per_s": round(sum(lens) / dt), "wall_s": round(dt),
            "acc": round(st.mean(rates), 3), "never": sum(r == 0 for r in rates), "always": sum(r == 1 for r in rates),
            "gar_filter_0_to_0.5": sum(0 < r <= 0.5 for r in rates), "len_median": st.median(lens),
            "len_p95": lens[int(0.95 * (len(lens) - 1))], "truncated": trunc, "well_formed": wf, "n": len(lens),
        }), flush=True)
        print(f"--- example {fmt} output ---\n{outs[0].outputs[0].text[:1500]}\n---", flush=True)


if __name__ == "__main__":
    main()
