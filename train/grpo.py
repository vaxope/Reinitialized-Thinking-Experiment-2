"""Stage A: solver-only GRPO (TRL + vLLM colocate, 1 GPU), loss and hyperparameters from GAR-Official.

GAR-Official's GRPO (RL_trainner.py): group-normalised advantage, per-sequence token mean ('grpo' loss),
clip 0.2, KL beta 0.01, lr 5e-6, global batch 32. Its Lean prover/fuser loop is Stage B and not used here.
Resumes from the last checkpoint in --out.
"""
import argparse, importlib.metadata as md, json, sys, time
from pathlib import Path

import torch
from datasets import Dataset
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from formats import FORMATS, user_message
from rewards import analysis_format, trl_correct_reward
from gpqa_eval import git_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", required=True, choices=list(FORMATS))
    ap.add_argument("--train", required=True, help="train.jsonl from data_prep.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--max-completion", type=int, default=4096, help="set from passrate_pilot p95")
    ap.add_argument("--max-steps", type=int, default=-1, help="optimizer steps; -1 = one pass over --train")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--num-gen", type=int, default=8)
    ap.add_argument("--len-lambda", type=float, default=0.0, help="length penalty on correct answers")
    ap.add_argument("--fmt-bonus", type=float, default=0.1, help="gar format only")
    ap.add_argument("--save-steps", type=int, default=40)
    # per-token: over ~2.5k-token answers, sequence ratios scattered (job 3488573: mean 0.58, many masked)
    ap.add_argument("--is-mode", default="token_truncate", help="TRL vllm_importance_sampling_mode")
    ap.add_argument("--report-to", default="wandb")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(a.train, encoding="utf-8")]
    ds = Dataset.from_list([{**r, "prompt": [{"role": "user", "content": user_message(r["question"], a.format)}]}
                            for r in rows])

    correct = trl_correct_reward(a.len_lambda)
    funcs, weights = [correct], [1.0]
    if a.format == "gar" and a.fmt_bonus:
        funcs, weights = [correct, analysis_format], [1.0, a.fmt_bonus]

    cfg = GRPOConfig(
        output_dir=str(out), seed=a.seed, max_steps=a.max_steps, num_train_epochs=1,
        learning_rate=a.lr, beta=a.beta, epsilon=0.2, loss_type="grpo", scale_rewards="group",
        num_generations=a.num_gen, reward_weights=weights,
        per_device_train_batch_size=2, gradient_accumulation_steps=16,   # 32 sequences per update (GAR)
        generation_batch_size=256,                                       # 32 problems x 8 per vLLM call
        max_completion_length=a.max_completion,
        temperature=1.0, top_p=1.0, top_k=0, min_p=0.0,   # plain sampling: vLLM's logprobs then match the trainer's (RUN-005)
        chat_template_kwargs={"enable_thinking": FORMATS[a.format]["enable_thinking"]},
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.35, vllm_importance_sampling_mode=a.is_mode,
        vllm_max_model_length=a.max_completion + 2048, vllm_enable_sleep_mode=True,
        bf16=True, gradient_checkpointing=True,  # fp32 weights + bf16 autocast: in pure bf16, lr 5e-6 updates round to 0
        logging_steps=1, save_steps=a.save_steps, save_total_limit=2, report_to=a.report_to,
        run_name=out.name, log_completions=True, num_completions_to_print=2,
    )
    # a resumed job keeps the original config and writes its own next to it
    cfg_name = "run_config.json" if not (out / "run_config.json").exists() else f"run_config-{time.strftime('%Y%m%d-%H%M%S')}.json"
    (out / cfg_name).write_text(json.dumps({
        "args": vars(a), "format": FORMATS[a.format], "user_message_example": user_message("<question>", a.format),
        "grpo_config": cfg.to_dict(), "git": git_hash(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "versions": {p: md.version(p) for p in ("torch", "transformers", "trl", "vllm", "accelerate")},
    }, indent=2, default=str))

    trainer = GRPOTrainer(model=a.model, reward_funcs=funcs, args=cfg, train_dataset=ds)
    trainer.train(resume_from_checkpoint=get_last_checkpoint(str(out)))
    trainer.save_model(str(out / "final"))
    print(f"peak torch GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB (vLLM not included)", flush=True)


if __name__ == "__main__":
    main()
