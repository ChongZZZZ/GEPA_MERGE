# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa
#
# Unified evaluation script for proposal experiments.
# Runs GEPA optimization on a given task + model combination.
#
# Usage:
#   uv run python src/gepa/examples/run_eval.py --task hotpotqa --model openai/gpt-4.1-mini
#   uv run python src/gepa/examples/run_eval.py --task hover --model ollama/qwen3:8b
#   uv run python src/gepa/examples/run_eval.py --task ifbench --model ollama/qwen3:14b --use_merge

import argparse
import json
import os


# Single-prompt seeds used by the simple (DefaultAdapter) path. HotpotQA below
# uses a 4-module seed instead (see HOTPOTQA_MULTIHOP_SEED).
SEED_PROMPTS = {
    "hotpotqa": "Answer the question concisely based on your knowledge. Put your final answer in one sentence.",
    "hover": (
        "You are a fact-checking assistant. Given a claim, determine if it is SUPPORTED or NOT_SUPPORTED "
        "based on your knowledge. Reply with exactly one word: SUPPORTED or NOT_SUPPORTED."
    ),
    "ifbench": "Follow the user's instructions carefully and completely.",
}


# 4-module seed for multi-hop HotpotQA. Base prompts mirror GEPA paper
# Section L.1 (HotpotQA GPT-4.1-Mini). Wired via MultiHopHotpotAdapter.
HOTPOTQA_MULTIHOP_SEED = {
    "summarize1": "Given the fields `question`, `passages`, produce the fields `summary`.",
    "create_query_hop2": "Given the fields `question`, `summary_1`, produce the fields `query`.",
    "summarize2": "Given the fields `question`, `context`, `passages`, produce the fields `summary`.",
    "final_answer": "Given the fields `question`, `summary_1`, `summary_2`, produce the fields `answer`.",
}


# Paper-verified multi-module seeds for HoVer, IFBench (GEPA paper
# Sections L.5, L.3 respectively). These are the ACTUAL seeds used in
# the paper — ready to wire once we build the corresponding multi-module
# adapters. Until then, the single-prompt SEED_PROMPTS entries above are
# what runs via DefaultAdapter.
#
# K (module count) per task:
#   HoVer   K=4  (3-hop retrieval: 2 query writers + 2 summarizers)
#   IFBench K=2  (generate response → ensure constraints)
HOVER_MULTIHOP_SEED = {
    "summarize1":        "Given the fields `claim`, `passages`, produce the fields `summary`.",
    "create_query_hop2": "Given the fields `claim`, `summary_1`, produce the fields `query`.",
    "summarize2":        "Given the fields `claim`, `context`, `passages`, produce the fields `summary`.",
    "create_query_hop3": "Given the fields `claim`, `summary_1`, `summary_2`, produce the fields `query`.",
}

IFBENCH_MULTI_SEED = {
    "generate_response":       "Respond to the query",
    "ensure_correct_response": (
        "Ensure the response is correct and adheres to the given constraints. "
        "Your response will be used as the final response."
    ),
}


# Paper Table 1 per-task defaults (train / val / rollout budget). We follow
# GEPA paper splits; rollout budget defaults can be reduced via CLI for
# cost-constrained API runs (we do this for Phase A on gpt-4.1-mini).
TASK_DEFAULTS = {
    "hotpotqa": {"train_size": 150, "val_size": 300, "max_metric_calls": 6871},
    "hover":    {"train_size": 150, "val_size": 300, "max_metric_calls": 7051},
    "ifbench":  {"train_size": 150, "val_size": 300, "max_metric_calls": 3593},
}


def load_task(task: str):
    if task == "hotpotqa":
        from gepa.examples.hotpotqa import hotpotqa_evaluator, init_dataset

        return init_dataset, hotpotqa_evaluator
    elif task == "hover":
        from gepa.examples.hover import hover_evaluator, init_dataset

        return init_dataset, hover_evaluator
    elif task == "ifbench":
        from gepa.examples.ifbench import ifbench_evaluator, init_dataset

        return init_dataset, ifbench_evaluator
    else:
        raise ValueError(f"Unknown task: {task}. Choose from: hotpotqa, hover, ifbench")


def main():
    parser = argparse.ArgumentParser(description="Run GEPA evaluation for proposal experiments.")
    parser.add_argument("--task", type=str, required=True, choices=["hotpotqa", "hover", "ifbench"])
    parser.add_argument("--model", type=str, default="openai/gpt-4.1-mini", help="task_lm model string")
    parser.add_argument("--reflection_lm", type=str, default="openai/gpt-4o", help="reflection_lm model string")
    parser.add_argument(
        "--max_metric_calls", type=int, default=None,
        help="Rollout budget. Defaults to paper Table 1 per task (hotpotqa=6871, hover=7051, "
             "ifbench=3593). Override for cost-constrained runs.",
    )
    parser.add_argument(
        "--train_size", type=int, default=None,
        help="Train set size. Defaults to paper per-task split (hotpotqa/hover/ifbench=150).",
    )
    parser.add_argument(
        "--val_size", type=int, default=None,
        help="Val set size. Defaults to paper per-task split (hotpotqa/hover/ifbench=300).",
    )
    parser.add_argument("--use_merge", action="store_true", help="Enable GEPA+Merge")
    parser.add_argument(
        "--max_merge_invocations", type=int, default=5,
        help="Merge-invocation cap (paper default = 5).",
    )
    parser.add_argument(
        "--merge_selection",
        type=str,
        default="random",
        choices=["random", "divergence", "score", "complementary"],
        help="Candidate-pair selection strategy (Dimension A).",
    )
    parser.add_argument(
        "--merge_start",
        type=str,
        default="immediate",
        choices=["immediate", "delayed", "periodic", "diversity"],
        help="Merge start policy (Dimension B).",
    )
    parser.add_argument(
        "--merge_start_param",
        type=float,
        default=None,
        help=(
            "Parameter for the start policy: "
            "delayed -> min_candidates (int), periodic -> merge_every_n (int), "
            "diversity -> diversity_threshold (float in [0,1]). Ignored for 'immediate'."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility.")
    parser.add_argument("--output_file", type=str, default=None, help="Save result to JSON file")
    parser.add_argument(
        "--merge_quality",
        action="store_true",
        help="Enable Tier A merge-quality diagnostics. Writes candidates.jsonl + merge_quality.jsonl into run_dir.",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Directory for run artifacts (required for --merge_quality sidecars).",
    )
    parser.add_argument(
        "--single_prompt_hotpot",
        action="store_true",
        help="Fall back to the legacy single-module HotpotQA setup (DefaultAdapter, 1 seed prompt). "
             "Default is the 4-module multi-hop setup from GEPA paper Section L.1.",
    )
    args = parser.parse_args()

    defaults = TASK_DEFAULTS[args.task]
    if args.train_size is None:
        args.train_size = defaults["train_size"]
    if args.val_size is None:
        args.val_size = defaults["val_size"]
    if args.max_metric_calls is None:
        args.max_metric_calls = defaults["max_metric_calls"]

    import gepa

    init_dataset, evaluator = load_task(args.task)
    trainset, valset, _ = init_dataset(train_size=args.train_size, val_size=args.val_size)

    is_multihop_hotpot = args.task == "hotpotqa" and not args.single_prompt_hotpot

    if is_multihop_hotpot:
        from gepa.adapters.multi_hop_hotpot_adapter import MultiHopHotpotAdapter

        seed_prompt = dict(HOTPOTQA_MULTIHOP_SEED)
        adapter = MultiHopHotpotAdapter(model=args.model)
        task_lm_arg = None  # adapter holds the model
    else:
        seed_prompt = {"system_prompt": SEED_PROMPTS[args.task]}
        adapter = None
        task_lm_arg = args.model

    print(f"Task:           {args.task}")
    print(f"Model:          {args.model}")
    print(f"Reflection LM:  {args.reflection_lm}")
    print(f"Max calls:      {args.max_metric_calls}")
    print(f"Use merge:      {args.use_merge}")
    print(f"Multi-hop:      {is_multihop_hotpot}")
    print(f"Modules:        {list(seed_prompt.keys())}")
    print(f"Train size:     {len(trainset)}  Val size: {len(valset)}")
    print("-" * 60)

    optimize_kwargs = dict(
        seed_candidate=seed_prompt,
        trainset=trainset,
        valset=valset,
        reflection_lm=args.reflection_lm,
        max_metric_calls=args.max_metric_calls,
        use_merge=args.use_merge,
        max_merge_invocations=args.max_merge_invocations if args.use_merge else 0,
        merge_selection_strategy=args.merge_selection,
        merge_start_policy=args.merge_start,
        merge_start_param=args.merge_start_param,
        merge_quality=args.merge_quality,
        task_name=args.task,
        run_dir=args.run_dir,
        seed=args.seed,
    )
    if adapter is not None:
        optimize_kwargs["adapter"] = adapter
    else:
        optimize_kwargs["task_lm"] = task_lm_arg
        optimize_kwargs["evaluator"] = evaluator

    result = gepa.optimize(**optimize_kwargs)

    best = result.best_candidate
    print("\n=== Optimized Candidate ===")
    for k, v in best.items():
        print(f"--- {k} ---")
        print(v)
        print()

    if args.output_file:
        out = {
            "task": args.task,
            "model": args.model,
            "reflection_lm": args.reflection_lm,
            "use_merge": args.use_merge,
            "max_metric_calls": args.max_metric_calls,
            "best_candidate": best,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResult saved to {args.output_file}")


if __name__ == "__main__":
    main()
