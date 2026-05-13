import json
from .run_experiments import get_benchmarks, get_optimizers
from .experiment_configs import LM_CONFIGS

# Set your experiment script file!
EXPERIMENT_SCRIPT = "scripts.run_experiments"

# Seeds, run_on_testset, dry_run as in your loop:
SEEDS = [0]
DRY_RUN = False

benchmark_metas, optimizers = get_benchmarks(), get_optimizers()

for seed in SEEDS:
    for bm_idx, benchmark_meta in enumerate(benchmark_metas):
        dataset_mode = getattr(benchmark_meta, 'dataset_mode', None)
        # Get the instantiated benchmark just for the name
        benchmark = benchmark_meta.benchmark(dataset_mode=dataset_mode) if dataset_mode else benchmark_meta.benchmark()
        benchmark_name = benchmark_meta.name or benchmark.__class__.__name__
        num_threads = getattr(benchmark_meta, 'num_threads', None) or 32

        for program_idx, program in enumerate(benchmark_meta.program):
            prog_name = getattr(program, "_name", program.__class__.__name__)
            for opt_idx, (optim_name, optimizer_config) in enumerate(optimizers):

                requires_80_gb_gpu = "GRPO" in optim_name
                num_gpus = 3 if "GRPO" in optim_name else 4

                name_clause = ""

                if "use_cache_from_opt" in optimizer_config.langProBe_configs and optimizer_config.langProBe_configs["use_cache_from_opt"] is not None:
                    use_cache_from_opt = optimizer_config.langProBe_configs["use_cache_from_opt"]
                else:
                    use_cache_from_opt = None

                for lm_config in LM_CONFIGS:
                    cmd = [
                        "uv run python -m", EXPERIMENT_SCRIPT,
                        f'--bm_idx {bm_idx}',
                        f'--benchmark_name "{benchmark_name}"',
                        f'--num_threads {num_threads}',
                        f'--program_idx {program_idx}',
                        f'--prog_name "{prog_name}"',
                        f'--opt_idx {opt_idx}',
                        f'--optim_name "{optim_name}"',
                        f"--lm_config '{json.dumps(lm_config)}'",
                        f"--seed {seed}",
                    ]
                    if use_cache_from_opt is not None:
                        cmd.append(f"--use_cache_from_opt {use_cache_from_opt}")
                    if DRY_RUN:
                        cmd.append("--dry_run")

                    cmd_to_launch = " ".join(cmd)
                    print(cmd_to_launch)
