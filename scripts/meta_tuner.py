#!/usr/bin/env python3
"""
Optuna Meta-Tuner for V2 Gradient Placer

Searches over λ_den, λ_cong schedules, learning rate, and step count
to minimise average proxy cost on ibm01 + ibm06 (best + hardest cases).

Usage:
    uv run python scripts/meta_tuner.py              # 50 trials
    uv run python scripts/meta_tuner.py --trials 100 # custom trial count
    uv run python scripts/meta_tuner.py --benchmarks ibm01 ibm06 ibm12
"""

import argparse
import os
import re
import subprocess
import sys

import optuna


PLACER = "submissions/v2_gradient_placer.py"
DEFAULT_BENCHMARKS = ["ibm01", "ibm06"]


def run_benchmark(bench: str, env_overrides: dict) -> float:
    """Run the placer on a single benchmark, return proxy cost (or penalty)."""
    env = {**os.environ, **env_overrides, "PATH": os.environ.get("PATH", "")}
    # Ensure uv is on PATH
    uv_dir = os.path.expanduser("~/.local/bin")
    env["PATH"] = f"{uv_dir}:{env['PATH']}"

    result = subprocess.run(
        ["uv", "run", "evaluate", PLACER, "-b", bench],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )

    # Parse proxy cost from output like "proxy=1.2345"
    match = re.search(r"proxy=(\d+\.\d+)", result.stdout)
    if match:
        cost = float(match.group(1))
        # Check for overlaps
        if "VALID" not in result.stdout:
            cost += 5.0  # Heavy penalty for invalid placements
        return cost

    # Failed run
    return 10.0


def objective(trial: optuna.Trial, benchmarks: list) -> float:
    """Optuna objective: average proxy cost over target benchmarks."""

    # ── Hyperparameters to tune ───────────────────────────────────────
    lr = trial.suggest_float("lr", 0.01, 0.15, log=True)
    num_steps = trial.suggest_int("num_steps", 200, 800, step=50)
    lam_den_start = trial.suggest_float("lambda_den_start", 0.5, 5.0)
    lam_den_end = trial.suggest_float("lambda_den_end", 0.05, 1.0)
    lam_cong_start = trial.suggest_float("lambda_cong_start", 0.05, 1.0)
    lam_cong_end = trial.suggest_float("lambda_cong_end", 0.3, 3.0)

    env = {
        "PLACER_LR": str(lr),
        "PLACER_NUM_STEPS": str(num_steps),
        "PLACER_LAMBDA_DEN_START": str(lam_den_start),
        "PLACER_LAMBDA_DEN_END": str(lam_den_end),
        "PLACER_LAMBDA_CONG_START": str(lam_cong_start),
        "PLACER_LAMBDA_CONG_END": str(lam_cong_end),
    }

    total = 0.0
    for bench in benchmarks:
        cost = run_benchmark(bench, env)
        total += cost
        # Prune early if clearly bad
        trial.report(total / (benchmarks.index(bench) + 1), benchmarks.index(bench))
        if trial.should_prune():
            raise optuna.TrialPruned()

    avg = total / len(benchmarks)

    print(
        f"  Trial {trial.number:3d}: avg={avg:.4f}  "
        f"lr={lr:.4f} steps={num_steps} "
        f"den=[{lam_den_start:.2f}→{lam_den_end:.2f}] "
        f"cong=[{lam_cong_start:.2f}→{lam_cong_end:.2f}]"
    )
    return avg


def main():
    parser = argparse.ArgumentParser(description="Optuna meta-tuner for V2 placer")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS,
                        help="Benchmarks to evaluate on")
    args = parser.parse_args()

    print(f"╔══════════════════════════════════════════════════════╗")
    print(f"║  Optuna Meta-Tuner for V2 Gradient Placer           ║")
    print(f"║  Benchmarks: {', '.join(args.benchmarks):<40s}║")
    print(f"║  Trials: {args.trials:<44d}║")
    print(f"╚══════════════════════════════════════════════════════╝")

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        study_name="v2_gradient_placer_tuning",
    )

    study.optimize(
        lambda trial: objective(trial, args.benchmarks),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    # ── Report ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    best = study.best_trial
    print(f"  Avg Proxy Cost: {best.value:.4f}")
    print(f"  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    print(f"\n  To use these params:")
    parts = " ".join(f"PLACER_{k.upper()}={v}" for k, v in best.params.items())
    print(f"    {parts} uv run evaluate {PLACER} --all")

    # ── Top 5 trials ─────────────────────────────────────────────────
    print(f"\nTOP 5 TRIALS:")
    for t in sorted(study.trials, key=lambda t: t.value if t.value else 999)[:5]:
        print(f"  #{t.number}: {t.value:.4f}  {t.params}")


if __name__ == "__main__":
    main()
