import os
import sys
import time
import threading
import numpy as np
import pandas as pd
import psutil

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from Knapsack_GA import run_ga
from Knapsack_cGA import run_cga
from Knapsack_PSO import run_pso
from Knapsack_ABC import run_abc
from Knapsack_ACOR import run_acor
from Knapsack_AFSA import run_afsa
from Knapsack_AIS import run_ais
from Knapsack_BFOA import run_bfoa
from Knapsack_FA import run_fa
from Knapsack_GWO import run_gwo
from Knapsack_IWD import run_iwd
from Knapsack_IWO import run_iwo
from Knapsack_PFA import run_pfa
from Knapsack_WOA import run_woa

from knapsack_common import (
    DEFAULT_ITEMS,
    DEFAULT_CAPACITY,
    reset,
)

RESULTS_DIR = os.path.join(CURRENT_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BENCHMARK_RESULTS = os.path.join(RESULTS_DIR, "benchmark_results.csv")
CONVERGENCE_RESULTS = os.path.join(RESULTS_DIR, "convergence_results.csv")

EVAL_BUDGET = 20000
N_RUNS = 30
SEEDS = range(30)

ITEMS = DEFAULT_ITEMS
CAPACITY = DEFAULT_CAPACITY
N_DIM = len(ITEMS)

POP_SIZE = 50


def run_with_resources(func, *args, **kwargs):
    process = psutil.Process(os.getpid())
    stop_monitor = threading.Event()
    peak_memory = [process.memory_info().rss]

    def monitor():
        while not stop_monitor.is_set():
            try:
                memory = process.memory_info().rss
                if memory > peak_memory[0]:
                    peak_memory[0] = memory
            except psutil.Error:
                break
            time.sleep(0.005)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    cpu_before = process.cpu_times()
    start = time.perf_counter()
    monitor_thread.start()

    try:
        result = func(*args, **kwargs)
    finally:
        stop_monitor.set()
        monitor_thread.join(timeout=1)

    elapsed = time.perf_counter() - start
    cpu_after = process.cpu_times()

    cpu_time = (
        (cpu_after.user - cpu_before.user) +
        (cpu_after.system - cpu_before.system)
    )

    return result, elapsed, cpu_time, peak_memory[0] / (1024 ** 2)


def save_results(rows):
    pd.DataFrame(rows).to_csv(BENCHMARK_RESULTS, index=False)


def save_convergence(rows):
    pd.DataFrame(rows).to_csv(CONVERGENCE_RESULTS, index=False)


def budget_iterations(population, multiplier=1):
    return max(
        1,
        (EVAL_BUDGET - population) // max(1, population * multiplier)
    )


def run_algorithm(name, runner, kwargs, seed):
    reset()

    result, runtime, cpu_time, peak_memory = run_with_resources(
        runner,
        seed=seed,
        verbose=False,
        **kwargs
    )

    if len(result) == 5:
        best_x, best_f, best_history, mean_history, best_position_history = result
    elif len(result) == 4:
        best_x, best_f, best_history, best_position_history = result
        mean_history = None
    else:
        raise ValueError(f"Unsupported return format from {name}: expected 4 or 5 values, got {len(result)}")

    evaluations = int(best_history[-1][0]) if best_history and isinstance(best_history[-1], (tuple, list)) else None

    if evaluations is None:
        try:
            from knapsack_common import evaluation_history
            evaluations = int(evaluation_history[-1])
        except Exception:
            evaluations = 0

    best_value = float(-best_f)

    convergence_rows = []
    for i, value in enumerate(best_history):
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            convergence_rows.append({
                "algorithm": name,
                "seed": seed,
                "iteration": i,
                "evaluation": int(value[0]),
                "best_fitness": float(value[1]),
                "best_value": float(-value[1])
            })
        else:
            convergence_rows.append({
                "algorithm": name,
                "seed": seed,
                "iteration": i,
                "evaluation": i,
                "best_fitness": float(value),
                "best_value": float(-value)
            })

    return {
        "algorithm": name,
        "problem": "Knapsack",
        "dimension": N_DIM,
        "seed": seed,
        "best_value": best_value,
        "best_fitness": float(best_f),
        "evaluations": evaluations,
        "runtime_seconds": runtime,
        "cpu_seconds": cpu_time,
        "peak_memory_mb": peak_memory,
        "success": int(best_value >= 27)
    }, convergence_rows


ALGORITHMS = {
    "GA": (run_ga, {
        "n_dim": N_DIM,
        "pop_size": 100,
        "n_generations": budget_iterations(100),
        "items": ITEMS,
        "capacity": CAPACITY,
    }),
    "cGA": (run_cga, {
        "n_dim": N_DIM,
        "pop_size": 100,
        "n_generations": max(1, EVAL_BUDGET // 2),
        "items": ITEMS,
        "capacity": CAPACITY,
    }),
    "PSO": (run_pso, {
        "n_dim": N_DIM,
        "n_particles": 50,
        "n_iterations": budget_iterations(50),
        "items": ITEMS,
        "capacity": CAPACITY,
    }),
    "ABC": (run_abc, {
        "n_dim": N_DIM,
        "pop_size": 50,
        "n_iterations": budget_iterations(50, 2),
        "items": ITEMS,
        "capacity": CAPACITY,
    }),
    "ACOR": (run_acor, {
        "n_dim": N_DIM,
        "n_ants": 50,
        "archive_size": 50,
        "n_iterations": budget_iterations(50),
        "items": ITEMS,
        "capacity": CAPACITY,
    }),
    "AFSA": (run_afsa, {
        "n_dim": N_DIM,
        "pop_size": 30,
        "n_iterations": budget_iterations(30),
        "items": ITEMS,
        "capacity": CAPACITY,
    }),
    "AIS": (run_ais, {
        "n_dim": N_DIM,
        "pop_size": 30,
        "n_iterations": budget_iterations(30),
        "items": ITEMS,
        "capacity": CAPACITY,
    }),
    "BFOA": (run_bfoa, {
        "n_dim": N_DIM,
        "pop_size": 30,
        "n_iterations": budget_iterations(30),
        "items": ITEMS,
        "capacity": CAPACITY,
    }),
    "FA": (run_fa, {
        "n_dim": N_DIM,
        "pop_size": 30,
        "n_iterations": budget_iterations(30),
        "items": ITEMS,
        "capacity": CAPACITY,
    }),
    "GWO": (run_gwo, {
        "n_dim": N_DIM,
        "pop_size": 50,
        "n_iterations": budget_iterations(50),
        "items": ITEMS,
        "capacity": CAPACITY,
    }),
    "IWD": (run_iwd, {
        "n_dim": N_DIM,
        "pop_size": 30,
        "n_iterations": budget_iterations(30),
        "n_candidates": 5,
        "items": ITEMS,
        "capacity": CAPACITY,
    }),
    "IWO": (run_iwo, {
        "n_dim": N_DIM,
        "pop_size": 30,
        "n_iterations": budget_iterations(30),
        "items": ITEMS,
        "capacity": CAPACITY,
    }),
    "PFA": (run_pfa, {
        "n_dim": N_DIM,
        "pop_size": 30,
        "n_iterations": budget_iterations(30),
        "items": ITEMS,
        "capacity": CAPACITY,
    }),
    "WOA": (run_woa, {
        "n_dim": N_DIM,
        "pop_size": 50,
        "n_iterations": budget_iterations(50),
        "items": ITEMS,
        "capacity": CAPACITY,
    }),
}


def main():
    benchmark_rows = []
    convergence_rows = []

    for name, (runner, kwargs) in ALGORITHMS.items():
        for seed in SEEDS:
            print(f"{name} - seed {seed + 1}/{N_RUNS}")

            try:
                row, convergence = run_algorithm(
                    name,
                    runner,
                    kwargs,
                    seed
                )
                benchmark_rows.append(row)
                convergence_rows.extend(convergence)
            except Exception as exc:
                print(f"Error en {name}, seed {seed}: {exc}")

    final_df = pd.DataFrame(benchmark_rows)

    if convergence_rows:
        convergence_df = pd.DataFrame(convergence_rows)
    else:
        convergence_df = pd.DataFrame(columns=[
            "algorithm", "seed", "iteration", "evaluation",
            "best_fitness", "best_value"
        ])

    final_df.to_csv(BENCHMARK_RESULTS, index=False)
    convergence_df.to_csv(CONVERGENCE_RESULTS, index=False)

    summary_df = (
        final_df.groupby("algorithm").agg(
            mean_best_value=("best_value", "mean"),
            median_best_value=("best_value", "median"),
            std_best_value=("best_value", "std"),
            best_best_value=("best_value", "min"),
            worst_best_value=("best_value", "max"),
            mean_runtime=("runtime_seconds", "mean"),
            std_runtime=("runtime_seconds", "std"),
            mean_cpu_time=("cpu_seconds", "mean"),
            std_cpu_time=("cpu_seconds", "std"),
            mean_peak_memory=("peak_memory_mb", "mean"),
            std_peak_memory=("peak_memory_mb", "std"),
            mean_evaluations=("evaluations", "mean"),
            std_evaluations=("evaluations", "std"),
            success_rate=("success", "mean"),
        ).reset_index()
    )

    summary_df["success_rate_percent"] = summary_df["success_rate"] * 100

    summary_path = os.path.join(RESULTS_DIR, "benchmark_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\nResultados guardados en:")
    print(BENCHMARK_RESULTS)
    print(CONVERGENCE_RESULTS)
    print(summary_path)

    print("\nResumen estadístico:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
