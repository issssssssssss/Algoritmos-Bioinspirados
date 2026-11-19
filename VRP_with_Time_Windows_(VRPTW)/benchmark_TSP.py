"""Benchmark común para los algoritmos del problema VRPTW."""

import os
import platform
import threading
import time

import numpy as np
import pandas as pd

try:
    import psutil
except ImportError as exc:
    raise ImportError(
        "Este benchmark requiere psutil. Instálalo con: pip install psutil"
    ) from exc

from VRPTW_Problem import generate_instance
from VRPTW_ABC import run_abc
from VRPTW_ACO import run_aco
from VRPTW_ACOR import run_acor
from VRPTW_AFSA import run_afsa
from VRPTW_AIS import run_ais
from VRPTW_BFOA import run_bfoa
from VRPTW_FA import run_fa
from VRPTW_GA import run_ga
from VRPTW_GWO import run_gwo
from VRPTW_IWD import run_iwd
from VRPTW_IWO import run_iwo
from VRPTW_PFA import run_pfa
from VRPTW_PSO import run_pso
from VRPTW_WOA import run_woa
from VRPTW_cGA import run_cga


N_CUSTOMERS = 25
INSTANCE_SEED = 42
N_RUNS = 30
EVAL_BUDGET = 60_000
BITS_PER_KEY = 16
POP_SIZE = 30

ALGORITHM_ORDER = [
    "GA", "cGA", "PSO", "ABC", "ACO", "ACOR", "GWO", "WOA",
    "FA", "AFSA", "AIS", "BFOA", "IWO", "PFA", "IWD",
]

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "Results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "benchmark_results.csv")
CONVERGENCE_FILE = os.path.join(RESULTS_DIR, "convergence_results.csv")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "benchmark_summary.csv")


def iterations_for_budget(evaluations_per_iteration, initial_evaluations=0):
    return max(1, (EVAL_BUDGET - initial_evaluations) // evaluations_per_iteration)


def build_algorithm_configs():
    """Devuelve la función y parámetros de cada algoritmo VRPTW."""
    return {
        "GA": (run_ga, {
            "pop_size": 150,
            "n_generations": iterations_for_budget(150, 150),
        }),
        "cGA": (run_cga, {
            "bits_per_key": BITS_PER_KEY,
            "pop_size": 100,
            "n_generations": iterations_for_budget(2),
        }),
        "PSO": (run_pso, {
            "n_particles": 80,
            "n_iterations": iterations_for_budget(80, 80),
        }),
        "ABC": (run_abc, {
            "pop_size": POP_SIZE,
            "n_iterations": iterations_for_budget(2 * POP_SIZE, POP_SIZE),
            "limit": 50,
        }),
        "ACO": (run_aco, {
            "n_ants": POP_SIZE,
            "n_iterations": iterations_for_budget(POP_SIZE),
            "alpha": 1.0,
            "beta": 3.0,
            "rho": 0.3,
            "Q": 100.0,
        }),
        "ACOR": (run_acor, {
            "archive_size": POP_SIZE,
            "n_ants": POP_SIZE,
            "n_iterations": iterations_for_budget(POP_SIZE, POP_SIZE),
        }),
        "GWO": (run_gwo, {
            "pop_size": POP_SIZE,
            "n_iterations": iterations_for_budget(POP_SIZE, POP_SIZE),
        }),
        "WOA": (run_woa, {
            "pop_size": POP_SIZE,
            "n_iterations": iterations_for_budget(POP_SIZE, POP_SIZE),
            "b": 1.0,
        }),
        "FA": (run_fa, {
            "pop_size": POP_SIZE,
            "n_iterations": iterations_for_budget(POP_SIZE, POP_SIZE),
        }),
        "AFSA": (run_afsa, {
            "pop_size": POP_SIZE,
            "n_iterations": iterations_for_budget(POP_SIZE, POP_SIZE),
        }),
        "AIS": (run_ais, {
            "pop_size": POP_SIZE,
            "n_iterations": iterations_for_budget(POP_SIZE, POP_SIZE),
        }),
        "BFOA": (run_bfoa, {
            "pop_size": POP_SIZE,
            "n_iterations": iterations_for_budget(POP_SIZE, POP_SIZE),
        }),
        "IWO": (run_iwo, {
            "initial_pop_size": 10,
            "max_pop_size": POP_SIZE,
            "n_iterations": iterations_for_budget(POP_SIZE, 10),
        }),
        "PFA": (run_pfa, {
            "pop_size": POP_SIZE,
            "n_iterations": iterations_for_budget(POP_SIZE, POP_SIZE),
        }),
        "IWD": (run_iwd, {
            "n_drops": POP_SIZE,
            "n_iterations": iterations_for_budget(POP_SIZE),
        }),
    }


class MemoryMonitor:
    def __init__(self, process, interval=0.01):
        self.process = process
        self.interval = interval
        self.running = False
        self.thread = None
        self.baseline_mb = 0.0
        self.peak_mb = 0.0

    def start(self):
        self.baseline_mb = self.process.memory_info().rss / (1024 ** 2)
        self.peak_mb = self.baseline_mb
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def _monitor(self):
        while self.running:
            try:
                current_mb = self.process.memory_info().rss / (1024 ** 2)
                self.peak_mb = max(self.peak_mb, current_mb)
            except psutil.Error:
                break
            time.sleep(self.interval)

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        return self.baseline_mb, self.peak_mb


def run_with_resources(algorithm_call):
    process = psutil.Process(os.getpid())
    memory_monitor = MemoryMonitor(process)
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    memory_monitor.start()
    result = algorithm_call()
    baseline_mb, peak_mb = memory_monitor.stop()
    cpu_time = time.process_time() - cpu_start
    runtime = time.perf_counter() - wall_start
    peak_delta_mb = max(0.0, peak_mb - baseline_mb)
    cpu_count = psutil.cpu_count(logical=True) or 1
    cpu_utilization = cpu_time / runtime / cpu_count * 100 if runtime > 0 else 0.0
    return (
        result,
        runtime,
        cpu_time,
        baseline_mb,
        peak_mb,
        peak_delta_mb,
        cpu_utilization,
    )


def normalize_result(result):
    """Extrae solución, fitness e historia de cualquier algoritmo VRPTW."""
    if len(result) < 3:
        raise ValueError(
            "El algoritmo debe devolver solución, fitness e historia"
        )
    return result[0], float(result[1]), np.asarray(result[2], dtype=float)


def run_benchmark():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    instance = generate_instance(
        n_customers=N_CUSTOMERS,
        seed=INSTANCE_SEED,
    )
    configs = build_algorithm_configs()
    benchmark_results = []
    convergence_results = []

    print("=" * 75)
    print("VRPTW BENCHMARK")
    print("=" * 75)
    print(f"Customers:           {N_CUSTOMERS}")
    print(f"Instance seed:       {INSTANCE_SEED}")
    print(f"Runs per algorithm:  {N_RUNS}")
    print(f"Evaluation budget:   {EVAL_BUDGET}")
    print("=" * 75)

    for seed in range(N_RUNS):
        print(f"\nRUN {seed + 1}/{N_RUNS} | SEED = {seed}")

        for algorithm in ALGORITHM_ORDER:
            run_function, parameters = configs[algorithm]
            print(f"Running {algorithm}...")
            kwargs = {
                "instance": instance,
                "n_customers": N_CUSTOMERS,
                **parameters,
                "seed": seed,
                "verbose": False,
            }
            measured = run_with_resources(
                lambda: run_function(**kwargs)
            )
            (
                result,
                runtime,
                cpu_time,
                baseline_mb,
                peak_mb,
                peak_delta_mb,
                cpu_utilization,
            ) = measured
            _, best_fitness, history = normalize_result(result)
            evaluations = list(
                getattr(run_function, "evaluation_history", [])
            )
            if not evaluations:
                raise RuntimeError(
                    f"{algorithm} no publicó evaluation_history"
                )

            benchmark_results.append({
                "algorithm": algorithm,
                "problem": "VRPTW",
                "dimension": N_CUSTOMERS,
                "seed": seed,
                "best_fitness": best_fitness,
                "runtime_seconds": runtime,
                "cpu_time_seconds": cpu_time,
                "cpu_utilization_percent": cpu_utilization,
                "memory_baseline_mb": baseline_mb,
                "peak_memory_mb": peak_mb,
                "peak_memory_delta_mb": peak_delta_mb,
                "evaluations": evaluations[-1],
                "system": platform.system(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(logical=True),
            })

            for evaluation, fitness in zip(evaluations, history):
                convergence_results.append({
                    "algorithm": algorithm,
                    "problem": "VRPTW",
                    "dimension": N_CUSTOMERS,
                    "seed": seed,
                    "evaluation": evaluation,
                    "fitness": fitness,
                })

    results_df = pd.DataFrame(benchmark_results)
    convergence_df = pd.DataFrame(convergence_results)
    results_df.to_csv(RESULTS_FILE, index=False)
    convergence_df.to_csv(CONVERGENCE_FILE, index=False)

    summary = results_df.groupby("algorithm").agg(
        mean_fitness=("best_fitness", "mean"),
        median_fitness=("best_fitness", "median"),
        std_fitness=("best_fitness", "std"),
        best_fitness=("best_fitness", "min"),
        worst_fitness=("best_fitness", "max"),
        mean_runtime=("runtime_seconds", "mean"),
        std_runtime=("runtime_seconds", "std"),
        mean_cpu_time=("cpu_time_seconds", "mean"),
        std_cpu_time=("cpu_time_seconds", "std"),
        mean_cpu_utilization=("cpu_utilization_percent", "mean"),
        std_cpu_utilization=("cpu_utilization_percent", "std"),
        mean_peak_memory=("peak_memory_mb", "mean"),
        std_peak_memory=("peak_memory_mb", "std"),
        mean_peak_memory_delta=("peak_memory_delta_mb", "mean"),
        std_peak_memory_delta=("peak_memory_delta_mb", "std"),
        mean_evaluations=("evaluations", "mean"),
        std_evaluations=("evaluations", "std"),
    ).reindex(ALGORITHM_ORDER).reset_index()
    summary.to_csv(SUMMARY_FILE, index=False)

    print("\nBENCHMARK COMPLETED")
    print(summary.to_string(index=False))
    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"Convergence saved to: {CONVERGENCE_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")


if __name__ == "__main__":
    run_benchmark()
