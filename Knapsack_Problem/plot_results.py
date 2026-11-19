import os
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "Results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "Plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

BENCHMARK_RESULTS = os.path.join(RESULTS_DIR, "benchmark_results.csv")
CONVERGENCE_RESULTS = os.path.join(RESULTS_DIR, "convergence_results.csv")


def load_results():
    benchmark = pd.read_csv(BENCHMARK_RESULTS)
    convergence = pd.read_csv(CONVERGENCE_RESULTS)
    return benchmark, convergence


def plot_best_value(benchmark):
    summary = benchmark.groupby("algorithm")["best_value"].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    summary.plot(kind="bar")
    plt.xlabel("Algorithm")
    plt.ylabel("Mean best value")
    plt.title("Knapsack - Mean best value")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "best_value.png"), dpi=300)
    plt.show()


def plot_runtime(benchmark):
    summary = benchmark.groupby("algorithm")["runtime_seconds"].mean().sort_values()

    plt.figure(figsize=(12, 6))
    summary.plot(kind="bar")
    plt.xlabel("Algorithm")
    plt.ylabel("Mean runtime (s)")
    plt.title("Knapsack - Mean runtime")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "runtime.png"), dpi=300)
    plt.show()


def plot_evaluations(benchmark):
    summary = benchmark.groupby("algorithm")["evaluations"].mean().sort_values()

    plt.figure(figsize=(12, 6))
    summary.plot(kind="bar")
    plt.xlabel("Algorithm")
    plt.ylabel("Mean evaluations")
    plt.title("Knapsack - Mean evaluations")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "evaluations.png"), dpi=300)
    plt.show()


def plot_memory(benchmark):
    summary = benchmark.groupby("algorithm")["peak_memory_mb"].mean().sort_values()

    plt.figure(figsize=(12, 6))
    summary.plot(kind="bar")
    plt.xlabel("Algorithm")
    plt.ylabel("Mean peak memory (MB)")
    plt.title("Knapsack - Mean peak memory")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "memory.png"), dpi=300)
    plt.show()


def plot_success(benchmark):
    summary = benchmark.groupby("algorithm")["success"].mean() * 100

    plt.figure(figsize=(12, 6))
    summary.plot(kind="bar")
    plt.xlabel("Algorithm")
    plt.ylabel("Success rate (%)")
    plt.title("Knapsack - Success rate")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "success_rate.png"), dpi=300)
    plt.show()


def plot_convergence(convergence):
    plt.figure(figsize=(12, 7))

    for algorithm, data in convergence.groupby("algorithm"):
        grouped = data.groupby("evaluation")["best_value"].mean()
        plt.plot(grouped.index, grouped.values, label=algorithm)

    plt.xlabel("Evaluations")
    plt.ylabel("Mean best value")
    plt.title("Knapsack - Convergence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "convergence.png"), dpi=300)
    plt.show()


def main():
    benchmark, convergence = load_results()

    plot_best_value(benchmark)
    plot_runtime(benchmark)
    plot_evaluations(benchmark)
    plot_memory(benchmark)
    plot_success(benchmark)
    plot_convergence(convergence)

    print(f"Plots saved in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
