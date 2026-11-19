import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

TASK_DIR = os.path.dirname(
    CURRENT_DIR
)

RESULTS_DIR = os.path.join(
    TASK_DIR,
    "Results"
)

PLOTS_DIR = os.path.join(
    RESULTS_DIR,
    "Plots"
)

os.makedirs(
    PLOTS_DIR,
    exist_ok=True
)


# ============================================================
# FILES
# ============================================================

BENCHMARK_FILE = os.path.join(
    RESULTS_DIR,
    "benchmark_results.csv"
)

CONVERGENCE_FILE = os.path.join(
    RESULTS_DIR,
    "convergence_results.csv"
)


# ============================================================
# ALGORITHMS
# ============================================================

ALGORITHMS = [
    "GA",
    "cGA",
    "PSO",
    "ABC",
    "ACOR",
    "AFSA",
    "AIS",
    "BFOA",
    "FA",
    "GWO",
    "IWD",
    "IWO",
    "PFA",
    "WOA"
]


# ============================================================
# LOAD DATA
# ============================================================

results = pd.read_csv(
    BENCHMARK_FILE
)

convergence = pd.read_csv(
    CONVERGENCE_FILE
)


# ============================================================
# VALIDATE ALGORITHMS
# ============================================================

available_algorithms = set(
    results["algorithm"].dropna().unique()
)

missing_algorithms = [
    algorithm
    for algorithm in ALGORITHMS
    if algorithm not in available_algorithms
]

if missing_algorithms:
    raise ValueError(
        "The following algorithms are missing "
        "from benchmark_results.csv: "
        + ", ".join(missing_algorithms)
    )


print(
    "\nAlgorithms found:"
)

for algorithm in ALGORITHMS:
    print(
        f"  - {algorithm}"
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_figure(
    fig,
    filename
):
    path = os.path.join(
        PLOTS_DIR,
        filename
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"Saved: {path}"
    )


def get_boxplot_data(
    dataframe,
    column,
    log_scale=False
):
    data = []

    for algorithm in ALGORITHMS:

        values = (
            dataframe.loc[
                dataframe["algorithm"] == algorithm,
                column
            ]
            .dropna()
            .to_numpy()
        )

        if log_scale:

            values = np.clip(
                values,
                np.finfo(float).tiny,
                None
            )

        data.append(
            values
        )

    return data


def create_boxplot(
    dataframe,
    column,
    title,
    ylabel,
    filename,
    log_scale=False
):

    data = get_boxplot_data(
        dataframe,
        column,
        log_scale
    )

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    ax.boxplot(
        data,
        labels=ALGORITHMS
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "Algorithm"
    )

    ax.set_ylabel(
        ylabel
    )

    if log_scale:
        ax.set_yscale(
            "log"
        )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    ax.tick_params(
        axis="x",
        rotation=25
    )

    fig.tight_layout()

    save_figure(
        fig,
        filename
    )


# ============================================================
# 1. FINAL FITNESS
# ============================================================

create_boxplot(
    results,
    "best_fitness",
    "Final Fitness - Sphere 10D",
    "Best Fitness",
    "01_final_fitness_boxplot.png",
    log_scale=True
)


# ============================================================
# 2. RUNTIME
# ============================================================

create_boxplot(
    results,
    "runtime_seconds",
    "Runtime - Sphere 10D",
    "Runtime (seconds)",
    "02_runtime_boxplot.png"
)


# ============================================================
# 3. EVALUATIONS
# ============================================================

create_boxplot(
    results,
    "evaluations",
    "Function Evaluations - Sphere 10D",
    "Evaluations",
    "03_evaluations_boxplot.png"
)


# ============================================================
# 4. CPU TIME
# ============================================================

create_boxplot(
    results,
    "cpu_time_seconds",
    "CPU Time - Sphere 10D",
    "CPU Time (seconds)",
    "04_cpu_time_boxplot.png"
)


# ============================================================
# 5. CPU UTILIZATION
# ============================================================

create_boxplot(
    results,
    "cpu_utilization_percent",
    "CPU Utilization - Sphere 10D",
    "CPU Utilization (%)",
    "05_cpu_utilization_boxplot.png"
)


# ============================================================
# 6. PEAK MEMORY
# ============================================================

create_boxplot(
    results,
    "peak_memory_mb",
    "Peak Memory Usage - Sphere 10D",
    "Peak Memory (MB)",
    "06_peak_memory_boxplot.png"
)


# ============================================================
# 7. PEAK ADDITIONAL MEMORY
# ============================================================

create_boxplot(
    results,
    "peak_memory_delta_mb",
    "Peak Additional Memory - Sphere 10D",
    "Peak Additional Memory (MB)",
    "07_peak_additional_memory_boxplot.png"
)


# ============================================================
# 8. OPTIMALITY GAP
# ============================================================

results["optimality_gap"] = (
    np.abs(
        results["best_fitness"]
    )
)

create_boxplot(
    results,
    "optimality_gap",
    "Optimality Gap - Sphere 10D",
    "Optimality Gap",
    "08_optimality_gap_boxplot.png",
    log_scale=True
)


# ============================================================
# 9. SUCCESS RATE
# ============================================================

success_rate = (
    results
    .groupby("algorithm")["success"]
    .mean()
    .reindex(ALGORITHMS)
    * 100
)

fig, ax = plt.subplots(
    figsize=(13, 6)
)

bars = ax.bar(
    ALGORITHMS,
    success_rate.values
)

ax.set_title(
    "Success Rate - Sphere 10D"
)

ax.set_xlabel(
    "Algorithm"
)

ax.set_ylabel(
    "Success Rate (%)"
)

ax.set_ylim(
    0,
    100
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

ax.tick_params(
    axis="x",
    rotation=25
)

for bar, value in zip(
    bars,
    success_rate.values
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height()
        + 1,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=8
    )

fig.tight_layout()

save_figure(
    fig,
    "09_success_rate.png"
)


# ============================================================
# 10. MEAN CONVERGENCE
# ============================================================

fig, ax = plt.subplots(
    figsize=(13, 7)
)

for algorithm in ALGORITHMS:

    algorithm_data = (
        convergence[
            convergence["algorithm"] == algorithm
        ]
        .groupby("evaluation")["fitness"]
        .mean()
        .reset_index()
        .sort_values("evaluation")
    )

    if algorithm_data.empty:
        continue

    fitness = np.clip(
        algorithm_data["fitness"].to_numpy(),
        np.finfo(float).tiny,
        None
    )

    ax.plot(
        algorithm_data["evaluation"],
        fitness,
        label=algorithm,
        linewidth=1.5
    )


ax.set_title(
    "Mean Convergence - Sphere 10D"
)

ax.set_xlabel(
    "Function Evaluations"
)

ax.set_ylabel(
    "Fitness"
)

ax.set_yscale(
    "log"
)

ax.grid(
    True,
    which="both",
    linestyle="--",
    alpha=0.3
)

ax.legend(
    ncol=3,
    fontsize=9
)

fig.tight_layout()

save_figure(
    fig,
    "10_mean_convergence.png"
)


# ============================================================
# 11. MEAN CONVERGENCE ± STD
# ============================================================

fig, ax = plt.subplots(
    figsize=(13, 7)
)

for algorithm in ALGORITHMS:

    algorithm_data = (
        convergence[
            convergence["algorithm"] == algorithm
        ]
        .groupby("evaluation")["fitness"]
        .agg(
            ["mean", "std"]
        )
        .reset_index()
        .sort_values("evaluation")
    )

    if algorithm_data.empty:
        continue

    evaluation = (
        algorithm_data["evaluation"]
        .to_numpy()
    )

    mean = np.clip(
        algorithm_data["mean"].to_numpy(),
        np.finfo(float).tiny,
        None
    )

    std = (
        algorithm_data["std"]
        .fillna(0)
        .to_numpy()
    )

    lower = np.clip(
        mean - std,
        np.finfo(float).tiny,
        None
    )

    upper = np.maximum(
        mean + std,
        np.finfo(float).tiny
    )

    ax.plot(
        evaluation,
        mean,
        label=algorithm,
        linewidth=1.5
    )

    ax.fill_between(
        evaluation,
        lower,
        upper,
        alpha=0.10
    )


ax.set_title(
    "Mean Convergence ± Standard Deviation - Sphere 10D"
)

ax.set_xlabel(
    "Function Evaluations"
)

ax.set_ylabel(
    "Fitness"
)

ax.set_yscale(
    "log"
)

ax.grid(
    True,
    which="both",
    linestyle="--",
    alpha=0.3
)

ax.legend(
    ncol=3,
    fontsize=9
)

fig.tight_layout()

save_figure(
    fig,
    "11_mean_convergence_std.png"
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

summary = (
    results
    .groupby("algorithm")
    .agg(
        mean_fitness=(
            "best_fitness",
            "mean"
        ),

        std_fitness=(
            "best_fitness",
            "std"
        ),

        best_fitness=(
            "best_fitness",
            "min"
        ),

        worst_fitness=(
            "best_fitness",
            "max"
        ),

        mean_runtime=(
            "runtime_seconds",
            "mean"
        ),

        std_runtime=(
            "runtime_seconds",
            "std"
        ),

        mean_evaluations=(
            "evaluations",
            "mean"
        ),

        std_evaluations=(
            "evaluations",
            "std"
        ),

        mean_cpu_time=(
            "cpu_time_seconds",
            "mean"
        ),

        std_cpu_time=(
            "cpu_time_seconds",
            "std"
        ),

        mean_cpu_utilization=(
            "cpu_utilization_percent",
            "mean"
        ),

        std_cpu_utilization=(
            "cpu_utilization_percent",
            "std"
        ),

        mean_peak_memory=(
            "peak_memory_mb",
            "mean"
        ),

        mean_peak_additional_memory=(
            "peak_memory_delta_mb",
            "mean"
        ),

        std_peak_additional_memory=(
            "peak_memory_delta_mb",
            "std"
        ),

        success_rate=(
            "success",
            "mean"
        )
    )
    .reindex(ALGORITHMS)
)

summary["success_rate"] *= 100


# ============================================================
# 12. MEAN FINAL FITNESS
# ============================================================

fig, ax = plt.subplots(
    figsize=(13, 6)
)

bars = ax.bar(
    ALGORITHMS,
    summary["mean_fitness"],
    yerr=summary["std_fitness"],
    capsize=4
)

ax.set_title(
    "Mean Final Fitness ± Std - Sphere 10D"
)

ax.set_xlabel(
    "Algorithm"
)

ax.set_ylabel(
    "Mean Final Fitness"
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

ax.tick_params(
    axis="x",
    rotation=25
)

fig.tight_layout()

save_figure(
    fig,
    "12_mean_final_fitness.png"
)


# ============================================================
# 13. MEAN RUNTIME
# ============================================================

fig, ax = plt.subplots(
    figsize=(13, 6)
)

bars = ax.bar(
    ALGORITHMS,
    summary["mean_runtime"],
    yerr=summary["std_runtime"],
    capsize=4
)

ax.set_title(
    "Mean Runtime ± Std - Sphere 10D"
)

ax.set_xlabel(
    "Algorithm"
)

ax.set_ylabel(
    "Runtime (seconds)"
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

ax.tick_params(
    axis="x",
    rotation=25
)

fig.tight_layout()

save_figure(
    fig,
    "13_mean_runtime.png"
)


# ============================================================
# 14. MEAN EVALUATIONS
# ============================================================

fig, ax = plt.subplots(
    figsize=(13, 6)
)

bars = ax.bar(
    ALGORITHMS,
    summary["mean_evaluations"],
    yerr=summary["std_evaluations"],
    capsize=4
)

ax.set_title(
    "Mean Function Evaluations ± Std - Sphere 10D"
)

ax.set_xlabel(
    "Algorithm"
)

ax.set_ylabel(
    "Function Evaluations"
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

ax.tick_params(
    axis="x",
    rotation=25
)

fig.tight_layout()

save_figure(
    fig,
    "14_mean_evaluations.png"
)


# ============================================================
# 15. MEAN CPU TIME
# ============================================================

fig, ax = plt.subplots(
    figsize=(13, 6)
)

bars = ax.bar(
    ALGORITHMS,
    summary["mean_cpu_time"],
    yerr=summary["std_cpu_time"],
    capsize=4
)

ax.set_title(
    "Mean CPU Time ± Std - Sphere 10D"
)

ax.set_xlabel(
    "Algorithm"
)

ax.set_ylabel(
    "CPU Time (seconds)"
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

ax.tick_params(
    axis="x",
    rotation=25
)

fig.tight_layout()

save_figure(
    fig,
    "15_mean_cpu_time.png"
)


# ============================================================
# 16. MEAN PEAK ADDITIONAL MEMORY
# ============================================================

fig, ax = plt.subplots(
    figsize=(13, 6)
)

bars = ax.bar(
    ALGORITHMS,
    summary["mean_peak_additional_memory"],
    yerr=summary["std_peak_additional_memory"],
    capsize=4
)

ax.set_title(
    "Mean Peak Additional Memory ± Std - Sphere 10D"
)

ax.set_xlabel(
    "Algorithm"
)

ax.set_ylabel(
    "Peak Additional Memory (MB)"
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

ax.tick_params(
    axis="x",
    rotation=25
)

fig.tight_layout()

save_figure(
    fig,
    "16_mean_peak_additional_memory.png"
)


# ============================================================
# SAVE COMPLETE SUMMARY
# ============================================================

summary_file = os.path.join(
    PLOTS_DIR,
    "sphere_complete_summary.csv"
)

summary.to_csv(
    summary_file
)

print(
    f"\nSummary saved: {summary_file}"
)


# ============================================================
# SHOW ALL FIGURES
# ============================================================

plt.show()