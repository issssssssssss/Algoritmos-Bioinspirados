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

RESULTS_DIR = os.path.join(
    CURRENT_DIR,
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

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "benchmark_results.csv"
)

CONVERGENCE_FILE = os.path.join(
    RESULTS_DIR,
    "convergence_results.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

results = pd.read_csv(
    RESULTS_FILE
)

convergence = pd.read_csv(
    CONVERGENCE_FILE
)


print(
    f"Benchmark results: "
    f"{len(results)} rows"
)

print(
    f"Convergence results: "
    f"{len(convergence)} rows"
)


# ============================================================
# ALGORITHM ORDER
# ============================================================

ALGORITHMS = [
    "GA",
    "cGA",
    "PSO",
    "ABC",
    "ACO",
    "GWO",
    "WOA",
    "FA",
    "AFSA",
    "AIS",
    "BFOA",
    "IWO",
    "PFA",
    "IWD"
]


# ============================================================
# VERIFY AVAILABLE ALGORITHMS
# ============================================================

available_algorithms = set(
    results["algorithm"].unique()
)

missing_algorithms = [
    algorithm
    for algorithm in ALGORITHMS
    if algorithm not in available_algorithms
]

if missing_algorithms:

    print()
    print(
        "Warning: The following algorithms "
        "were not found in the results:"
    )

    for algorithm in missing_algorithms:

        print(
            f"  - {algorithm}"
        )

    ALGORITHMS = [
        algorithm
        for algorithm in ALGORITHMS
        if algorithm in available_algorithms
    ]


print()
print(
    "Algorithms included:"
)

for algorithm in ALGORITHMS:

    print(
        f"  - {algorithm}"
    )


# ============================================================
# SAVE FIGURE
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


def convergence_statistics(data, points=100):

    runs = []

    for _, run in data.groupby("seed"):
        run = (
            run
            .sort_values("evaluation")
            .drop_duplicates("evaluation")
        )
        if len(run) >= 2:
            runs.append(run)

    if not runs:
        return pd.DataFrame(columns=["evaluation", "mean", "std"])

    start = max(run["evaluation"].iloc[0] for run in runs)
    end = min(run["evaluation"].iloc[-1] for run in runs)

    if start >= end:
        return pd.DataFrame(columns=["evaluation", "mean", "std"])

    grid = np.linspace(
        start,
        end,
        min(points, max(2, int(end - start + 1)))
    )

    values = np.asarray([
        np.interp(
            grid,
            run["evaluation"].to_numpy(dtype=float),
            run["fitness"].to_numpy(dtype=float)
        )
        for run in runs
    ])

    return pd.DataFrame({
        "evaluation": grid,
        "mean": values.mean(axis=0),
        "std": values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(len(grid))
    })


# ============================================================
# BOXPLOT FUNCTION
# ============================================================

def create_boxplot(
    column,
    ylabel,
    title,
    filename,
    log_scale=False
):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    data = [

        results.loc[
            results["algorithm"] == algorithm,
            column
        ]
        .dropna()
        .values

        for algorithm in ALGORITHMS

    ]

    ax.boxplot(
        data,
        labels=ALGORITHMS
    )

    if log_scale:

        ax.set_yscale(
            "log"
        )

    ax.set_xlabel(
        "Algorithm"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    save_figure(
        fig,
        filename
    )

    return fig


# ============================================================
# 1. FINAL TOUR LENGTH
# ============================================================

create_boxplot(

    column="best_fitness",

    ylabel="Final tour length",

    title=(
        "Final Tour Length Distribution — "
        "TSP (25 Cities)"
    ),

    filename="01_fitness_boxplot.png"

)


# ============================================================
# 2. RUNTIME
# ============================================================

create_boxplot(

    column="runtime_seconds",

    ylabel="Runtime (seconds)",

    title=(
        "Runtime Distribution — "
        "TSP (25 Cities)"
    ),

    filename="02_runtime_boxplot.png"

)


# ============================================================
# 3. FUNCTION EVALUATIONS
# ============================================================

create_boxplot(

    column="evaluations",

    ylabel="Function evaluations",

    title=(
        "Function Evaluations — "
        "TSP (25 Cities)"
    ),

    filename="03_evaluations_boxplot.png"

)


# ============================================================
# 4. CPU TIME
# ============================================================

create_boxplot(

    column="cpu_time_seconds",

    ylabel="CPU time (seconds)",

    title=(
        "CPU Time Distribution — "
        "TSP (25 Cities)"
    ),

    filename="04_cpu_time_boxplot.png"

)


# ============================================================
# 5. CPU UTILIZATION
# ============================================================

create_boxplot(

    column="cpu_utilization_percent",

    ylabel="Average CPU utilization (%)",

    title=(
        "CPU Utilization — "
        "TSP (25 Cities)"
    ),

    filename="05_cpu_utilization_boxplot.png"

)


# ============================================================
# 6. PEAK MEMORY
# ============================================================

create_boxplot(

    column="peak_memory_mb",

    ylabel="Peak memory (MB)",

    title=(
        "Peak Memory Usage — "
        "TSP (25 Cities)"
    ),

    filename="06_peak_memory_boxplot.png"

)


# ============================================================
# 7. ADDITIONAL MEMORY
# ============================================================

create_boxplot(

    column="peak_memory_delta_mb",

    ylabel="Peak additional memory (MB)",

    title=(
        "Peak Additional Memory — "
        "TSP (25 Cities)"
    ),

    filename="07_memory_delta_boxplot.png"

)


# ============================================================
# 8. MEAN CONVERGENCE
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 7)
)


for algorithm in ALGORITHMS:

    data = convergence[
        convergence["algorithm"] == algorithm
    ]

    if data.empty:

        continue

    grouped = convergence_statistics(data)

    ax.plot(

        grouped["evaluation"].to_numpy(dtype=float),

        grouped["mean"].to_numpy(dtype=float),

        label=algorithm

    )


ax.set_xlabel(
    "Function evaluations"
)

ax.set_ylabel(
    "Mean best-so-far tour length"
)

ax.set_title(
    "Mean Convergence — TSP (25 Cities)"
)

ax.legend()

ax.grid(
    True,
    alpha=0.25
)

fig.tight_layout()

save_figure(
    fig,
    "08_convergence_mean.png"
)


# ============================================================
# 9. MEAN CONVERGENCE ± STD
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 7)
)


for algorithm in ALGORITHMS:

    data = convergence[
        convergence["algorithm"] == algorithm
    ]

    if data.empty:

        continue

    grouped = convergence_statistics(data)


    lower = (

        grouped["mean"]
        - grouped["std"]

    )

    upper = (

        grouped["mean"]
        + grouped["std"]

    )


    lower = lower.clip(
        lower=0
    )


    ax.plot(

        grouped["evaluation"].to_numpy(dtype=float),

        grouped["mean"].to_numpy(dtype=float),

        label=algorithm

    )


    ax.fill_between(

        grouped["evaluation"].to_numpy(dtype=float),

        lower.to_numpy(dtype=float),

        upper.to_numpy(dtype=float),

        alpha=0.12

    )


ax.set_xlabel(
    "Function evaluations"
)

ax.set_ylabel(
    "Tour length"
)

ax.set_title(
    "Mean Convergence ± Standard Deviation — TSP (25 Cities)"
)

ax.legend(
    ncol=2
)

ax.grid(
    True,
    alpha=0.25
)

fig.tight_layout()

save_figure(
    fig,
    "09_convergence_std.png"
)


# ============================================================
# FUNCTION FOR MEAN BARPLOTS
# ============================================================

def create_mean_barplot(

    column,
    ylabel,
    title,
    filename,
    formatter=None

):

    summary = (

        results
        .groupby("algorithm")[column]
        .agg(
            mean="mean",
            std="std"
        )
        .reindex(ALGORITHMS)

    )


    fig, ax = plt.subplots(
        figsize=(10, 6)
    )


    x = np.arange(
        len(ALGORITHMS)
    )


    bars = ax.bar(

        x,

        summary["mean"].values,

        yerr=summary["std"]
        .fillna(0)
        .values,

        capsize=5

    )


    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        ALGORITHMS
    )

    ax.set_xlabel(
        "Algorithm"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )


    # --------------------------------------------------------
    # VALUE LABELS
    # --------------------------------------------------------

    y_offset = (
        summary["mean"].max()
        * 0.01
    )

    for bar, value in zip(

        bars,

        summary["mean"].values

    ):

        if pd.isna(value):

            continue


        if formatter is not None:

            text = formatter(
                value
            )

        else:

            text = f"{value:.4g}"


        ax.text(

            bar.get_x()
            + bar.get_width() / 2,

            value + y_offset,

            text,

            ha="center",

            va="bottom",

            rotation=0

        )


    fig.tight_layout()


    save_figure(

        fig,

        filename

    )


# ============================================================
# 10. MEAN FINAL TOUR LENGTH
# ============================================================

create_mean_barplot(

    column="best_fitness",

    ylabel="Mean final tour length",

    title=(
        "Mean Final Tour Length — "
        "TSP (25 Cities)"
    ),

    filename="10_mean_fitness.png",

    formatter=lambda x:
        f"{x:.3f}"

)


# ============================================================
# 11. MEAN RUNTIME
# ============================================================

create_mean_barplot(

    column="runtime_seconds",

    ylabel="Mean runtime (seconds)",

    title=(
        "Mean Runtime — "
        "TSP (25 Cities)"
    ),

    filename="11_mean_runtime.png",

    formatter=lambda x:
        f"{x:.3f}"

)


# ============================================================
# 12. MEAN FUNCTION EVALUATIONS
# ============================================================

create_mean_barplot(

    column="evaluations",

    ylabel="Mean function evaluations",

    title=(
        "Mean Function Evaluations — "
        "TSP (25 Cities)"
    ),

    filename="12_mean_evaluations.png",

    formatter=lambda x:
        f"{x:,.0f}"

)


# ============================================================
# 13. MEAN CPU TIME
# ============================================================

create_mean_barplot(

    column="cpu_time_seconds",

    ylabel="Mean CPU time (seconds)",

    title=(
        "Mean CPU Time — "
        "TSP (25 Cities)"
    ),

    filename="13_mean_cpu_time.png",

    formatter=lambda x:
        f"{x:.3f}"

)


# ============================================================
# 14. MEAN ADDITIONAL MEMORY
# ============================================================

create_mean_barplot(

    column="peak_memory_delta_mb",

    ylabel="Mean peak additional memory (MB)",

    title=(
        "Mean Peak Additional Memory — "
        "TSP (25 Cities)"
    ),

    filename="14_mean_memory.png",

    formatter=lambda x:
        f"{x:.3f}"

)


# ============================================================
# COMPLETE SUMMARY
# ============================================================

summary = (

    results

    .groupby("algorithm")

    .agg(

        mean_fitness=(
            "best_fitness",
            "mean"
        ),

        median_fitness=(
            "best_fitness",
            "median"
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

        std_peak_memory=(
            "peak_memory_mb",
            "std"
        ),

        mean_peak_memory_delta=(
            "peak_memory_delta_mb",
            "mean"
        ),

        std_peak_memory_delta=(
            "peak_memory_delta_mb",
            "std"
        ),

        mean_evaluations=(
            "evaluations",
            "mean"
        ),

        std_evaluations=(
            "evaluations",
            "std"
        )

    )

    .reindex(ALGORITHMS)

    .reset_index()

)


# ============================================================
# SAVE SUMMARY
# ============================================================

SUMMARY_FILE = os.path.join(

    RESULTS_DIR,

    "tsp_complete_summary.csv"

)


summary.to_csv(

    SUMMARY_FILE,

    index=False

)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print()
print("=" * 75)
print("TSP COMPLETE SUMMARY")
print("=" * 75)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("ALL TSP PLOTS GENERATED")
print("=" * 70)

print(
    f"Plots saved in:\n{PLOTS_DIR}"
)

print(
    f"\nSummary saved in:\n{SUMMARY_FILE}"
)

print(
    "\nThe figures will remain open "
    "for interactive inspection."
)


# ============================================================
# SHOW FIGURES
# ============================================================

plt.show()