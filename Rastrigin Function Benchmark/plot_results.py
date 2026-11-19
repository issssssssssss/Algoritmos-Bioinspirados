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
# ARCHIVOS
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
# CARGAR DATOS
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
# ORDEN DE LOS ALGORITMOS
# ============================================================

ALGORITHMS = [
    "GA",
    "cGA",
    "PSO",
    "GWO",
    "WOA",
    "ABC",
    "ACOR",
    "FA",
    "AFSA",
    "AIS",
    "BFOA",
    "IWO",
    "PFA",
    "IWD"
]


# ============================================================
# VALIDAR ALGORITMOS DISPONIBLES
# ============================================================

available_algorithms = results[
    "algorithm"
].unique()

missing_algorithms = [
    algorithm
    for algorithm in ALGORITHMS
    if algorithm not in available_algorithms
]

if missing_algorithms:

    print(
        "\nWARNING: The following algorithms "
        "were not found in benchmark_results.csv:"
    )

    print(
        missing_algorithms
    )

    ALGORITHMS = [
        algorithm
        for algorithm in ALGORITHMS
        if algorithm in available_algorithms
    ]


# ============================================================
# FUNCIÓN PARA GUARDAR FIGURAS
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


# ============================================================
# PREPARAR CONVERGENCIA CON EVALUACIONES IRREGULARES
# ============================================================

def convergence_statistics(
    data,
    points=100
):

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
        return pd.DataFrame(
            columns=["evaluation", "mean", "std"]
        )

    start = max(
        run["evaluation"].iloc[0]
        for run in runs
    )

    end = min(
        run["evaluation"].iloc[-1]
        for run in runs
    )

    if start >= end:
        return pd.DataFrame(
            columns=["evaluation", "mean", "std"]
        )

    evaluation_grid = np.linspace(
        start,
        end,
        min(points, max(2, int(end - start + 1)))
    )

    interpolated = []

    for run in runs:

        interpolated.append(
            np.interp(
                evaluation_grid,
                run["evaluation"].to_numpy(),
                run["fitness"].to_numpy()
            )
        )

    values = np.asarray(interpolated)

    return pd.DataFrame({
        "evaluation": evaluation_grid,
        "mean": values.mean(axis=0),
        "std": values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(len(evaluation_grid))
    })


# ============================================================
# FUNCIÓN PARA CREAR BOXPLOT
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
        ].dropna().values

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
# 1. FITNESS FINAL
# ============================================================

create_boxplot(
    column="best_fitness",
    ylabel="Final fitness",
    title="Final Fitness Distribution — Rastrigin 10D",
    filename="01_fitness_boxplot.png",
    log_scale=True
)


# ============================================================
# 2. RUNTIME
# ============================================================

create_boxplot(
    column="runtime_seconds",
    ylabel="Runtime (seconds)",
    title="Runtime Distribution — Rastrigin 10D",
    filename="02_runtime_boxplot.png"
)


# ============================================================
# 3. EVALUACIONES
# ============================================================

create_boxplot(
    column="evaluations",
    ylabel="Function evaluations",
    title="Function Evaluations — Rastrigin 10D",
    filename="03_evaluations_boxplot.png"
)


# ============================================================
# 4. CPU TIME
# ============================================================

create_boxplot(
    column="cpu_time_seconds",
    ylabel="CPU time (seconds)",
    title="CPU Time Distribution — Rastrigin 10D",
    filename="04_cpu_time_boxplot.png"
)


# ============================================================
# 5. CPU UTILIZATION
# ============================================================

create_boxplot(
    column="cpu_utilization_percent",
    ylabel="Average CPU utilization (%)",
    title="CPU Utilization — Rastrigin 10D",
    filename="05_cpu_utilization_boxplot.png"
)


# ============================================================
# 6. RAM MÁXIMA
# ============================================================

create_boxplot(
    column="peak_memory_mb",
    ylabel="Peak memory (MB)",
    title="Peak Memory Usage — Rastrigin 10D",
    filename="06_peak_memory_boxplot.png"
)


# ============================================================
# 7. INCREMENTO DE RAM
# ============================================================

create_boxplot(
    column="peak_memory_delta_mb",
    ylabel="Peak additional memory (MB)",
    title="Peak Additional Memory — Rastrigin 10D",
    filename="07_memory_delta_boxplot.png"
)


# ============================================================
# 8. OPTIMALITY GAP
# ============================================================

OPTIMAL_VALUE = 0.0


results["optimality_gap"] = np.abs(
    results["best_fitness"]
    - OPTIMAL_VALUE
)


fig, ax = plt.subplots(
    figsize=(10, 6)
)


data = [
    results.loc[
        results["algorithm"] == algorithm,
        "optimality_gap"
    ].dropna().values

    for algorithm in ALGORITHMS
]


# Evitar problemas con escala logarítmica si existe un cero
data_for_plot = [
    np.clip(
        values,
        np.finfo(float).tiny,
        None
    )

    for values in data
]


ax.boxplot(
    data_for_plot,
    labels=ALGORITHMS
)


ax.set_yscale(
    "log"
)


ax.set_xlabel(
    "Algorithm"
)

ax.set_ylabel(
    "Optimality gap"
)

ax.set_title(
    "Optimality Gap Distribution — Rastrigin 10D"
)

ax.grid(
    axis="y",
    which="both",
    alpha=0.25
)

fig.tight_layout()

save_figure(
    fig,
    "08_optimality_gap_boxplot.png"
)


# ============================================================
# 9. TASA DE ÉXITO
# ============================================================

success_summary = (
    results
    .groupby("algorithm")["success"]
    .mean()
    .reindex(ALGORITHMS)
    * 100
)


fig, ax = plt.subplots(
    figsize=(10, 6)
)


x = np.arange(
    len(ALGORITHMS)
)


bars = ax.bar(
    x,
    success_summary.values
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    ALGORITHMS
)


ax.set_ylim(
    0,
    105
)


ax.set_xlabel(
    "Algorithm"
)

ax.set_ylabel(
    "Success rate (%)"
)

ax.set_title(
    "Success Rate — Rastrigin 10D"
)

ax.grid(
    axis="y",
    alpha=0.25
)


for bar, value in zip(
    bars,
    success_summary.values
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,

        value + 2,

        f"{value:.1f}%",

        ha="center"
    )


fig.tight_layout()

save_figure(
    fig,
    "09_success_rate.png"
)


# ============================================================
# 10. CONVERGENCIA PROMEDIO
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

    grouped["fitness"] = (
        grouped["mean"]
        .clip(
            lower=np.finfo(float).tiny
        )
    )

    ax.plot(
        grouped["evaluation"],
        grouped["fitness"],
        label=algorithm
    )


ax.set_yscale(
    "log"
)

ax.set_xlabel(
    "Function evaluations"
)

ax.set_ylabel(
    "Mean best-so-far fitness"
)

ax.set_title(
    "Mean Convergence — Rastrigin 10D"
)

ax.legend()

ax.grid(
    True,
    which="both",
    alpha=0.25
)

fig.tight_layout()

save_figure(
    fig,
    "10_convergence_mean.png"
)


# ============================================================
# 11. CONVERGENCIA + DESVIACIÓN ESTÁNDAR
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

    grouped["mean"] = (
        grouped["mean"]
        .clip(
            lower=np.finfo(float).tiny
        )
    )


    lower = (
        grouped["mean"]
        - grouped["std"].fillna(0)
    )

    upper = (
        grouped["mean"]
        + grouped["std"].fillna(0)
    )


    lower = lower.clip(
        lower=np.finfo(float).tiny
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


ax.set_yscale(
    "log"
)

ax.set_xlabel(
    "Function evaluations"
)

ax.set_ylabel(
    "Fitness"
)

ax.set_title(
    "Mean Convergence ± Standard Deviation — Rastrigin 10D"
)

ax.legend()

ax.grid(
    True,
    which="both",
    alpha=0.25
)

fig.tight_layout()

save_figure(
    fig,
    "11_convergence_std.png"
)


# ============================================================
# FUNCIÓN PARA BARRAS DE PROMEDIO
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
        yerr=summary["std"].fillna(0).values,
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


    for bar, value in zip(
        bars,
        summary["mean"].values
    ):

        if formatter is not None:

            text = formatter(
                value
            )

        else:

            text = f"{value:.4g}"


        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            value,

            text,

            ha="center",
            va="bottom"
        )


    fig.tight_layout()

    save_figure(
        fig,
        filename
    )


# ============================================================
# 12. FITNESS MEDIO
# ============================================================

create_mean_barplot(
    column="best_fitness",
    ylabel="Mean final fitness",
    title="Mean Final Fitness — Rastrigin 10D",
    filename="12_mean_fitness.png",
    formatter=lambda x: f"{x:.3e}"
)


# ============================================================
# 13. RUNTIME MEDIO
# ============================================================

create_mean_barplot(
    column="runtime_seconds",
    ylabel="Mean runtime (seconds)",
    title="Mean Runtime — Rastrigin 10D",
    filename="13_mean_runtime.png",
    formatter=lambda x: f"{x:.3f}"
)


# ============================================================
# 14. EVALUACIONES MEDIAS
# ============================================================

create_mean_barplot(
    column="evaluations",
    ylabel="Mean function evaluations",
    title="Mean Function Evaluations — Rastrigin 10D",
    filename="14_mean_evaluations.png",
    formatter=lambda x: f"{x:,.0f}"
)


# ============================================================
# 15. CPU TIME MEDIO
# ============================================================

create_mean_barplot(
    column="cpu_time_seconds",
    ylabel="Mean CPU time (seconds)",
    title="Mean CPU Time — Rastrigin 10D",
    filename="15_mean_cpu_time.png",
    formatter=lambda x: f"{x:.3f}"
)


# ============================================================
# 16. RAM MEDIA
# ============================================================

create_mean_barplot(
    column="peak_memory_delta_mb",
    ylabel="Mean peak additional memory (MB)",
    title="Mean Peak Additional Memory — Rastrigin 10D",
    filename="16_mean_memory.png",
    formatter=lambda x: f"{x:.2f}"
)


# ============================================================
# TABLA RESUMEN
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

        mean_peak_memory=(
            "peak_memory_mb",
            "mean"
        ),

        mean_peak_memory_delta=(
            "peak_memory_delta_mb",
            "mean"
        ),

        mean_evaluations=(
            "evaluations",
            "mean"
        ),

        std_evaluations=(
            "evaluations",
            "std"
        ),

        success_rate=(
            "success",
            "mean"
        )
    )

    .reindex(ALGORITHMS)

    .reset_index()
)


summary["success_rate_percent"] = (
    summary["success_rate"]
    * 100
)


summary_path = os.path.join(
    PLOTS_DIR,
    "rastrigin_complete_summary.csv"
)


summary.to_csv(
    summary_path,
    index=False
)


print(
    f"\nSummary saved: {summary_path}"
)


# ============================================================
# MOSTRAR TODAS LAS FIGURAS
# ============================================================

print("\n")
print("=" * 70)
print("ALL PLOTS GENERATED")
print("=" * 70)

print(
    "\nClose the graph window when you are finished "
    "analyzing the figures."
)


# Importante:
# NO usamos plt.close() porque queremos que las figuras
# permanezcan disponibles para interacción.

plt.show()