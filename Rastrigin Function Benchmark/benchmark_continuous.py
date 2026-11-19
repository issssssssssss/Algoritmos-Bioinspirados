import os
import sys
import time
import threading

import numpy as np
import pandas as pd
import psutil


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

TASK_DIR = os.path.dirname(
    CURRENT_DIR
)

sys.path.append(TASK_DIR)


# ============================================================
# IMPORTS
# ============================================================

import Rastrigin_GA as ga_module
import Rastrigin_cGA as cga_module
import Rastrigin_PSO as pso_module

import Rastrigin_GWO as gwo_module
import Rastrigin_WOA as woa_module
import Rastrigin_ABC as abc_module
import Rastrigin_ACOR as acor_module
import Rastrigin_FA as fa_module
import Rastrigin_AFSA as afsa_module
import Rastrigin_AIS as ais_module
import Rastrigin_BFOA as bfoa_module
import Rastrigin_IWO as iwo_module
import Rastrigin_PFA as pfa_module
import Rastrigin_IWD as iwd_module


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROBLEM = "Rastrigin"

N_DIM = 10

BOUNDS = (-5.12, 5.12)

EVAL_BUDGET = 20_000

N_RUNS = 30

SEEDS = range(N_RUNS)


# ============================================================
# PARÁMETROS
# ============================================================

GA_POP_SIZE = 100

CGA_POP_SIZE = 100
CGA_BITS_PER_VAR = 16

PSO_PARTICLES = 50

GWO_POP_SIZE = 30

WOA_POP_SIZE = 30

ABC_POP_SIZE = 30
ABC_LIMIT = 50

ACOR_ARCHIVE_SIZE = 30
ACOR_N_ANTS = 30

FA_POP_SIZE = 30

AFSA_POP_SIZE = 30

AIS_POP_SIZE = 30

BFOA_POP_SIZE = 30

IWO_POP_SIZE = 30

PFA_POP_SIZE = 30

IWD_POP_SIZE = 30


# ============================================================
# CRITERIO DE ÉXITO
# ============================================================

SUCCESS_THRESHOLD = 1e-8


# ============================================================
# RECURSOS
# ============================================================

PROCESS = psutil.Process(
    os.getpid()
)


# ============================================================
# MONITOR DE MEMORIA
# ============================================================

class MemoryMonitor:

    def __init__(
        self,
        process,
        interval=0.01
    ):

        self.process = process
        self.interval = interval

        self.running = False
        self.thread = None

        self.baseline_mb = 0.0
        self.peak_mb = 0.0

    def start(self):

        self.baseline_mb = (
            self.process.memory_info().rss
            / (1024 ** 2)
        )

        self.peak_mb = self.baseline_mb

        self.running = True

        self.thread = threading.Thread(
            target=self._monitor,
            daemon=True
        )

        self.thread.start()


    def _monitor(self):

        while self.running:

            try:

                current_mb = (
                    self.process.memory_info().rss
                    / (1024 ** 2)
                )

                if current_mb > self.peak_mb:
                    self.peak_mb = current_mb

            except psutil.Error:
                break

            time.sleep(
                self.interval
            )


    def stop(self):

        self.running = False

        if self.thread is not None:
            self.thread.join()

        return (
            self.baseline_mb,
            self.peak_mb
        )


# ============================================================
# EJECUTAR CON MÉTRICAS
# ============================================================

def run_with_resources(
    algorithm_function
):

    memory_monitor = MemoryMonitor(
        PROCESS
    )

    wall_start = time.perf_counter()

    cpu_start = time.process_time()

    memory_monitor.start()

    result = algorithm_function()

    baseline_mb, peak_mb = (
        memory_monitor.stop()
    )

    wall_time = (
        time.perf_counter()
        - wall_start
    )

    cpu_time = (
        time.process_time()
        - cpu_start
    )

    peak_delta_mb = max(
        0.0,
        peak_mb - baseline_mb
    )

    cpu_count = psutil.cpu_count(
        logical=True
    )

    if (
        wall_time > 0
        and cpu_count
    ):

        cpu_utilization = (
            cpu_time
            / wall_time
            / cpu_count
            * 100
        )

    else:

        cpu_utilization = 0.0


    return (
        result,
        wall_time,
        cpu_time,
        baseline_mb,
        peak_mb,
        peak_delta_mb,
        cpu_utilization
    )


# ============================================================
# ESTRUCTURAS DE RESULTADOS
# ============================================================

final_results = []

convergence_results = []


# ============================================================
# FUNCIÓN AUXILIAR
# ============================================================

def save_result(
    algorithm,
    seed,
    best_fitness,
    runtime,
    cpu_time,
    cpu_utilization,
    baseline_memory,
    peak_memory,
    peak_delta_memory,
    evaluations
):

    success = (
        best_fitness
        <= SUCCESS_THRESHOLD
    )

    final_results.append({

        "algorithm": algorithm,

        "problem": PROBLEM,

        "dimension": N_DIM,

        "seed": seed,

        "best_fitness":
            best_fitness,

        "runtime_seconds":
            runtime,

        "cpu_time_seconds":
            cpu_time,

        "cpu_utilization_percent":
            cpu_utilization,

        "memory_baseline_mb":
            baseline_memory,

        "peak_memory_mb":
            peak_memory,

        "peak_memory_delta_mb":
            peak_delta_memory,

        "evaluations":
            evaluations,

        "success":
            int(success)
    })


def run_additional_algorithm(
    module,
    algorithm,
    run_kwargs,
    seed
):

    run_function = getattr(
        module,
        f"run_{algorithm.lower()}"
    )

    run_kwargs = {
        **run_kwargs,
        "seed": seed,
        "verbose": False
    }

    (
        result,
        runtime,
        cpu_time,
        baseline_memory,
        peak_memory,
        peak_delta_memory,
        cpu_utilization
    ) = run_with_resources(
        lambda: run_function(**run_kwargs)
    )

    (
        best_x,
        best_f,
        history,
        mean_history,
        position_history
    ) = result

    evaluations = list(
        run_function.evaluation_history
    )

    save_result(
        algorithm,
        seed,
        best_f,
        runtime,
        cpu_time,
        cpu_utilization,
        baseline_memory,
        peak_memory,
        peak_delta_memory,
        evaluations[-1]
    )

    convergence_results.append(
        pd.DataFrame({
            "algorithm": algorithm,
            "problem": PROBLEM,
            "dimension": N_DIM,
            "seed": seed,
            "evaluation": evaluations,
            "fitness": history
        })
    )


# ============================================================
# EJECUCIONES
# ============================================================

for seed in SEEDS:

    print()
    print("=" * 70)

    print(
        f"RUN {seed + 1}/{N_RUNS} | "
        f"SEED = {seed}"
    )

    print("=" * 70)


    # ========================================================
    # GA
    # ========================================================

    print("\nRunning GA...")

    GA_GENERATIONS = (
        max(
            1,
            (EVAL_BUDGET - GA_POP_SIZE)
            // GA_POP_SIZE
        )
    )


    def run_ga_experiment():

        return ga_module.run_ga(
            n_dim=N_DIM,
            pop_size=GA_POP_SIZE,
            n_generations=GA_GENERATIONS,
            bounds=BOUNDS,
            seed=seed,
            verbose=False
        )


    (
        ga_result,
        ga_runtime,
        ga_cpu_time,
        ga_baseline_memory,
        ga_peak_memory,
        ga_peak_delta_memory,
        ga_cpu_utilization
    ) = run_with_resources(
        run_ga_experiment
    )


    (
        ga_best_x,
        ga_best_f,
        ga_history,
        ga_mean_history,
        ga_position_history
    ) = ga_result


    ga_total_evaluations = ga_module.run_ga.evaluation_history[-1]


    save_result(

        "GA",

        seed,

        ga_best_f,

        ga_runtime,

        ga_cpu_time,

        ga_cpu_utilization,

        ga_baseline_memory,

        ga_peak_memory,

        ga_peak_delta_memory,

        ga_total_evaluations
    )


    # ========================================================
    # CONVERGENCIA GA
    # ========================================================

    convergence_results.append(

        pd.DataFrame({

            "algorithm": "GA",

            "problem": PROBLEM,

            "dimension": N_DIM,

            "seed": seed,

            "evaluation":
                ga_module.run_ga.evaluation_history,

            "fitness":
                ga_history
        })
    )


    # ========================================================
    # ALGORITMOS ADICIONALES
    # ========================================================

    additional_algorithms = [
        (
            fa_module,
            "FA",
            {
                "n_dim": N_DIM,
                "pop_size": FA_POP_SIZE,
                "n_iterations": max(1, (EVAL_BUDGET - FA_POP_SIZE) // FA_POP_SIZE),
                "bounds": BOUNDS
            }
        ),
        (
            afsa_module,
            "AFSA",
            {
                "n_dim": N_DIM,
                "pop_size": AFSA_POP_SIZE,
                "n_iterations": max(1, (EVAL_BUDGET - AFSA_POP_SIZE) // AFSA_POP_SIZE),
                "bounds": BOUNDS
            }
        ),
        (
            ais_module,
            "AIS",
            {
                "n_dim": N_DIM,
                "pop_size": AIS_POP_SIZE,
                "n_iterations": max(1, (EVAL_BUDGET - AIS_POP_SIZE) // AIS_POP_SIZE),
                "bounds": BOUNDS
            }
        ),
        (
            bfoa_module,
            "BFOA",
            {
                "n_dim": N_DIM,
                "pop_size": BFOA_POP_SIZE,
                "n_iterations": max(1, (EVAL_BUDGET - BFOA_POP_SIZE) // BFOA_POP_SIZE),
                "bounds": BOUNDS
            }
        ),
        (
            iwo_module,
            "IWO",
            {
                "n_dim": N_DIM,
                "pop_size": IWO_POP_SIZE,
                "n_iterations": max(1, (EVAL_BUDGET - IWO_POP_SIZE) // IWO_POP_SIZE),
                "bounds": BOUNDS
            }
        ),
        (
            pfa_module,
            "PFA",
            {
                "n_dim": N_DIM,
                "pop_size": PFA_POP_SIZE,
                "n_iterations": max(1, (EVAL_BUDGET - PFA_POP_SIZE) // PFA_POP_SIZE),
                "bounds": BOUNDS
            }
        ),
        (
            iwd_module,
            "IWD",
            {
                "n_dim": N_DIM,
                "pop_size": IWD_POP_SIZE,
                "n_iterations": max(1, (EVAL_BUDGET - IWD_POP_SIZE) // IWD_POP_SIZE),
                "bounds": BOUNDS
            }
        )
    ]

    for module, algorithm, run_kwargs in additional_algorithms:
        print(f"Running {algorithm}...")
        run_additional_algorithm(
            module,
            algorithm,
            run_kwargs,
            seed
        )


    # ========================================================
    # cGA
    # ========================================================

    print("Running cGA...")

    CGA_GENERATIONS = (
        EVAL_BUDGET // 2
    )


    def run_cga_experiment():

        return cga_module.run_cga(

            n_dim=N_DIM,

            bits_per_var=CGA_BITS_PER_VAR,

            bounds=BOUNDS,

            pop_size=CGA_POP_SIZE,

            n_generations=CGA_GENERATIONS,

            seed=seed,

            verbose=False
        )


    (
        cga_result,
        cga_runtime,
        cga_cpu_time,
        cga_baseline_memory,
        cga_peak_memory,
        cga_peak_delta_memory,
        cga_cpu_utilization
    ) = run_with_resources(
        run_cga_experiment
    )


    (
        cga_best_x,
        cga_best_f,
        cga_history,
        cga_position_history
    ) = cga_result


    cga_total_evaluations = cga_module.run_cga.evaluation_history[-1]


    save_result(

        "cGA",

        seed,

        cga_best_f,

        cga_runtime,

        cga_cpu_time,

        cga_cpu_utilization,

        cga_baseline_memory,

        cga_peak_memory,

        cga_peak_delta_memory,

        cga_total_evaluations
    )


    convergence_results.append(

        pd.DataFrame({

            "algorithm": "cGA",

            "problem": PROBLEM,

            "dimension": N_DIM,

            "seed": seed,

            "evaluation":
                cga_module.run_cga.evaluation_history,

            "fitness":
                cga_history
        })
    )


    # ========================================================
    # PSO
    # ========================================================

    print("Running PSO...")

    PSO_ITERATIONS = (
        max(
            1,
            (EVAL_BUDGET - PSO_PARTICLES)
            // PSO_PARTICLES
        )
    )


    def run_pso_experiment():

        return pso_module.run_pso(

            n_dim=N_DIM,

            n_particles=PSO_PARTICLES,

            n_iterations=PSO_ITERATIONS,

            bounds=BOUNDS,

            seed=seed,

            verbose=False
        )


    (
        pso_result,
        pso_runtime,
        pso_cpu_time,
        pso_baseline_memory,
        pso_peak_memory,
        pso_peak_delta_memory,
        pso_cpu_utilization
    ) = run_with_resources(
        run_pso_experiment
    )


    (
        pso_best_x,
        pso_best_f,
        pso_history,
        pso_position_history
    ) = pso_result


    pso_total_evaluations = pso_module.run_pso.evaluation_history[-1]


    save_result(

        "PSO",

        seed,

        pso_best_f,

        pso_runtime,

        pso_cpu_time,

        pso_cpu_utilization,

        pso_baseline_memory,

        pso_peak_memory,

        pso_peak_delta_memory,

        pso_total_evaluations
    )


    convergence_results.append(

        pd.DataFrame({

            "algorithm": "PSO",

            "problem": PROBLEM,

            "dimension": N_DIM,

            "seed": seed,

            "evaluation":
                pso_module.run_pso.evaluation_history,

            "fitness":
                pso_history
        })
    )


    # ========================================================
    # GWO
    # ========================================================

    print("Running GWO...")

    GWO_ITERATIONS = (
        max(
            1,
            (EVAL_BUDGET - GWO_POP_SIZE)
            // GWO_POP_SIZE
        )
    )


    def run_gwo_experiment():

        return gwo_module.run_gwo(

            n_dim=N_DIM,

            pop_size=GWO_POP_SIZE,

            n_iterations=GWO_ITERATIONS,

            bounds=BOUNDS,

            seed=seed,

            verbose=False
        )


    (
        gwo_result,
        gwo_runtime,
        gwo_cpu_time,
        gwo_baseline_memory,
        gwo_peak_memory,
        gwo_peak_delta_memory,
        gwo_cpu_utilization
    ) = run_with_resources(
        run_gwo_experiment
    )


    (
        gwo_best_x,
        gwo_best_f,
        gwo_history,
        gwo_mean_history,
        gwo_position_history
    ) = gwo_result


    gwo_total_evaluations = gwo_module.run_gwo.evaluation_history[-1]


    save_result(

        "GWO",

        seed,

        gwo_best_f,

        gwo_runtime,

        gwo_cpu_time,

        gwo_cpu_utilization,

        gwo_baseline_memory,

        gwo_peak_memory,

        gwo_peak_delta_memory,

        gwo_total_evaluations
    )


    convergence_results.append(

        pd.DataFrame({

            "algorithm": "GWO",

            "problem": PROBLEM,

            "dimension": N_DIM,

            "seed": seed,

            "evaluation":
                gwo_module.run_gwo.evaluation_history,

            "fitness":
                gwo_history
        })
    )


    # ========================================================
    # WOA
    # ========================================================

    print("Running WOA...")

    WOA_ITERATIONS = (
        max(
            1,
            (EVAL_BUDGET - WOA_POP_SIZE)
            // WOA_POP_SIZE
        )
    )


    def run_woa_experiment():

        return woa_module.run_woa(

            n_dim=N_DIM,

            pop_size=WOA_POP_SIZE,

            n_iterations=WOA_ITERATIONS,

            bounds=BOUNDS,

            seed=seed,

            verbose=False
        )


    (
        woa_result,
        woa_runtime,
        woa_cpu_time,
        woa_baseline_memory,
        woa_peak_memory,
        woa_peak_delta_memory,
        woa_cpu_utilization
    ) = run_with_resources(
        run_woa_experiment
    )


    (
        woa_best_x,
        woa_best_f,
        woa_history,
        woa_mean_history,
        woa_position_history
    ) = woa_result


    woa_total_evaluations = woa_module.run_woa.evaluation_history[-1]


    save_result(

        "WOA",

        seed,

        woa_best_f,

        woa_runtime,

        woa_cpu_time,

        woa_cpu_utilization,

        woa_baseline_memory,

        woa_peak_memory,

        woa_peak_delta_memory,

        woa_total_evaluations
    )


    convergence_results.append(

        pd.DataFrame({

            "algorithm": "WOA",

            "problem": PROBLEM,

            "dimension": N_DIM,

            "seed": seed,

            "evaluation":
                woa_module.run_woa.evaluation_history,

            "fitness":
                woa_history
        })
    )


    # ========================================================
    # ABC
    # ========================================================

    print("Running ABC...")

    # ABC:
    #
    # Inicialización:
    # 30 evaluaciones
    #
    # Cada iteración:
    # 30 employed bees
    # +
    # 30 onlooker bees
    #
    # = aproximadamente 60 evaluaciones
    #
    # Las scout bees pueden agregar evaluaciones extras,
    # por eso el contador real es el que se almacena.

    ABC_ITERATIONS = (
        max(
            1,
            (EVAL_BUDGET - ABC_POP_SIZE)
            // (2 * ABC_POP_SIZE)
        )
    )


    def run_abc_experiment():

        return abc_module.run_abc(

            n_dim=N_DIM,

            pop_size=ABC_POP_SIZE,

            n_iterations=ABC_ITERATIONS,

            bounds=BOUNDS,

            limit=ABC_LIMIT,

            seed=seed,

            verbose=False
        )


    (
        abc_result,
        abc_runtime,
        abc_cpu_time,
        abc_baseline_memory,
        abc_peak_memory,
        abc_peak_delta_memory,
        abc_cpu_utilization
    ) = run_with_resources(
        run_abc_experiment
    )


    (
        abc_best_x,
        abc_best_f,
        abc_history,
        abc_mean_history,
        abc_position_history
    ) = abc_result


    abc_total_evaluations = abc_module.run_abc.evaluation_history[-1]


    save_result(

        "ABC",

        seed,

        abc_best_f,

        abc_runtime,

        abc_cpu_time,

        abc_cpu_utilization,

        abc_baseline_memory,

        abc_peak_memory,

        abc_peak_delta_memory,

        abc_total_evaluations
    )


    convergence_results.append(

        pd.DataFrame({

            "algorithm": "ABC",

            "problem": PROBLEM,

            "dimension": N_DIM,

            "seed": seed,

            "evaluation":
                abc_module.run_abc.evaluation_history,

            "fitness":
                abc_history
        })
    )


    # ========================================================
    # ACOR
    # ========================================================

    print("Running ACOR...")

    ACOR_ITERATIONS = (
        max(
            1,
            (EVAL_BUDGET - ACOR_ARCHIVE_SIZE)
            // ACOR_N_ANTS
        )
    )


    def run_acor_experiment():

        return acor_module.run_acor(

            n_dim=N_DIM,

            archive_size=ACOR_ARCHIVE_SIZE,

            n_ants=ACOR_N_ANTS,

            n_iterations=ACOR_ITERATIONS,

            bounds=BOUNDS,

            seed=seed,

            verbose=False
        )


    (
        acor_result,
        acor_runtime,
        acor_cpu_time,
        acor_baseline_memory,
        acor_peak_memory,
        acor_peak_delta_memory,
        acor_cpu_utilization
    ) = run_with_resources(
        run_acor_experiment
    )


    (
        acor_best_x,
        acor_best_f,
        acor_history,
        acor_mean_history,
        acor_position_history
    ) = acor_result


    acor_total_evaluations = acor_module.run_acor.evaluation_history[-1]


    save_result(

        "ACOR",

        seed,

        acor_best_f,

        acor_runtime,

        acor_cpu_time,

        acor_cpu_utilization,

        acor_baseline_memory,

        acor_peak_memory,

        acor_peak_delta_memory,

        acor_total_evaluations
    )


    convergence_results.append(

        pd.DataFrame({

            "algorithm": "ACOR",

            "problem": PROBLEM,

            "dimension": N_DIM,

            "seed": seed,

            "evaluation":
                acor_module.run_acor.evaluation_history,

            "fitness":
                acor_history
        })
    )


    # ========================================================
    # RESUMEN DE LA EJECUCIÓN
    # ========================================================

    print()

    for result in final_results[-14:]:

        print(

            f"{result['algorithm']:<5} | "

            f"fitness={result['best_fitness']:.6e} | "

            f"time={result['runtime_seconds']:.4f}s | "

            f"CPU={result['cpu_time_seconds']:.4f}s | "

            f"RAM+={result['peak_memory_delta_mb']:.2f}MB | "

            f"evals={result['evaluations']}"
        )


# ============================================================
# DATAFRAMES
# ============================================================

final_df = pd.DataFrame(
    final_results
)


convergence_df = pd.concat(
    convergence_results,
    ignore_index=True
)


# ============================================================
# RESULTADOS
# ============================================================

RESULTS_DIR = os.path.join(
    CURRENT_DIR,
    "Results"
)


os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


final_path = os.path.join(
    RESULTS_DIR,
    "benchmark_results.csv"
)


convergence_path = os.path.join(
    RESULTS_DIR,
    "convergence_results.csv"
)


final_df.to_csv(
    final_path,
    index=False
)


convergence_df.to_csv(
    convergence_path,
    index=False
)


# ============================================================
# RESUMEN ESTADÍSTICO
# ============================================================

summary_df = (

    final_df

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
        ),

        success_rate=(
            "success",
            "mean"
        )
    )

    .reset_index()
)


summary_df[
    "success_rate_percent"
] = (

    summary_df[
        "success_rate"
    ]

    * 100
)


summary_path = os.path.join(
    RESULTS_DIR,
    "benchmark_summary.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# RESULTADO FINAL
# ============================================================

print()
print("=" * 70)

print(
    "BENCHMARK FINISHED"
)

print("=" * 70)


print("\nIndividual results:")

print(
    final_df.to_string(
        index=False
    )
)


print("\nStatistical summary:")

print(
    summary_df.to_string(
        index=False
    )
)


print("\nFiles saved:")

print(final_path)

print(convergence_path)

print(summary_path)