import os
import time
import platform
import threading

import numpy as np
import pandas as pd

try:
    import psutil
except ImportError:
    raise ImportError(
        "Este benchmark requiere psutil. "
        "Instálalo con: pip install psutil"
    )


# ============================================================
# IMPORTS
# ============================================================

from TSP_Problem import (
    generate_cities,
    distance_matrix,
)

from TSP_GA import run_ga
from TSP_cGA import run_cga
from TSP_PSO import run_pso

from TSP_ABC import run_abc
from TSP_ACO import run_aco
from TSP_GWO import run_gwo
from TSP_WOA import run_woa
from TSP_FA import run_fa
from TSP_AFSA import run_afsa
from TSP_AIS import run_ais
from TSP_BFOA import run_bfoa
from TSP_IWO import run_iwo
from TSP_PFA import run_pfa
from TSP_IWD import run_iwd


# ============================================================
# CONFIGURACIÓN
# ============================================================

N_CITIES = 25

CITY_SEED = 42

N_RUNS = 30

EVAL_BUDGET = 60_000


# ============================================================
# PARÁMETROS DE LOS ALGORITMOS
# ============================================================

BITS_PER_KEY = 16


# ----------------------------
# GA
# ----------------------------

GA_POP_SIZE = 150


# ----------------------------
# cGA
# ----------------------------

CGA_POP_SIZE = 100


# ----------------------------
# PSO
# ----------------------------

PSO_PARTICLES = 80


# ----------------------------
# ABC
# ----------------------------

ABC_POP_SIZE = 30
ABC_LIMIT = 50


# ----------------------------
# ACO
# ----------------------------

ACO_ANTS = 30

ACO_ALPHA = 1.0
ACO_BETA = 3.0
ACO_RHO = 0.3
ACO_Q = 100.0


# ----------------------------
# GWO
# ----------------------------

GWO_POP_SIZE = 30


# ----------------------------
# WOA
# ----------------------------

WOA_POP_SIZE = 30
WOA_B = 1.0

ADDITIONAL_POP_SIZE = 30


# ============================================================
# ALGORITMOS
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
    "IWD",
]


# ============================================================
# DIRECTORIOS
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RESULTS_DIR = os.path.join(
    CURRENT_DIR,
    "Results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "benchmark_results.csv"
)

CONVERGENCE_FILE = os.path.join(
    RESULTS_DIR,
    "convergence_results.csv"
)

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "benchmark_summary.csv"
)


# ============================================================
# INSTANCIA DEL TSP
# ============================================================

# Las mismas ciudades se utilizan para todas las ejecuciones
# y para todos los algoritmos.

cities = generate_cities(
    n_cities=N_CITIES,
    seed=CITY_SEED
)

dmat = distance_matrix(cities)


# ============================================================
# CONFIGURACIÓN DEL PRESUPUESTO DE EVALUACIONES
# ============================================================

# ------------------------------------------------------------
# GA
#
# Evaluación inicial:
#     GA_POP_SIZE
#
# Cada generación:
#     GA_POP_SIZE
# ------------------------------------------------------------

GA_GENERATIONS = (
    max(1, (EVAL_BUDGET - GA_POP_SIZE) // GA_POP_SIZE)
)


# ------------------------------------------------------------
# cGA
#
# Cada generación compara exactamente dos individuos.
# ------------------------------------------------------------

CGA_GENERATIONS = (
    max(1, EVAL_BUDGET // 2)
)


# ------------------------------------------------------------
# PSO
#
# Evaluación inicial:
#     PSO_PARTICLES
#
# Cada iteración:
#     PSO_PARTICLES
# ------------------------------------------------------------

PSO_ITERATIONS = (
    max(1, (EVAL_BUDGET - PSO_PARTICLES) // PSO_PARTICLES)
)


# ------------------------------------------------------------
# ABC
#
# Evaluación inicial:
#     ABC_POP_SIZE
#
# Cada iteración:
#
# Employed bees:
#     ABC_POP_SIZE evaluaciones
#
# Onlooker bees:
#     ABC_POP_SIZE evaluaciones
#
# Scouts:
#     variable
#
# Para mantener un presupuesto comparable usamos como
# referencia el costo base:
#
#     2 * ABC_POP_SIZE por iteración
#
# Las evaluaciones extra de scouts se contabilizan
# aproximadamente por separado en el algoritmo real, pero
# aquí usamos el presupuesto base para determinar iteraciones.
# ------------------------------------------------------------

ABC_ITERATIONS = (
    max(1, (EVAL_BUDGET - ABC_POP_SIZE) // (2 * ABC_POP_SIZE))
)


# ------------------------------------------------------------
# ACO
#
# Cada hormiga construye y evalúa un tour completo.
#
# Cada iteración:
#     ACO_ANTS evaluaciones
# ------------------------------------------------------------

ACO_ITERATIONS = (
    max(1, EVAL_BUDGET // ACO_ANTS)
)


# ------------------------------------------------------------
# GWO
#
# Evaluación inicial:
#     GWO_POP_SIZE
#
# Cada iteración:
#     GWO_POP_SIZE
# ------------------------------------------------------------

GWO_ITERATIONS = (
    max(1, (EVAL_BUDGET - GWO_POP_SIZE) // GWO_POP_SIZE)
)


# ------------------------------------------------------------
# WOA
#
# Evaluación inicial:
#     WOA_POP_SIZE
#
# Cada iteración:
#     WOA_POP_SIZE
# ------------------------------------------------------------

WOA_ITERATIONS = (
    max(1, (EVAL_BUDGET - WOA_POP_SIZE) // WOA_POP_SIZE)
)


# ============================================================
# INFORMACIÓN DEL BENCHMARK
# ============================================================

print("=" * 75)
print("TSP BENCHMARK")
print("=" * 75)

print(f"Cities:              {N_CITIES}")
print(f"City seed:           {CITY_SEED}")
print(f"Runs per algorithm:  {N_RUNS}")
print(f"Evaluation budget:   {EVAL_BUDGET}")

print()

print(f"GA generations:      {GA_GENERATIONS}")
print(f"cGA generations:     {CGA_GENERATIONS}")
print(f"PSO iterations:      {PSO_ITERATIONS}")
print(f"ABC iterations:      {ABC_ITERATIONS}")
print(f"ACO iterations:      {ACO_ITERATIONS}")
print(f"GWO iterations:      {GWO_ITERATIONS}")
print(f"WOA iterations:      {WOA_ITERATIONS}")

print("=" * 75)


# ============================================================
# RESULTADOS
# ============================================================

benchmark_results = []

convergence_results = []


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
# EJECUTAR Y MEDIR RECURSOS
# ============================================================

def run_with_resources(
    algorithm_function
):

    process = psutil.Process(
        os.getpid()
    )

    memory_monitor = MemoryMonitor(
        process,
        interval=0.01
    )


    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    cpu_start = time.process_time()


    # --------------------------------------------------------
    # Tiempo real
    # --------------------------------------------------------

    wall_start = time.perf_counter()


    # --------------------------------------------------------
    # Memoria
    # --------------------------------------------------------

    memory_monitor.start()


    # --------------------------------------------------------
    # Ejecutar algoritmo
    # --------------------------------------------------------

    result = algorithm_function()


    # --------------------------------------------------------
    # Detener memoria
    # --------------------------------------------------------

    baseline_mb, peak_mb = (
        memory_monitor.stop()
    )


    # --------------------------------------------------------
    # Medir CPU
    # --------------------------------------------------------

    cpu_time = (
        time.process_time()
        - cpu_start
    )


    # --------------------------------------------------------
    # Medir runtime
    # --------------------------------------------------------

    runtime = (
        time.perf_counter()
        - wall_start
    )

    # --------------------------------------------------------
    # Memoria adicional
    # --------------------------------------------------------

    peak_delta_mb = max(
        0.0,
        peak_mb - baseline_mb
    )


    # --------------------------------------------------------
    # Utilización CPU aproximada
    # --------------------------------------------------------

    cpu_count = psutil.cpu_count(
        logical=True
    )

    if runtime > 0 and cpu_count:
        cpu_utilization = (
            cpu_time
            / runtime
            / cpu_count
            * 100
        )

    else:
        cpu_utilization = 0.0

    return (
        result,
        runtime,
        cpu_time,
        baseline_mb,
        peak_mb,
        peak_delta_mb,
        cpu_utilization
    )


# ============================================================
# CREAR DATOS DE CONVERGENCIA
# ============================================================

def add_convergence_data(
    algorithm,
    history,
    evaluations_per_iteration,
    initial_evaluations,
    seed,
    evaluation_history=None
):

    history = np.asarray(
        history
    )

    actual_runs = {
        "GA": run_ga,
        "cGA": run_cga,
        "PSO": run_pso,
        "ABC": run_abc,
        "ACO": run_aco,
        "GWO": run_gwo,
        "WOA": run_woa,
    }

    if algorithm in actual_runs:
        evaluations = actual_runs[algorithm].evaluation_history
    elif evaluation_history is not None:
        evaluations = evaluation_history
    else:
        evaluations = (
            initial_evaluations
            + np.arange(1, len(history) + 1)
            * evaluations_per_iteration
        )



    for evaluation, fitness in zip(
        evaluations,
        history
    ):

        convergence_results.append({

            "algorithm":
                algorithm,

            "problem":
                "TSP",

            "dimension":
                N_CITIES,

            "seed":
                seed,

            "evaluation":
                evaluation,

            "fitness":
                fitness
        })


# ============================================================
# EJECUTAR BENCHMARK
# ============================================================

for seed in range(N_RUNS):

    print()

    print("=" * 75)

    print(
        f"RUN {seed + 1}/{N_RUNS} | SEED = {seed}"
    )

    print("=" * 75)


    # ========================================================
    # GA
    # ========================================================

    print("\nRunning GA...")


    def run_ga_experiment():

        return run_ga(

            dmat,

            N_CITIES,

            pop_size=GA_POP_SIZE,

            n_generations=GA_GENERATIONS,

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
        ga_best_tour,
        ga_best_fitness,
        ga_history
    ) = ga_result


    ga_initial_evaluations = (
        GA_POP_SIZE
    )

    ga_evaluations_per_iteration = (
        GA_POP_SIZE
    )


    ga_total_evaluations = run_ga.evaluation_history[-1]


    benchmark_results.append({

        "algorithm": "GA",
        "problem": "TSP",
        "dimension": N_CITIES,
        "seed": seed,

        "best_fitness":
            ga_best_fitness,

        "runtime_seconds":
            ga_runtime,

        "cpu_time_seconds":
            ga_cpu_time,

        "cpu_utilization_percent":
            ga_cpu_utilization,

        "memory_baseline_mb":
            ga_baseline_memory,

        "peak_memory_mb":
            ga_peak_memory,

        "peak_memory_delta_mb":
            ga_peak_delta_memory,

        "evaluations":
            ga_total_evaluations,

        "system":
            platform.system(),

        "python_version":
            platform.python_version(),

        "cpu_count":
            psutil.cpu_count(logical=True)
    })


    add_convergence_data(

        algorithm="GA",

        history=ga_history,

        evaluations_per_iteration=
            ga_evaluations_per_iteration,

        initial_evaluations=
            ga_initial_evaluations,

        seed=seed
    )


    # ========================================================
    # cGA
    # ========================================================

    print("Running cGA...")


    def run_cga_experiment():

        return run_cga(

            dmat,

            N_CITIES,

            bits_per_key=BITS_PER_KEY,

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
        cga_best_tour,
        cga_best_fitness,
        cga_history
    ) = cga_result


    cga_initial_evaluations = 0

    cga_evaluations_per_iteration = 2


    cga_total_evaluations = run_cga.evaluation_history[-1]


    benchmark_results.append({

        "algorithm": "cGA",
        "problem": "TSP",
        "dimension": N_CITIES,
        "seed": seed,

        "best_fitness":
            cga_best_fitness,

        "runtime_seconds":
            cga_runtime,

        "cpu_time_seconds":
            cga_cpu_time,

        "cpu_utilization_percent":
            cga_cpu_utilization,

        "memory_baseline_mb":
            cga_baseline_memory,

        "peak_memory_mb":
            cga_peak_memory,

        "peak_memory_delta_mb":
            cga_peak_delta_memory,

        "evaluations":
            cga_total_evaluations,

        "system":
            platform.system(),

        "python_version":
            platform.python_version(),

        "cpu_count":
            psutil.cpu_count(logical=True)
    })


    add_convergence_data(

        algorithm="cGA",

        history=cga_history,

        evaluations_per_iteration=
            cga_evaluations_per_iteration,

        initial_evaluations=
            cga_initial_evaluations,

        seed=seed
    )


    # ========================================================
    # PSO
    # ========================================================

    print("Running PSO...")


    def run_pso_experiment():

        return run_pso(

            dmat,

            N_CITIES,

            n_particles=PSO_PARTICLES,

            n_iterations=PSO_ITERATIONS,

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
        pso_best_tour,
        pso_best_fitness,
        pso_history
    ) = pso_result


    pso_initial_evaluations = (
        PSO_PARTICLES
    )

    pso_evaluations_per_iteration = (
        PSO_PARTICLES
    )


    pso_total_evaluations = run_pso.evaluation_history[-1]


    benchmark_results.append({

        "algorithm": "PSO",
        "problem": "TSP",
        "dimension": N_CITIES,
        "seed": seed,

        "best_fitness":
            pso_best_fitness,

        "runtime_seconds":
            pso_runtime,

        "cpu_time_seconds":
            pso_cpu_time,

        "cpu_utilization_percent":
            pso_cpu_utilization,

        "memory_baseline_mb":
            pso_baseline_memory,

        "peak_memory_mb":
            pso_peak_memory,

        "peak_memory_delta_mb":
            pso_peak_delta_memory,

        "evaluations":
            pso_total_evaluations,

        "system":
            platform.system(),

        "python_version":
            platform.python_version(),

        "cpu_count":
            psutil.cpu_count(logical=True)
    })


    add_convergence_data(

        algorithm="PSO",

        history=pso_history,

        evaluations_per_iteration=
            pso_evaluations_per_iteration,

        initial_evaluations=
            pso_initial_evaluations,

        seed=seed
    )


    # ========================================================
    # ABC
    # ========================================================

    print("Running ABC...")


    def run_abc_experiment():

        return run_abc(

            dmat,

            N_CITIES,

            pop_size=ABC_POP_SIZE,

            n_iterations=ABC_ITERATIONS,

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
        abc_best_tour,
        abc_best_fitness,
        abc_history,
        abc_mean_history,
        abc_position_history
    ) = abc_result


    abc_initial_evaluations = (
        ABC_POP_SIZE
    )

    abc_evaluations_per_iteration = (
        2 * ABC_POP_SIZE
    )


    abc_total_evaluations = run_abc.evaluation_history[-1]


    benchmark_results.append({

        "algorithm": "ABC",
        "problem": "TSP",
        "dimension": N_CITIES,
        "seed": seed,

        "best_fitness":
            abc_best_fitness,

        "runtime_seconds":
            abc_runtime,

        "cpu_time_seconds":
            abc_cpu_time,

        "cpu_utilization_percent":
            abc_cpu_utilization,

        "memory_baseline_mb":
            abc_baseline_memory,

        "peak_memory_mb":
            abc_peak_memory,

        "peak_memory_delta_mb":
            abc_peak_delta_memory,

        "evaluations":
            abc_total_evaluations,

        "system":
            platform.system(),

        "python_version":
            platform.python_version(),

        "cpu_count":
            psutil.cpu_count(logical=True)
    })


    add_convergence_data(

        algorithm="ABC",

        history=abc_history,

        evaluations_per_iteration=
            abc_evaluations_per_iteration,

        initial_evaluations=
            abc_initial_evaluations,

        seed=seed
    )


    # ========================================================
    # ACO
    # ========================================================

    print("Running ACO...")


    def run_aco_experiment():

        return run_aco(

            dmat,

            N_CITIES,

            n_ants=ACO_ANTS,

            n_iterations=ACO_ITERATIONS,

            alpha=ACO_ALPHA,

            beta=ACO_BETA,

            rho=ACO_RHO,

            Q=ACO_Q,

            seed=seed,

            verbose=False
        )


    (
        aco_result,
        aco_runtime,
        aco_cpu_time,
        aco_baseline_memory,
        aco_peak_memory,
        aco_peak_delta_memory,
        aco_cpu_utilization
    ) = run_with_resources(
        run_aco_experiment
    )


    (
        aco_best_tour,
        aco_best_fitness,
        aco_history,
        aco_mean_history,
        aco_position_history
    ) = aco_result


    aco_initial_evaluations = 0

    aco_evaluations_per_iteration = (
        ACO_ANTS
    )


    aco_total_evaluations = run_aco.evaluation_history[-1]


    benchmark_results.append({

        "algorithm": "ACO",
        "problem": "TSP",
        "dimension": N_CITIES,
        "seed": seed,

        "best_fitness":
            aco_best_fitness,

        "runtime_seconds":
            aco_runtime,

        "cpu_time_seconds":
            aco_cpu_time,

        "cpu_utilization_percent":
            aco_cpu_utilization,

        "memory_baseline_mb":
            aco_baseline_memory,

        "peak_memory_mb":
            aco_peak_memory,

        "peak_memory_delta_mb":
            aco_peak_delta_memory,

        "evaluations":
            aco_total_evaluations,

        "system":
            platform.system(),

        "python_version":
            platform.python_version(),

        "cpu_count":
            psutil.cpu_count(logical=True)
    })


    add_convergence_data(

        algorithm="ACO",

        history=aco_history,

        evaluations_per_iteration=
            aco_evaluations_per_iteration,

        initial_evaluations=
            aco_initial_evaluations,

        seed=seed
    )


    # ========================================================
    # GWO
    # ========================================================

    print("Running GWO...")


    def run_gwo_experiment():

        return run_gwo(

            dmat,

            N_CITIES,

            pop_size=GWO_POP_SIZE,

            n_iterations=GWO_ITERATIONS,

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
        gwo_best_tour,
        gwo_best_fitness,
        gwo_history,
        gwo_mean_history,
        gwo_position_history
    ) = gwo_result


    gwo_initial_evaluations = (
        GWO_POP_SIZE
    )

    gwo_evaluations_per_iteration = (
        GWO_POP_SIZE
    )


    gwo_total_evaluations = run_gwo.evaluation_history[-1]


    benchmark_results.append({

        "algorithm": "GWO",
        "problem": "TSP",
        "dimension": N_CITIES,
        "seed": seed,

        "best_fitness":
            gwo_best_fitness,

        "runtime_seconds":
            gwo_runtime,

        "cpu_time_seconds":
            gwo_cpu_time,

        "cpu_utilization_percent":
            gwo_cpu_utilization,

        "memory_baseline_mb":
            gwo_baseline_memory,

        "peak_memory_mb":
            gwo_peak_memory,

        "peak_memory_delta_mb":
            gwo_peak_delta_memory,

        "evaluations":
            gwo_total_evaluations,

        "system":
            platform.system(),

        "python_version":
            platform.python_version(),

        "cpu_count":
            psutil.cpu_count(logical=True)
    })


    add_convergence_data(

        algorithm="GWO",

        history=gwo_history,

        evaluations_per_iteration=
            gwo_evaluations_per_iteration,

        initial_evaluations=
            gwo_initial_evaluations,

        seed=seed
    )


    # ========================================================
    # WOA
    # ========================================================

    print("Running WOA...")


    def run_woa_experiment():

        return run_woa(

            dmat,

            N_CITIES,

            pop_size=WOA_POP_SIZE,

            n_iterations=WOA_ITERATIONS,

            b=WOA_B,

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
        woa_best_tour,
        woa_best_fitness,
        woa_history,
        woa_mean_history,
        woa_position_history
    ) = woa_result


    woa_initial_evaluations = (
        WOA_POP_SIZE
    )

    woa_evaluations_per_iteration = (
        WOA_POP_SIZE
    )


    woa_total_evaluations = run_woa.evaluation_history[-1]


    benchmark_results.append({

        "algorithm": "WOA",
        "problem": "TSP",
        "dimension": N_CITIES,
        "seed": seed,

        "best_fitness":
            woa_best_fitness,

        "runtime_seconds":
            woa_runtime,

        "cpu_time_seconds":
            woa_cpu_time,

        "cpu_utilization_percent":
            woa_cpu_utilization,

        "memory_baseline_mb":
            woa_baseline_memory,

        "peak_memory_mb":
            woa_peak_memory,

        "peak_memory_delta_mb":
            woa_peak_delta_memory,

        "evaluations":
            woa_total_evaluations,

        "system":
            platform.system(),

        "python_version":
            platform.python_version(),

        "cpu_count":
            psutil.cpu_count(logical=True)
    })


    add_convergence_data(

        algorithm="WOA",

        history=woa_history,

        evaluations_per_iteration=
            woa_evaluations_per_iteration,

        initial_evaluations=
            woa_initial_evaluations,

        seed=seed
    )


    # ========================================================
    # ALGORITMOS ADICIONALES
    # ========================================================

    additional_algorithms = [
        ("FA", run_fa),
        ("AFSA", run_afsa),
        ("AIS", run_ais),
        ("BFOA", run_bfoa),
        ("IWO", run_iwo),
        ("PFA", run_pfa),
        ("IWD", run_iwd),
    ]

    for algorithm, run_function in additional_algorithms:

        print(f"Running {algorithm}...")

        run_kwargs = {
            "dist_matrix": dmat,
            "n_cities": N_CITIES,
            "pop_size": ADDITIONAL_POP_SIZE,
            "n_iterations": max(
                1,
                (EVAL_BUDGET - ADDITIONAL_POP_SIZE)
                // ADDITIONAL_POP_SIZE
            ),
            "seed": seed,
            "verbose": False,
        }

        if algorithm == "IWD":
            run_kwargs = {
                "dist_matrix": dmat,
                "n_cities": N_CITIES,
                "n_drops": ADDITIONAL_POP_SIZE,
                "n_iterations": max(
                    1,
                    (EVAL_BUDGET - ADDITIONAL_POP_SIZE)
                    // ADDITIONAL_POP_SIZE
                ),
                "seed": seed,
                "verbose": False,
            }

        (
            result,
            runtime,
            cpu_time,
            baseline_memory,
            peak_memory,
            peak_delta_memory,
            cpu_utilization,
        ) = run_with_resources(
            lambda run_function=run_function, run_kwargs=run_kwargs:
                run_function(**run_kwargs)
        )

        (
            best_tour,
            best_fitness,
            history,
            mean_history,
            position_history,
        ) = result

        evaluations = list(
            run_function.evaluation_history
        )

        benchmark_results.append({
            "algorithm": algorithm,
            "problem": "TSP",
            "dimension": N_CITIES,
            "seed": seed,
            "best_fitness": best_fitness,
            "runtime_seconds": runtime,
            "cpu_time_seconds": cpu_time,
            "cpu_utilization_percent": cpu_utilization,
            "memory_baseline_mb": baseline_memory,
            "peak_memory_mb": peak_memory,
            "peak_memory_delta_mb": peak_delta_memory,
            "evaluations": evaluations[-1],
            "system": platform.system(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=True),
        })

        for evaluation, fitness in zip(evaluations, history):
            convergence_results.append({
                "algorithm": algorithm,
                "problem": "TSP",
                "dimension": N_CITIES,
                "seed": seed,
                "evaluation": evaluation,
                "fitness": fitness,
            })


    # ========================================================
    # RESUMEN DE LA EJECUCIÓN
    # ========================================================

    print()

    print(
        f"GA  | fitness={ga_best_fitness:.4f} | "
        f"time={ga_runtime:.4f}s | "
        f"evals={ga_total_evaluations}"
    )

    print(
        f"cGA | fitness={cga_best_fitness:.4f} | "
        f"time={cga_runtime:.4f}s | "
        f"evals={cga_total_evaluations}"
    )

    print(
        f"PSO | fitness={pso_best_fitness:.4f} | "
        f"time={pso_runtime:.4f}s | "
        f"evals={pso_total_evaluations}"
    )

    print(
        f"ABC | fitness={abc_best_fitness:.4f} | "
        f"time={abc_runtime:.4f}s | "
        f"evals={abc_total_evaluations}"
    )

    print(
        f"ACO | fitness={aco_best_fitness:.4f} | "
        f"time={aco_runtime:.4f}s | "
        f"evals={aco_total_evaluations}"
    )

    print(
        f"GWO | fitness={gwo_best_fitness:.4f} | "
        f"time={gwo_runtime:.4f}s | "
        f"evals={gwo_total_evaluations}"
    )

    print(
        f"WOA | fitness={woa_best_fitness:.4f} | "
        f"time={woa_runtime:.4f}s | "
        f"evals={woa_total_evaluations}"
    )

    for result in benchmark_results[-7:]:
        print(
            f"{result['algorithm']:<4} | "
            f"fitness={result['best_fitness']:.4f} | "
            f"time={result['runtime_seconds']:.4f}s | "
            f"evals={result['evaluations']}"
        )


# ============================================================
# DATAFRAMES
# ============================================================

results_df = pd.DataFrame(
    benchmark_results
)

convergence_df = pd.DataFrame(
    convergence_results
)


# ============================================================
# GUARDAR RESULTADOS
# ============================================================

results_df.to_csv(

    RESULTS_FILE,

    index=False
)


convergence_df.to_csv(

    CONVERGENCE_FILE,

    index=False
)


# ============================================================
# RESUMEN ESTADÍSTICO
# ============================================================

summary = (

    results_df

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

    .reindex(
        ALGORITHMS
    )

    .reset_index()
)


# ============================================================
# GUARDAR RESUMEN
# ============================================================

summary.to_csv(

    SUMMARY_FILE,

    index=False
)


# ============================================================
# RESULTADO FINAL
# ============================================================

print()

print("=" * 75)

print("BENCHMARK COMPLETED")

print("=" * 75)


print()

print(
    summary.to_string(
        index=False
    )
)


print()

print(
    "Results saved to:"
)

print(
    RESULTS_FILE
)


print()

print(
    "Convergence saved to:"
)

print(
    CONVERGENCE_FILE
)


print()

print(
    "Summary saved to:"
)

print(
    SUMMARY_FILE
)