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

sys.path.append(
    TASK_DIR
)


# ============================================================
# IMPORTS
# ============================================================

from Sphere_Function_Benchmark.Sphere_GA import run_ga
from Sphere_Function_Benchmark.Sphere_cGA import run_cga
from Sphere_Function_Benchmark.Sphere_PSO import run_pso

from Sphere_Function_Benchmark.Sphere_ABC import run_abc
from Sphere_Function_Benchmark.Sphere_ACOR import run_acor
from Sphere_Function_Benchmark.Sphere_AFSA import run_afsa
from Sphere_Function_Benchmark.Sphere_AIS import run_ais
from Sphere_Function_Benchmark.Sphere_BFOA import run_bfoa
from Sphere_Function_Benchmark.Sphere_FA import run_fa
from Sphere_Function_Benchmark.Sphere_GWO import run_gwo
from Sphere_Function_Benchmark.Sphere_IWD import run_iwd
from Sphere_Function_Benchmark.Sphere_IWO import run_iwo
from Sphere_Function_Benchmark.Sphere_PFA import run_pfa
from Sphere_Function_Benchmark.Sphere_WOA import run_woa


# ============================================================
# CONFIGURACIÓN DEL EXPERIMENTO
# ============================================================

PROBLEM = "Sphere"

N_DIM = 10

BOUNDS = (-5.12, 5.12)

EVAL_BUDGET = 20_000

N_RUNS = 30

SEEDS = range(
    N_RUNS
)


# ============================================================
# PARÁMETROS DE LOS ALGORITMOS
# ============================================================

# -----------------------------
# GA
# -----------------------------

GA_POP_SIZE = 100


# -----------------------------
# cGA
# -----------------------------

CGA_POP_SIZE = 100

CGA_BITS_PER_VAR = 16


# -----------------------------
# PSO
# -----------------------------

PSO_PARTICLES = 50


# -----------------------------
# ABC
# -----------------------------

ABC_POP_SIZE = 30

ABC_LIMIT = 50


# -----------------------------
# ACOR
# -----------------------------

ACOR_ARCHIVE_SIZE = 30

ACOR_N_ANTS = 30

ACOR_Q = 0.5

ACOR_XI = 0.85


# -----------------------------
# AFSA
# -----------------------------

AFSA_POP_SIZE = 30

AFSA_VISUAL = 2.0

AFSA_STEP = 0.5

AFSA_TRY_NUMBER = 5

AFSA_CROWDING_FACTOR = 0.6


# -----------------------------
# AIS
# -----------------------------

AIS_POP_SIZE = 30

AIS_N_SELECT = 10

AIS_CLONES_PER_ANTIBODY = 5

AIS_BETA = 2.0

AIS_REPLACE_RATE = 0.2


# -----------------------------
# BFOA
# -----------------------------

BFOA_POP_SIZE = 30

BFOA_NC = 10

BFOA_NS = 4

BFOA_NRE = 4

BFOA_PED = 0.25

BFOA_C = 0.1


# -----------------------------
# FA
# -----------------------------

FA_POP_SIZE = 30

FA_ALPHA = 0.5

FA_BETA0 = 1.0

FA_GAMMA = 0.01

FA_ALPHA_DECAY = 0.97


# -----------------------------
# GWO
# -----------------------------

GWO_POP_SIZE = 30


# -----------------------------
# IWD
# -----------------------------

IWD_POP_SIZE = 30

IWD_N_CANDIDATES = 5

IWD_AV = 1.0

IWD_BV = 0.01

IWD_CV = 1.0

IWD_INIT_VEL = 1.0

IWD_RADIUS_INITIAL = 1.5

IWD_RADIUS_FINAL = 0.01

IWD_PULL_STRENGTH = 0.05


# -----------------------------
# IWO
# -----------------------------

IWO_POP_SIZE = 30

IWO_MIN_SEEDS = 0

IWO_MAX_SEEDS = 5

IWO_EXPONENT = 2

IWO_SIGMA_INITIAL = 0.5

IWO_SIGMA_FINAL = 0.001


# -----------------------------
# PFA
# -----------------------------

PFA_POP_SIZE = 30

PFA_SELECTION_RATE = 0.3

PFA_MAX_SEEDS = 5

PFA_SIGMA_INITIAL = 0.5

PFA_SIGMA_FINAL = 0.001

PFA_EXPONENT = 2

PFA_POLLINATION_RATE = 0.1


# -----------------------------
# WOA
# -----------------------------

WOA_POP_SIZE = 30

WOA_B = 1.0


# ============================================================
# CARPETA DE RESULTADOS
# ============================================================

RESULTS_DIR = os.path.join(
    TASK_DIR,
    "Results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


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
# EJECUTAR ALGORITMO Y MEDIR RECURSOS
# ============================================================

def run_with_resources(
    algorithm_function
):

    memory_monitor = MemoryMonitor(
        PROCESS,
        interval=0.01
    )

    wall_start = time.perf_counter()

    cpu_start = time.process_time()

    memory_monitor.start()

    result = algorithm_function()

    baseline_mb, peak_mb = (
        memory_monitor.stop()
    )

    cpu_time = (
        time.process_time()
        - cpu_start
    )

    wall_time = (
        time.perf_counter()
        - wall_start
    )

    peak_delta_mb = max(
        0.0,
        peak_mb - baseline_mb
    )

    cpu_count = psutil.cpu_count(
        logical=True
    )

    if wall_time > 0 and cpu_count:

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

        cpu_utilization,
    )


# ============================================================
# CONVERGENCIA
# ============================================================

def create_convergence_dataframe(
    algorithm,
    history,
    evaluation_history,
    seed,
):

    history = np.asarray(
        history
    )

    evaluation_history = np.asarray(
        evaluation_history
    )

    # Nos aseguramos de que ambas series tengan
    # exactamente la misma longitud.
    n = min(
        len(history),
        len(evaluation_history)
    )

    return pd.DataFrame({

        "algorithm": algorithm,

        "problem": PROBLEM,

        "dimension": N_DIM,

        "seed": seed,

        "evaluation":
            evaluation_history[:n],

        "fitness":
            history[:n],
    })


# ============================================================
# RESULTADOS
# ============================================================

final_results = []

convergence_results = []


# ============================================================
# EJECUCIONES
# ============================================================

for seed in SEEDS:

    print("\n")

    print("=" * 80)

    print(
        f"RUN {seed + 1}/{N_RUNS} | "
        f"SEED = {seed}"
    )

    print("=" * 80)


    # ========================================================
    # GA
    # ========================================================

    print("\nRunning GA...")


    ga_generations = (

        EVAL_BUDGET
        - GA_POP_SIZE

    ) // GA_POP_SIZE


    def run_ga_experiment():

        return run_ga(

            n_dim=N_DIM,

            pop_size=GA_POP_SIZE,

            n_generations=ga_generations,

            bounds=BOUNDS,

            seed=seed,

            verbose=False,
        )


    (
        ga_result,
        ga_runtime,
        ga_cpu_time,
        ga_baseline_memory,
        ga_peak_memory,
        ga_peak_delta_memory,
        ga_cpu_utilization,

    ) = run_with_resources(
        run_ga_experiment
    )


    (
        ga_best_x,
        ga_best_f,
        ga_history,
        ga_mean_history,
        ga_position_history,

    ) = ga_result


    ga_evaluation_history = (
        run_ga.evaluation_history
    )

    ga_total_evaluations = (
        ga_evaluation_history[-1]
    )


    ga_success = (
        ga_best_f
        <= SUCCESS_THRESHOLD
    )


    final_results.append({

        "algorithm": "GA",

        "problem": PROBLEM,

        "dimension": N_DIM,

        "seed": seed,

        "best_fitness": ga_best_f,

        "runtime_seconds": ga_runtime,

        "cpu_time_seconds": ga_cpu_time,

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

        "success":
            int(ga_success),
    })


    convergence_results.append(
        create_convergence_dataframe(
            "GA",
            ga_history,
            ga_evaluation_history,
            seed
        )
    )


    # ========================================================
    # cGA
    # ========================================================

    print("Running cGA...")


    cga_generations = (
        EVAL_BUDGET // 2
    )


    def run_cga_experiment():

        return run_cga(

            n_dim=N_DIM,

            bits_per_var=CGA_BITS_PER_VAR,

            bounds=BOUNDS,

            pop_size=CGA_POP_SIZE,

            n_generations=cga_generations,

            seed=seed,

            verbose=False,
        )


    (
        cga_result,
        cga_runtime,
        cga_cpu_time,
        cga_baseline_memory,
        cga_peak_memory,
        cga_peak_delta_memory,
        cga_cpu_utilization,

    ) = run_with_resources(
        run_cga_experiment
    )


    (
        cga_best_x,
        cga_best_f,
        cga_history,
        cga_position_history,

    ) = cga_result


    cga_evaluation_history = (
        run_cga.evaluation_history
    )

    cga_total_evaluations = (
        cga_evaluation_history[-1]
    )


    cga_success = (
        cga_best_f
        <= SUCCESS_THRESHOLD
    )


    final_results.append({

        "algorithm": "cGA",

        "problem": PROBLEM,

        "dimension": N_DIM,

        "seed": seed,

        "best_fitness": cga_best_f,

        "runtime_seconds": cga_runtime,

        "cpu_time_seconds": cga_cpu_time,

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

        "success":
            int(cga_success),
    })


    convergence_results.append(
        create_convergence_dataframe(
            "cGA",
            cga_history,
            cga_evaluation_history,
            seed
        )
    )


    # ========================================================
    # PSO
    # ========================================================

    print("Running PSO...")


    pso_iterations = (

        EVAL_BUDGET
        - PSO_PARTICLES

    ) // PSO_PARTICLES


    def run_pso_experiment():

        return run_pso(

            n_dim=N_DIM,

            n_particles=PSO_PARTICLES,

            n_iterations=pso_iterations,

            bounds=BOUNDS,

            seed=seed,

            verbose=False,
        )


    (
        pso_result,
        pso_runtime,
        pso_cpu_time,
        pso_baseline_memory,
        pso_peak_memory,
        pso_peak_delta_memory,
        pso_cpu_utilization,

    ) = run_with_resources(
        run_pso_experiment
    )


    (
        pso_best_x,
        pso_best_f,
        pso_history,
        pso_position_history,

    ) = pso_result


    pso_evaluation_history = (
        run_pso.evaluation_history
    )

    pso_total_evaluations = (
        pso_evaluation_history[-1]
    )


    pso_success = (
        pso_best_f
        <= SUCCESS_THRESHOLD
    )


    final_results.append({

        "algorithm": "PSO",

        "problem": PROBLEM,

        "dimension": N_DIM,

        "seed": seed,

        "best_fitness": pso_best_f,

        "runtime_seconds": pso_runtime,

        "cpu_time_seconds": pso_cpu_time,

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

        "success":
            int(pso_success),
    })


    convergence_results.append(
        create_convergence_dataframe(
            "PSO",
            pso_history,
            pso_evaluation_history,
            seed
        )
    )


    # ========================================================
    # ABC
    # ========================================================

    print("Running ABC...")


    abc_iterations = (

        EVAL_BUDGET
        - ABC_POP_SIZE

    ) // (

        2 * ABC_POP_SIZE
    )


    def run_abc_experiment():

        return run_abc(

            n_dim=N_DIM,

            pop_size=ABC_POP_SIZE,

            n_iterations=abc_iterations,

            bounds=BOUNDS,

            limit=ABC_LIMIT,

            seed=seed,

            verbose=False,
        )


    (
        abc_result,
        abc_runtime,
        abc_cpu_time,
        abc_baseline_memory,
        abc_peak_memory,
        abc_peak_delta_memory,
        abc_cpu_utilization,

    ) = run_with_resources(
        run_abc_experiment
    )


    (
        abc_best_x,
        abc_best_f,
        abc_history,
        abc_mean_history,
        abc_position_history,

    ) = abc_result


    abc_evaluation_history = (
        run_abc.evaluation_history
    )

    abc_total_evaluations = (
        abc_evaluation_history[-1]
    )


    abc_success = (
        abc_best_f
        <= SUCCESS_THRESHOLD
    )


    final_results.append({

        "algorithm": "ABC",

        "problem": PROBLEM,

        "dimension": N_DIM,

        "seed": seed,

        "best_fitness": abc_best_f,

        "runtime_seconds": abc_runtime,

        "cpu_time_seconds": abc_cpu_time,

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

        "success":
            int(abc_success),
    })


    convergence_results.append(
        create_convergence_dataframe(
            "ABC",
            abc_history,
            abc_evaluation_history,
            seed
        )
    )


    # ========================================================
    # ACOR
    # ========================================================

    print("Running ACOR...")


    acor_iterations = (

        EVAL_BUDGET
        - ACOR_ARCHIVE_SIZE

    ) // ACOR_N_ANTS


    def run_acor_experiment():

        return run_acor(

            n_dim=N_DIM,

            archive_size=ACOR_ARCHIVE_SIZE,

            n_ants=ACOR_N_ANTS,

            n_iterations=acor_iterations,

            bounds=BOUNDS,

            q=ACOR_Q,

            xi=ACOR_XI,

            seed=seed,

            verbose=False,
        )


    (
        acor_result,
        acor_runtime,
        acor_cpu_time,
        acor_baseline_memory,
        acor_peak_memory,
        acor_peak_delta_memory,
        acor_cpu_utilization,

    ) = run_with_resources(
        run_acor_experiment
    )


    (
        acor_best_x,
        acor_best_f,
        acor_history,
        acor_mean_history,
        acor_position_history,

    ) = acor_result


    acor_evaluation_history = (
        run_acor.evaluation_history
    )

    acor_total_evaluations = (
        acor_evaluation_history[-1]
    )


    acor_success = (
        acor_best_f
        <= SUCCESS_THRESHOLD
    )


    final_results.append({

        "algorithm": "ACOR",

        "problem": PROBLEM,

        "dimension": N_DIM,

        "seed": seed,

        "best_fitness": acor_best_f,

        "runtime_seconds": acor_runtime,

        "cpu_time_seconds": acor_cpu_time,

        "cpu_utilization_percent":
            acor_cpu_utilization,

        "memory_baseline_mb":
            acor_baseline_memory,

        "peak_memory_mb":
            acor_peak_memory,

        "peak_memory_delta_mb":
            acor_peak_delta_memory,

        "evaluations":
            acor_total_evaluations,

        "success":
            int(acor_success),
    })


    convergence_results.append(
        create_convergence_dataframe(
            "ACOR",
            acor_history,
            acor_evaluation_history,
            seed
        )
    )


    # ========================================================
    # AFSA
    # ========================================================

    print("Running AFSA...")


    # Worst-case:
    #
    # - Initial population: pop_size
    #
    # - For each fish:
    #   * swarming center: 1 evaluation
    #   * following move: 1 evaluation
    #   * preying: try_number evaluations
    #   * resulting move: 1 evaluation
    #
    # Conservatively:
    # 1 + 1 + try_number + 1
    #
    # evaluations/fish.
    #
    # Therefore:
    # max_eval_per_iteration =
    # pop_size * (try_number + 3)

    AFSA_MAX_EVAL_PER_ITER = (
        AFSA_POP_SIZE
        * (AFSA_TRY_NUMBER + 3)
    )


    afsa_iterations = (

        EVAL_BUDGET
        - AFSA_POP_SIZE

    ) // AFSA_MAX_EVAL_PER_ITER


    def run_afsa_experiment():

        return run_afsa(

            n_dim=N_DIM,

            pop_size=AFSA_POP_SIZE,

            n_iterations=afsa_iterations,

            bounds=BOUNDS,

            visual=AFSA_VISUAL,

            step=AFSA_STEP,

            try_number=AFSA_TRY_NUMBER,

            crowding_factor=
                AFSA_CROWDING_FACTOR,

            seed=seed,

            verbose=False,
        )


    (
        afsa_result,
        afsa_runtime,
        afsa_cpu_time,
        afsa_baseline_memory,
        afsa_peak_memory,
        afsa_peak_delta_memory,
        afsa_cpu_utilization,

    ) = run_with_resources(
        run_afsa_experiment
    )


    (
        afsa_best_x,
        afsa_best_f,
        afsa_history,
        afsa_mean_history,
        afsa_position_history,

    ) = afsa_result


    afsa_evaluation_history = (
        run_afsa.evaluation_history
    )

    afsa_total_evaluations = (
        afsa_evaluation_history[-1]
    )


    afsa_success = (
        afsa_best_f
        <= SUCCESS_THRESHOLD
    )


    final_results.append({

        "algorithm": "AFSA",

        "problem": PROBLEM,

        "dimension": N_DIM,

        "seed": seed,

        "best_fitness": afsa_best_f,

        "runtime_seconds": afsa_runtime,

        "cpu_time_seconds": afsa_cpu_time,

        "cpu_utilization_percent":
            afsa_cpu_utilization,

        "memory_baseline_mb":
            afsa_baseline_memory,

        "peak_memory_mb":
            afsa_peak_memory,

        "peak_memory_delta_mb":
            afsa_peak_delta_memory,

        "evaluations":
            afsa_total_evaluations,

        "success":
            int(afsa_success),
    })


    convergence_results.append(
        create_convergence_dataframe(
            "AFSA",
            afsa_history,
            afsa_evaluation_history,
            seed
        )
    )


    # ========================================================
    # AIS
    # ========================================================

    print("Running AIS...")


    # Evaluations:
    #
    # Initial population:
    #     pop_size
    #
    # Each iteration:
    #     n_select * clones_per_antibody
    #     +
    #     n_replace
    #
    # n_replace = max(1, replace_rate * pop_size)

    AIS_N_REPLACE = max(
        1,
        int(
            AIS_REPLACE_RATE
            * AIS_POP_SIZE
        )
    )


    AIS_MAX_EVAL_PER_ITER = (

        AIS_N_SELECT
        * AIS_CLONES_PER_ANTIBODY

        + AIS_N_REPLACE
    )


    ais_iterations = (

        EVAL_BUDGET
        - AIS_POP_SIZE

    ) // AIS_MAX_EVAL_PER_ITER


    def run_ais_experiment():

        return run_ais(

            n_dim=N_DIM,

            pop_size=AIS_POP_SIZE,

            n_iterations=ais_iterations,

            bounds=BOUNDS,

            n_select=AIS_N_SELECT,

            clones_per_antibody=
                AIS_CLONES_PER_ANTIBODY,

            beta=AIS_BETA,

            replace_rate=
                AIS_REPLACE_RATE,

            seed=seed,

            verbose=False,
        )


    (
        ais_result,
        ais_runtime,
        ais_cpu_time,
        ais_baseline_memory,
        ais_peak_memory,
        ais_peak_delta_memory,
        ais_cpu_utilization,

    ) = run_with_resources(
        run_ais_experiment
    )


    (
        ais_best_x,
        ais_best_f,
        ais_history,
        ais_mean_history,
        ais_position_history,

    ) = ais_result


    ais_evaluation_history = (
        run_ais.evaluation_history
    )

    ais_total_evaluations = (
        ais_evaluation_history[-1]
    )


    ais_success = (
        ais_best_f
        <= SUCCESS_THRESHOLD
    )


    final_results.append({

        "algorithm": "AIS",

        "problem": PROBLEM,

        "dimension": N_DIM,

        "seed": seed,

        "best_fitness": ais_best_f,

        "runtime_seconds": ais_runtime,

        "cpu_time_seconds": ais_cpu_time,

        "cpu_utilization_percent":
            ais_cpu_utilization,

        "memory_baseline_mb":
            ais_baseline_memory,

        "peak_memory_mb":
            ais_peak_memory,

        "peak_memory_delta_mb":
            ais_peak_delta_memory,

        "evaluations":
            ais_total_evaluations,

        "success":
            int(ais_success),
    })


    convergence_results.append(
        create_convergence_dataframe(
            "AIS",
            ais_history,
            ais_evaluation_history,
            seed
        )
    )


    # ========================================================
    # BFOA
    # ========================================================

    print("Running BFOA...")


    # Conservative upper bound:
    #
    # Initial population:
    #     pop_size
    #
    # Every chemotaxis step:
    #
    #   candidate = 1 evaluation
    #
    #   if accepted:
    #       initial accepted position is evaluated
    #       up to Ns swim trials are evaluated
    #       each accepted swim is evaluated again
    #
    # The implementation can therefore perform several evaluations
    # per bacterium and step.
    #
    # To keep the benchmark safely under the budget, use:
    #
    #   1 + 1 + 2*Ns
    #
    # evaluations/bacterium as a conservative bound.

    BFOA_MAX_EVAL_PER_BACTERIUM = (
        1 + 1 + 2 * BFOA_NS
    )


    BFOA_MAX_EVAL_PER_ITER = (
        BFOA_POP_SIZE
        * BFOA_MAX_EVAL_PER_BACTERIUM
    )


    bfoa_iterations = (

        EVAL_BUDGET
        - BFOA_POP_SIZE

    ) // BFOA_MAX_EVAL_PER_ITER


    def run_bfoa_experiment():

        return run_bfoa(

            n_dim=N_DIM,

            pop_size=BFOA_POP_SIZE,

            n_iterations=bfoa_iterations,

            bounds=BOUNDS,

            Nc=BFOA_NC,

            Ns=BFOA_NS,

            Nre=BFOA_NRE,

            Ped=BFOA_PED,

            C=BFOA_C,

            seed=seed,

            verbose=False,
        )


    (
        bfoa_result,
        bfoa_runtime,
        bfoa_cpu_time,
        bfoa_baseline_memory,
        bfoa_peak_memory,
        bfoa_peak_delta_memory,
        bfoa_cpu_utilization,

    ) = run_with_resources(
        run_bfoa_experiment
    )


    (
        bfoa_best_x,
        bfoa_best_f,
        bfoa_history,
        bfoa_mean_history,
        bfoa_position_history,

    ) = bfoa_result


    bfoa_evaluation_history = (
        run_bfoa.evaluation_history
    )

    bfoa_total_evaluations = (
        bfoa_evaluation_history[-1]
    )


    bfoa_success = (
        bfoa_best_f
        <= SUCCESS_THRESHOLD
    )


    final_results.append({

        "algorithm": "BFOA",

        "problem": PROBLEM,

        "dimension": N_DIM,

        "seed": seed,

        "best_fitness": bfoa_best_f,

        "runtime_seconds": bfoa_runtime,

        "cpu_time_seconds": bfoa_cpu_time,

        "cpu_utilization_percent":
            bfoa_cpu_utilization,

        "memory_baseline_mb":
            bfoa_baseline_memory,

        "peak_memory_mb":
            bfoa_peak_memory,

        "peak_memory_delta_mb":
            bfoa_peak_delta_memory,

        "evaluations":
            bfoa_total_evaluations,

        "success":
            int(bfoa_success),
    })


    convergence_results.append(
        create_convergence_dataframe(
            "BFOA",
            bfoa_history,
            bfoa_evaluation_history,
            seed
        )
    )


    # ========================================================
    # FA
    # ========================================================

    print("Running FA...")


    # Worst-case:
    #
    # Every firefly can be moved by every other firefly:
    #
    # pop_size * (pop_size - 1)
    #
    # Each movement evaluates the new position.

    FA_MAX_EVAL_PER_ITER = (
        FA_POP_SIZE
        * (FA_POP_SIZE - 1)
    )


    fa_iterations = (

        EVAL_BUDGET
        - FA_POP_SIZE

    ) // FA_MAX_EVAL_PER_ITER


    def run_fa_experiment():

        return run_fa(

            n_dim=N_DIM,

            pop_size=FA_POP_SIZE,

            n_iterations=fa_iterations,

            bounds=BOUNDS,

            alpha=FA_ALPHA,

            beta0=FA_BETA0,

            gamma=FA_GAMMA,

            alpha_decay=
                FA_ALPHA_DECAY,

            seed=seed,

            verbose=False,
        )


    (
        fa_result,
        fa_runtime,
        fa_cpu_time,
        fa_baseline_memory,
        fa_peak_memory,
        fa_peak_delta_memory,
        fa_cpu_utilization,

    ) = run_with_resources(
        run_fa_experiment
    )


    (
        fa_best_x,
        fa_best_f,
        fa_history,
        fa_mean_history,
        fa_position_history,

    ) = fa_result


    fa_evaluation_history = (
        run_fa.evaluation_history
    )

    fa_total_evaluations = (
        fa_evaluation_history[-1]
    )


    fa_success = (
        fa_best_f
        <= SUCCESS_THRESHOLD
    )


    final_results.append({

        "algorithm": "FA",

        "problem": PROBLEM,

        "dimension": N_DIM,

        "seed": seed,

        "best_fitness": fa_best_f,

        "runtime_seconds": fa_runtime,

        "cpu_time_seconds": fa_cpu_time,

        "cpu_utilization_percent":
            fa_cpu_utilization,

        "memory_baseline_mb":
            fa_baseline_memory,

        "peak_memory_mb":
            fa_peak_memory,

        "peak_memory_delta_mb":
            fa_peak_delta_memory,

        "evaluations":
            fa_total_evaluations,

        "success":
            int(fa_success),
    })


    convergence_results.append(
        create_convergence_dataframe(
            "FA",
            fa_history,
            fa_evaluation_history,
            seed
        )
    )


    # ========================================================
    # GWO
    # ========================================================

    print("Running GWO...")


    gwo_iterations = (

        EVAL_BUDGET
        - GWO_POP_SIZE

    ) // GWO_POP_SIZE


    def run_gwo_experiment():

        return run_gwo(

            n_dim=N_DIM,

            pop_size=GWO_POP_SIZE,

            n_iterations=gwo_iterations,

            bounds=BOUNDS,

            seed=seed,

            verbose=False,
        )


    (
        gwo_result,
        gwo_runtime,
        gwo_cpu_time,
        gwo_baseline_memory,
        gwo_peak_memory,
        gwo_peak_delta_memory,
        gwo_cpu_utilization,

    ) = run_with_resources(
        run_gwo_experiment
    )


    (
        gwo_best_x,
        gwo_best_f,
        gwo_history,
        gwo_mean_history,
        gwo_position_history,

    ) = gwo_result


    gwo_evaluation_history = (
        run_gwo.evaluation_history
    )

    gwo_total_evaluations = (
        gwo_evaluation_history[-1]
    )


    gwo_success = (
        gwo_best_f
        <= SUCCESS_THRESHOLD
    )


    final_results.append({

        "algorithm": "GWO",

        "problem": PROBLEM,

        "dimension": N_DIM,

        "seed": seed,

        "best_fitness": gwo_best_f,

        "runtime_seconds": gwo_runtime,

        "cpu_time_seconds": gwo_cpu_time,

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

        "success":
            int(gwo_success),
    })


    convergence_results.append(
        create_convergence_dataframe(
            "GWO",
            gwo_history,
            gwo_evaluation_history,
            seed
        )
    )


    # ========================================================
    # IWD
    # ========================================================

    print("Running IWD...")


    # Per iteration:
    #
    # Each water drop generates n_candidates
    # and evaluates all of them.
    #
    # Then:
    #   evaluate_population(positions)
    #   evaluate_population(positions) again
    #
    # Therefore:
    #
    # pop_size * n_candidates
    # + pop_size
    # + pop_size
    #
    # evaluations / iteration.

    IWD_MAX_EVAL_PER_ITER = (

        IWD_POP_SIZE
        * IWD_N_CANDIDATES

        + IWD_POP_SIZE
        + IWD_POP_SIZE
    )


    iwd_iterations = (

        EVAL_BUDGET
        - IWD_POP_SIZE

    ) // IWD_MAX_EVAL_PER_ITER


    def run_iwd_experiment():

        return run_iwd(

            n_dim=N_DIM,

            pop_size=IWD_POP_SIZE,

            n_iterations=iwd_iterations,

            bounds=BOUNDS,

            n_candidates=
                IWD_N_CANDIDATES,

            av=IWD_AV,

            bv=IWD_BV,

            cv=IWD_CV,

            init_vel=IWD_INIT_VEL,

            radius_initial=
                IWD_RADIUS_INITIAL,

            radius_final=
                IWD_RADIUS_FINAL,

            pull_strength=
                IWD_PULL_STRENGTH,

            seed=seed,

            verbose=False,
        )


    (
        iwd_result,
        iwd_runtime,
        iwd_cpu_time,
        iwd_baseline_memory,
        iwd_peak_memory,
        iwd_peak_delta_memory,
        iwd_cpu_utilization,

    ) = run_with_resources(
        run_iwd_experiment
    )


    (
        iwd_best_x,
        iwd_best_f,
        iwd_history,
        iwd_mean_history,
        iwd_position_history,

    ) = iwd_result


    iwd_evaluation_history = (
        run_iwd.evaluation_history
    )

    iwd_total_evaluations = (
        iwd_evaluation_history[-1]
    )


    iwd_success = (
        iwd_best_f
        <= SUCCESS_THRESHOLD
    )


    final_results.append({

        "algorithm": "IWD",

        "problem": PROBLEM,

        "dimension": N_DIM,

        "seed": seed,

        "best_fitness": iwd_best_f,

        "runtime_seconds": iwd_runtime,

        "cpu_time_seconds": iwd_cpu_time,

        "cpu_utilization_percent":
            iwd_cpu_utilization,

        "memory_baseline_mb":
            iwd_baseline_memory,

        "peak_memory_mb":
            iwd_peak_memory,

        "peak_memory_delta_mb":
            iwd_peak_delta_memory,

        "evaluations":
            iwd_total_evaluations,

        "success":
            int(iwd_success),
    })


    convergence_results.append(
        create_convergence_dataframe(
            "IWD",
            iwd_history,
            iwd_evaluation_history,
            seed
        )
    )


    # ========================================================
    # IWO
    # ========================================================

    print("Running IWO...")


    # Maximum seeds:
    #
    # pop_size * max_seeds
    #
    # plus initial population evaluation.
    #
    # Worst case per iteration:
    #
    # pop_size * max_seeds

    IWO_MAX_EVAL_PER_ITER = (
        IWO_POP_SIZE
        * IWO_MAX_SEEDS
    )


    iwo_iterations = (

        EVAL_BUDGET
        - IWO_POP_SIZE

    ) // IWO_MAX_EVAL_PER_ITER


    def run_iwo_experiment():

        return run_iwo(

            n_dim=N_DIM,

            pop_size=IWO_POP_SIZE,

            n_iterations=iwo_iterations,

            bounds=BOUNDS,

            min_seeds=IWO_MIN_SEEDS,

            max_seeds=IWO_MAX_SEEDS,

            exponent=IWO_EXPONENT,

            sigma_initial=
                IWO_SIGMA_INITIAL,

            sigma_final=
                IWO_SIGMA_FINAL,

            seed=seed,

            verbose=False,
        )


    (
        iwo_result,
        iwo_runtime,
        iwo_cpu_time,
        iwo_baseline_memory,
        iwo_peak_memory,
        iwo_peak_delta_memory,
        iwo_cpu_utilization,

    ) = run_with_resources(
        run_iwo_experiment
    )


    (
        iwo_best_x,
        iwo_best_f,
        iwo_history,
        iwo_mean_history,
        iwo_position_history,

    ) = iwo_result


    iwo_evaluation_history = (
        run_iwo.evaluation_history
    )

    iwo_total_evaluations = (
        iwo_evaluation_history[-1]
    )


    iwo_success = (
        iwo_best_f
        <= SUCCESS_THRESHOLD
    )


    final_results.append({

        "algorithm": "IWO",

        "problem": PROBLEM,

        "dimension": N_DIM,

        "seed": seed,

        "best_fitness": iwo_best_f,

        "runtime_seconds": iwo_runtime,

        "cpu_time_seconds": iwo_cpu_time,

        "cpu_utilization_percent":
            iwo_cpu_utilization,

        "memory_baseline_mb":
            iwo_baseline_memory,

        "peak_memory_mb":
            iwo_peak_memory,

        "peak_memory_delta_mb":
            iwo_peak_delta_memory,

        "evaluations":
            iwo_total_evaluations,

        "success":
            int(iwo_success),
    })


    convergence_results.append(
        create_convergence_dataframe(
            "IWO",
            iwo_history,
            iwo_evaluation_history,
            seed
        )
    )


    # ========================================================
    # PFA
    # ========================================================

    print("Running PFA...")


    PFA_N_SELECT = max(
        1,
        int(
            PFA_SELECTION_RATE
            * PFA_POP_SIZE
        )
    )


    PFA_N_POLLINATE = max(
        1,
        int(
            PFA_POLLINATION_RATE
            * PFA_POP_SIZE
        )
    )


    # Worst-case:
    #
    # seeds:
    #     n_select * max_seeds
    #
    # pollination:
    #     n_pollinate
    #
    # Both generate one evaluation each.

    PFA_MAX_EVAL_PER_ITER = (

        PFA_N_SELECT
        * PFA_MAX_SEEDS

        + PFA_N_POLLINATE
    )


    pfa_iterations = (

        EVAL_BUDGET
        - PFA_POP_SIZE

    ) // PFA_MAX_EVAL_PER_ITER


    def run_pfa_experiment():

        return run_pfa(

            n_dim=N_DIM,

            pop_size=PFA_POP_SIZE,

            n_iterations=pfa_iterations,

            bounds=BOUNDS,

            selection_rate=
                PFA_SELECTION_RATE,

            max_seeds=
                PFA_MAX_SEEDS,

            sigma_initial=
                PFA_SIGMA_INITIAL,

            sigma_final=
                PFA_SIGMA_FINAL,

            exponent=
                PFA_EXPONENT,

            pollination_rate=
                PFA_POLLINATION_RATE,

            seed=seed,

            verbose=False,
        )


    (
        pfa_result,
        pfa_runtime,
        pfa_cpu_time,
        pfa_baseline_memory,
        pfa_peak_memory,
        pfa_peak_delta_memory,
        pfa_cpu_utilization,

    ) = run_with_resources(
        run_pfa_experiment
    )


    (
        pfa_best_x,
        pfa_best_f,
        pfa_history,
        pfa_mean_history,
        pfa_position_history,

    ) = pfa_result


    pfa_evaluation_history = (
        run_pfa.evaluation_history
    )

    pfa_total_evaluations = (
        pfa_evaluation_history[-1]
    )


    pfa_success = (
        pfa_best_f
        <= SUCCESS_THRESHOLD
    )


    final_results.append({

        "algorithm": "PFA",

        "problem": PROBLEM,

        "dimension": N_DIM,

        "seed": seed,

        "best_fitness": pfa_best_f,

        "runtime_seconds": pfa_runtime,

        "cpu_time_seconds": pfa_cpu_time,

        "cpu_utilization_percent":
            pfa_cpu_utilization,

        "memory_baseline_mb":
            pfa_baseline_memory,

        "peak_memory_mb":
            pfa_peak_memory,

        "peak_memory_delta_mb":
            pfa_peak_delta_memory,

        "evaluations":
            pfa_total_evaluations,

        "success":
            int(pfa_success),
    })


    convergence_results.append(
        create_convergence_dataframe(
            "PFA",
            pfa_history,
            pfa_evaluation_history,
            seed
        )
    )


    # ========================================================
    # WOA
    # ========================================================

    print("Running WOA...")


    woa_iterations = (

        EVAL_BUDGET
        - WOA_POP_SIZE

    ) // WOA_POP_SIZE


    def run_woa_experiment():

        return run_woa(

            n_dim=N_DIM,

            pop_size=WOA_POP_SIZE,

            n_iterations=woa_iterations,

            bounds=BOUNDS,

            b=WOA_B,

            seed=seed,

            verbose=False,
        )


    (
        woa_result,
        woa_runtime,
        woa_cpu_time,
        woa_baseline_memory,
        woa_peak_memory,
        woa_peak_delta_memory,
        woa_cpu_utilization,

    ) = run_with_resources(
        run_woa_experiment
    )


    (
        woa_best_x,
        woa_best_f,
        woa_history,
        woa_mean_history,
        woa_position_history,

    ) = woa_result


    woa_evaluation_history = (
        run_woa.evaluation_history
    )

    woa_total_evaluations = (
        woa_evaluation_history[-1]
    )


    woa_success = (
        woa_best_f
        <= SUCCESS_THRESHOLD
    )


    final_results.append({

        "algorithm": "WOA",

        "problem": PROBLEM,

        "dimension": N_DIM,

        "seed": seed,

        "best_fitness": woa_best_f,

        "runtime_seconds": woa_runtime,

        "cpu_time_seconds": woa_cpu_time,

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

        "success":
            int(woa_success),
    })


    convergence_results.append(
        create_convergence_dataframe(
            "WOA",
            woa_history,
            woa_evaluation_history,
            seed
        )
    )


    # ========================================================
    # RESUMEN DE LA EJECUCIÓN
    # ========================================================

    print("\nResults:")

    for name, fitness, runtime, cpu, ram, evals in [

        (
            "GA",
            ga_best_f,
            ga_runtime,
            ga_cpu_time,
            ga_peak_delta_memory,
            ga_total_evaluations,
        ),

        (
            "cGA",
            cga_best_f,
            cga_runtime,
            cga_cpu_time,
            cga_peak_delta_memory,
            cga_total_evaluations,
        ),

        (
            "PSO",
            pso_best_f,
            pso_runtime,
            pso_cpu_time,
            pso_peak_delta_memory,
            pso_total_evaluations,
        ),

        (
            "ABC",
            abc_best_f,
            abc_runtime,
            abc_cpu_time,
            abc_peak_delta_memory,
            abc_total_evaluations,
        ),

        (
            "ACOR",
            acor_best_f,
            acor_runtime,
            acor_cpu_time,
            acor_peak_delta_memory,
            acor_total_evaluations,
        ),

        (
            "AFSA",
            afsa_best_f,
            afsa_runtime,
            afsa_cpu_time,
            afsa_peak_delta_memory,
            afsa_total_evaluations,
        ),

        (
            "AIS",
            ais_best_f,
            ais_runtime,
            ais_cpu_time,
            ais_peak_delta_memory,
            ais_total_evaluations,
        ),

        (
            "BFOA",
            bfoa_best_f,
            bfoa_runtime,
            bfoa_cpu_time,
            bfoa_peak_delta_memory,
            bfoa_total_evaluations,
        ),

        (
            "FA",
            fa_best_f,
            fa_runtime,
            fa_cpu_time,
            fa_peak_delta_memory,
            fa_total_evaluations,
        ),

        (
            "GWO",
            gwo_best_f,
            gwo_runtime,
            gwo_cpu_time,
            gwo_peak_delta_memory,
            gwo_total_evaluations,
        ),

        (
            "IWD",
            iwd_best_f,
            iwd_runtime,
            iwd_cpu_time,
            iwd_peak_delta_memory,
            iwd_total_evaluations,
        ),

        (
            "IWO",
            iwo_best_f,
            iwo_runtime,
            iwo_cpu_time,
            iwo_peak_delta_memory,
            iwo_total_evaluations,
        ),

        (
            "PFA",
            pfa_best_f,
            pfa_runtime,
            pfa_cpu_time,
            pfa_peak_delta_memory,
            pfa_total_evaluations,
        ),

        (
            "WOA",
            woa_best_f,
            woa_runtime,
            woa_cpu_time,
            woa_peak_delta_memory,
            woa_total_evaluations,
        ),
    ]:

        print(
            f"{name:<5} | "
            f"fitness={fitness:.6e} | "
            f"time={runtime:.4f}s | "
            f"CPU={cpu:.4f}s | "
            f"RAM+={ram:.2f}MB | "
            f"evals={evals}"
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
# GUARDAR RESULTADOS
# ============================================================

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
        ),
    )

    .reset_index()
)


summary_df[
    "success_rate_percent"
] = (

    summary_df["success_rate"]
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

print("\n")

print("=" * 80)

print("BENCHMARK FINISHED")

print("=" * 80)


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

