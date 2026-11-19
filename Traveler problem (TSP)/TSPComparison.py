import numpy as np
import matplotlib.pyplot as plt

from TSP_Problem import generate_cities, distance_matrix, tour_length
from TSP_GA import run_ga
from TSP_cGA import run_cga
from TSP_PSO import run_pso


N_CITIES = 25
EVAL_BUDGET = 60000
SEED = 42

cities = generate_cities(n_cities=N_CITIES, seed=SEED)
dmat = distance_matrix(cities)

ga_pop = 150
ga_generations = EVAL_BUDGET // ga_pop
ga_best_tour, ga_best_len, ga_hist = run_ga(
    dmat, N_CITIES, pop_size=ga_pop, n_generations=ga_generations,
    seed=SEED, verbose=False,
)
ga_evals = np.arange(1, len(ga_hist) + 1) * ga_pop

cga_generations = EVAL_BUDGET // 2
cga_best_tour, cga_best_len, cga_hist = run_cga(
    dmat, N_CITIES, bits_per_key=16, pop_size=100,
    n_generations=cga_generations, seed=SEED, verbose=False,
)
cga_evals = np.arange(1, len(cga_hist) + 1) * 2

pso_particles = 80
pso_iterations = EVAL_BUDGET // pso_particles
pso_best_tour, pso_best_len, pso_hist = run_pso(
    dmat, N_CITIES, n_particles=pso_particles, n_iterations=pso_iterations,
    seed=SEED, verbose=False,
)
pso_evals = np.arange(1, len(pso_hist) + 1) * pso_particles

print(f"{'Algorithm':<10} {'Evaluations':<14} {'Best tour length':<18}")
print(f"{'GA':<10} {ga_evals[-1]:<14} {ga_best_len:<18.3f}")
print(f"{'cGA':<10} {cga_evals[-1]:<14} {cga_best_len:<18.3f}")
print(f"{'PSO':<10} {pso_evals[-1]:<14} {pso_best_len:<18.3f}")

plt.figure(figsize=(9, 6))
plt.plot(ga_evals, ga_hist, label="GA (permutation encoding)")
plt.plot(cga_evals, cga_hist, label="cGA (binary random-key)")
plt.plot(pso_evals, pso_hist, label="PSO (continuous random-key)")
plt.xlabel("Fitness evaluations")
plt.ylabel("Best tour length found")
plt.title(f"GA vs cGA vs PSO on TSP ({N_CITIES} cities)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# COMPARACION COMPLETA DE LOS 14 ALGORITMOS
# ============================================================

from TSP_FA import run_fa
from TSP_AFSA import run_afsa
from TSP_AIS import run_ais
from TSP_BFOA import run_bfoa
from TSP_IWO import run_iwo
from TSP_PFA import run_pfa
from TSP_IWD import run_iwd
from TSP_ABC import run_abc
from TSP_ACO import run_aco
from TSP_GWO import run_gwo
from TSP_WOA import run_woa


all_algorithms = [
    ("GA", run_ga, {"pop_size": 150, "n_generations": (EVAL_BUDGET - 150) // 150}),
    ("cGA", run_cga, {"bits_per_key": 16, "pop_size": 100, "n_generations": EVAL_BUDGET // 2}),
    ("PSO", run_pso, {"n_particles": 80, "n_iterations": (EVAL_BUDGET - 80) // 80}),
    ("ABC", run_abc, {"pop_size": 30, "n_iterations": (EVAL_BUDGET - 30) // 60}),
    ("ACO", run_aco, {"n_ants": 30, "n_iterations": EVAL_BUDGET // 30}),
    ("GWO", run_gwo, {"pop_size": 30, "n_iterations": (EVAL_BUDGET - 30) // 30}),
    ("WOA", run_woa, {"pop_size": 30, "n_iterations": (EVAL_BUDGET - 30) // 30}),
    ("FA", run_fa, {"pop_size": 30, "n_iterations": (EVAL_BUDGET - 30) // 30}),
    ("AFSA", run_afsa, {"pop_size": 30, "n_iterations": (EVAL_BUDGET - 30) // 30}),
    ("AIS", run_ais, {"pop_size": 30, "n_iterations": (EVAL_BUDGET - 30) // 30}),
    ("BFOA", run_bfoa, {"pop_size": 30, "n_iterations": (EVAL_BUDGET - 30) // 30}),
    ("IWO", run_iwo, {"pop_size": 30, "n_iterations": (EVAL_BUDGET - 30) // 30}),
    ("PFA", run_pfa, {"pop_size": 30, "n_iterations": (EVAL_BUDGET - 30) // 30}),
    ("IWD", run_iwd, {"n_drops": 30, "n_iterations": (EVAL_BUDGET - 30) // 30}),
]

comparison_results = {}

for name, algorithm, parameters in all_algorithms:
    result = algorithm(
        dmat,
        N_CITIES,
        **parameters,
        seed=SEED,
        verbose=False,
    )
    comparison_results[name] = (
        result[0],
        result[1],
        result[2],
        algorithm.evaluation_history,
    )

print("\nComplete 14-algorithm comparison")
print(f"{'Algorithm':<10} {'Evaluations':<14} {'Best tour length':<18}")
for name, (_, length, _, evaluations) in comparison_results.items():
    print(f"{name:<10} {evaluations[-1]:<14} {length:<18.3f}")

plt.figure(figsize=(12, 7))
for name, (_, length, history, evaluations) in comparison_results.items():
    plt.plot(evaluations, history, label=name)
plt.xlabel("Fitness evaluations")
plt.ylabel("Best tour length found")
plt.title(f"14 Algorithms on TSP ({N_CITIES} cities)")
plt.legend(ncol=2, fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()



fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, tour, length, name in [
    (axes[0], ga_best_tour, ga_best_len, "GA"),
    (axes[1], cga_best_tour, cga_best_len, "cGA"),
    (axes[2], pso_best_tour, pso_best_len, "PSO"),
]:
    tour_coords = cities[np.append(tour, tour[0])]
    ax.plot(tour_coords[:, 0], tour_coords[:, 1], "o-")
    ax.set_title(f"{name} (length={length:.1f})")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, tour, length, name in [
    (axes[0], ga_best_tour, ga_best_len, "GA"),
    (axes[1], cga_best_tour, cga_best_len, "cGA"),
    (axes[2], pso_best_tour, pso_best_len, "PSO"),
]:

    closed_tour = np.append(tour, tour[0])
    tour_coords = cities[closed_tour]

    ax.plot(
        tour_coords[:, 0],
        tour_coords[:, 1],
        "o-",
        linewidth=1.5
    )

    for i, city_idx in enumerate(tour):
        x, y = cities[city_idx]

        ax.annotate(
            str(city_idx),
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )

    for i in range(len(tour)):

        city_a = tour[i]
        city_b = tour[(i + 1) % len(tour)]

        x1, y1 = cities[city_a]
        x2, y2 = cities[city_b]

        distance = dmat[city_a, city_b]

        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        ax.annotate(
            f"{distance:.1f}",
            (mid_x, mid_y),
            fontsize=7,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                alpha=0.7
            )
        )

    ax.set_title(
        f"{name}\nLongitud total = {length:.1f}"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.grid(True, alpha=0.3)


plt.tight_layout()
plt.show()



"""
Compare GWO (random-key), WOA (random-key), ABC (random-key), and
classic ACO (native edge-pheromone) on the same TSP instance, using a
roughly equal number of fitness evaluations for a fair comparison.
"""

import numpy as np
import matplotlib.pyplot as plt

from TSP_GWO import run_gwo
from TSP_WOA import run_woa
from TSP_ABC import run_abc
from TSP_ACO import run_aco


N_CITIES = 25
EVAL_BUDGET = 60000
SEED = 42

cities = generate_cities(n_cities=N_CITIES, seed=SEED)
dmat = distance_matrix(cities)

pop = 30

# GWO: pop_size evaluations per iteration
gwo_iters = EVAL_BUDGET // pop
gwo_tour, gwo_len, gwo_hist, _, _ = run_gwo(dmat, N_CITIES, pop_size=pop, n_iterations=gwo_iters, seed=SEED, verbose=False)
gwo_evals = np.arange(1, len(gwo_hist) + 1) * pop

# WOA: pop_size evaluations per iteration
woa_iters = EVAL_BUDGET // pop
woa_tour, woa_len, woa_hist, _, _ = run_woa(dmat, N_CITIES, pop_size=pop, n_iterations=woa_iters, seed=SEED, verbose=False)
woa_evals = np.arange(1, len(woa_hist) + 1) * pop

# ABC: employed + onlooker = 2*pop_size evaluations per iteration
abc_iters = EVAL_BUDGET // (2 * pop)
abc_tour, abc_len, abc_hist, _, _ = run_abc(dmat, N_CITIES, pop_size=pop, n_iterations=abc_iters, seed=SEED, verbose=False)
abc_evals = np.arange(1, len(abc_hist) + 1) * 2 * pop

# ACO: n_ants evaluations per iteration
aco_iters = EVAL_BUDGET // pop
aco_tour, aco_len, aco_hist, _, _ = run_aco(dmat, N_CITIES, n_ants=pop, n_iterations=aco_iters, seed=SEED, verbose=False)
aco_evals = np.arange(1, len(aco_hist) + 1) * pop

print(f"{'Algorithm':<10} {'Evaluations':<14} {'Best tour length':<18}")
print(f"{'GWO':<10} {gwo_evals[-1]:<14} {gwo_len:<18.3f}")
print(f"{'WOA':<10} {woa_evals[-1]:<14} {woa_len:<18.3f}")
print(f"{'ABC':<10} {abc_evals[-1]:<14} {abc_len:<18.3f}")
print(f"{'ACO':<10} {aco_evals[-1]:<14} {aco_len:<18.3f}")

# ---- Convergence comparison ----
plt.figure(figsize=(9, 6))
plt.plot(gwo_evals, gwo_hist, label="GWO (random-key)")
plt.plot(woa_evals, woa_hist, label="WOA (random-key)")
plt.plot(abc_evals, abc_hist, label="ABC (random-key)")
plt.plot(aco_evals, aco_hist, label="ACO (native pheromone)")
plt.xlabel("Fitness evaluations")
plt.ylabel("Best tour length found")
plt.title(f"GWO vs WOA vs ABC vs ACO on TSP ({N_CITIES} cities)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ---- Side-by-side best tours ----
fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
for ax, tour, length, name in [
    (axes[0], gwo_tour, gwo_len, "GWO"),
    (axes[1], woa_tour, woa_len, "WOA"),
    (axes[2], abc_tour, abc_len, "ABC"),
    (axes[3], aco_tour, aco_len, "ACO"),
]:
    tour_coords = cities[np.append(tour, tour[0])]
    ax.plot(tour_coords[:, 0], tour_coords[:, 1], "o-")
    ax.set_title(f"{name} (length={length:.1f})")
plt.tight_layout()
plt.show()