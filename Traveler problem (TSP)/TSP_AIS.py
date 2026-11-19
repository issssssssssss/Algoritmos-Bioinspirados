"""
Artificial Immune System (AIS / CLONALG) applied to the TSP.

Same random-key encoding rationale as fa_tsp.py -- cloning and
hypermutation happen on continuous key vectors, decoded to tours via
argsort. See ais_sphere.py for the full phase-by-phase explanation of
the algorithm itself (selection, proportional cloning, affinity
maturation, receptor editing).
"""

import numpy as np
import matplotlib.pyplot as plt

from TSP_Problem import generate_cities, distance_matrix, tour_length


evaluation_count = 0


def decode_keys(keys):
    return np.argsort(keys)


def fitness_of_keys(keys, dist_matrix):
    global evaluation_count
    evaluation_count += 1
    return tour_length(decode_keys(keys), dist_matrix)


def run_ais(
    dist_matrix,
    n_cities,
    pop_size=30,
    n_iterations=500,
    n_select=10,
    clones_per_antibody=5,
    beta=2.0,
    replace_rate=0.2,
    seed=None,
    verbose=True,
):
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)
    lo, hi = 0.0, 1.0

    population = rng.uniform(lo, hi, size=(pop_size, n_cities))
    fitness = np.array([fitness_of_keys(ind, dist_matrix) for ind in population])

    best_history = []
    mean_history = []
    best_position_history = []
    n_replace = max(1, int(replace_rate * pop_size))

    for it in range(n_iterations):
        order = np.argsort(fitness)
        selected = order[:n_select]

        for rank, idx in enumerate(selected):
            affinity_normalized = rank / (n_select - 1) if n_select > 1 else 0
            mutation_std = 0.01 + 0.5 * affinity_normalized  # key-space is [0,1], no rescaling needed

            best_clone, best_clone_fit = population[idx].copy(), fitness[idx]
            for _ in range(clones_per_antibody):
                clone = population[idx] + rng.normal(0, mutation_std, size=n_cities)
                clone = np.clip(clone, lo, hi)
                clone_fit = fitness_of_keys(clone, dist_matrix)
                if clone_fit < best_clone_fit:
                    best_clone, best_clone_fit = clone, clone_fit

            if best_clone_fit < fitness[idx]:
                population[idx] = best_clone
                fitness[idx] = best_clone_fit

        worst_idx = np.argsort(fitness)[-n_replace:]
        for idx in worst_idx:
            population[idx] = rng.uniform(lo, hi, size=n_cities)
            fitness[idx] = fitness_of_keys(population[idx], dist_matrix)

        best_idx = np.argmin(fitness)
        best_history.append(fitness[best_idx])
        mean_history.append(np.mean(fitness))
        best_position_history.append(decode_keys(population[best_idx]))
        evaluation_history.append(evaluation_count)

        if verbose and (it % 50 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best tour length: {best_history[-1]:.3f}")

    best_idx = np.argmin(fitness)
    best_tour = decode_keys(population[best_idx])
    run_ais.evaluation_history = evaluation_history
    return best_tour, fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    N_CITIES = 25
    cities = generate_cities(n_cities=N_CITIES, seed=42)
    dmat = distance_matrix(cities)

    best_tour, best_len, best_hist, mean_hist, pos_hist = run_ais(
        dmat, N_CITIES, pop_size=30, n_iterations=500, seed=42,
    )

    print("\n=== Result (AIS on TSP) ===")
    print("Best tour:", best_tour)
    print("Best tour length:", best_len)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(best_hist, label="Best")
    axes[0].plot(mean_hist, label="Mean", alpha=0.7)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Tour length")
    axes[0].set_title("AIS Convergence on TSP")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    tour_coords = cities[np.append(best_tour, best_tour[0])]
    axes[1].plot(tour_coords[:, 0], tour_coords[:, 1], "o-")
    for i, (x, y) in enumerate(cities):
        axes[1].annotate(str(i), (x, y), fontsize=8)
    axes[1].set_title(f"Best Tour Found (length={best_len:.2f})")

    plt.tight_layout()
    plt.show()