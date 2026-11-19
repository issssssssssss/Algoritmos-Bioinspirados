"""
Grey Wolf Optimizer (GWO) applied to the TSP.

GWO is built for continuous vectors -- alpha/beta/delta wolves pull
every other wolf toward a blended position, which only makes sense in a
continuous space where "in-between" positions are meaningful. A tour
permutation has no such notion (there's no valid "position halfway
between tour A and tour B"), so GWO can't operate on permutations
directly.

Like cGA and PSO before, we use RANDOM-KEY ENCODING: each wolf's
position is a continuous vector of n_cities keys in [0, 1]. The tour is
recovered by sorting city indices by their keys (argsort). GWO's
encircling/blending update happens entirely in this continuous
key-space; only fitness evaluation needs the decoded tour.

See gwo_sphere.py for the full phase-by-phase explanation of the
algorithm itself (social hierarchy, encircling, exploration/exploitation
decay of `a`) -- unchanged here except for the key-based
encode/decode step.
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


def run_gwo(
    dist_matrix,
    n_cities,
    pop_size=30,
    n_iterations=500,
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

    for it in range(n_iterations):
        order = np.argsort(fitness)
        alpha, beta, delta = population[order[0]], population[order[1]], population[order[2]]

        a = 2 - it * (2 / n_iterations)

        new_population = np.empty_like(population)
        for i in range(pop_size):
            X = population[i]
            X_updates = []
            for leader in (alpha, beta, delta):
                r1, r2 = rng.rand(n_cities), rng.rand(n_cities)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * leader - X)
                X_updates.append(leader - A * D)
            new_population[i] = np.mean(X_updates, axis=0)

        population = np.clip(new_population, lo, hi)
        fitness = np.array([fitness_of_keys(ind, dist_matrix) for ind in population])

        best_idx = np.argmin(fitness)
        best_history.append(fitness[best_idx])
        mean_history.append(np.mean(fitness))
        best_position_history.append(decode_keys(population[best_idx]))
        evaluation_history.append(evaluation_count)

        if verbose and (it % 50 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best tour length: {best_history[-1]:.3f}")

    best_idx = np.argmin(fitness)
    best_tour = decode_keys(population[best_idx])
    run_gwo.evaluation_history = evaluation_history
    return best_tour, fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    N_CITIES = 25
    cities = generate_cities(n_cities=N_CITIES, seed=42)
    dmat = distance_matrix(cities)

    best_tour, best_len, best_hist, mean_hist, pos_hist = run_gwo(
        dmat, N_CITIES, pop_size=30, n_iterations=500, seed=42,
    )

    print("\n=== Result (GWO on TSP) ===")
    print("Best tour:", best_tour)
    print("Best tour length:", best_len)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(best_hist, label="Best")
    axes[0].plot(mean_hist, label="Mean", alpha=0.7)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Tour length")
    axes[0].set_title("GWO Convergence on TSP")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    tour_coords = cities[np.append(best_tour, best_tour[0])]
    axes[1].plot(tour_coords[:, 0], tour_coords[:, 1], "o-")
    for i, (x, y) in enumerate(cities):
        axes[1].annotate(str(i), (x, y), fontsize=8)
    axes[1].set_title(f"Best Tour Found (length={best_len:.2f})")

    plt.tight_layout()
    plt.show()