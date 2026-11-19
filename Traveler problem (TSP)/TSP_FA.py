"""
Firefly Algorithm (FA) applied to the TSP.

FA operates on continuous vectors -- fireflies move toward brighter
fireflies by blending positions, which only makes sense where
"in-between" points are meaningful. Like GWO/WOA/ABC before it, we use
RANDOM-KEY ENCODING: each firefly's position is a vector of n_cities
keys in [0, 1], and the tour is recovered via argsort. FA's attraction
movement happens entirely in this continuous key-space; only fitness
evaluation needs the decoded tour.

See fa_sphere.py for the full phase-by-phase explanation of the
algorithm itself (pairwise attraction, distance-based decay, gamma
scaling) -- unchanged here except for the key-based encode/decode step.
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


def run_fa(
    dist_matrix,
    n_cities,
    pop_size=30,
    n_iterations=500,
    alpha=0.5,
    beta0=1.0,
    gamma=1.0,     # key-space is already in [0,1], so no rescaling needed here (unlike sphere's gamma=0.01)
    alpha_decay=0.97,
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

    alpha_t = alpha
    for it in range(n_iterations):
        alpha_t *= alpha_decay

        for i in range(pop_size):
            for j in range(pop_size):
                if fitness[j] < fitness[i]:
                    r = np.linalg.norm(population[i] - population[j])
                    beta = beta0 * np.exp(-gamma * r ** 2)
                    eps = alpha_t * (rng.rand(n_cities) - 0.5)
                    population[i] = population[i] + beta * (population[j] - population[i]) + eps
                    population[i] = np.clip(population[i], lo, hi)
                    fitness[i] = fitness_of_keys(population[i], dist_matrix)

        best_idx = np.argmin(fitness)
        best_history.append(fitness[best_idx])
        mean_history.append(np.mean(fitness))
        best_position_history.append(decode_keys(population[best_idx]))
        evaluation_history.append(evaluation_count)

        if verbose and (it % 50 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best tour length: {best_history[-1]:.3f}")

    best_idx = np.argmin(fitness)
    best_tour = decode_keys(population[best_idx])
    run_fa.evaluation_history = evaluation_history
    return best_tour, fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    N_CITIES = 25
    cities = generate_cities(n_cities=N_CITIES, seed=42)
    dmat = distance_matrix(cities)

    best_tour, best_len, best_hist, mean_hist, pos_hist = run_fa(
        dmat, N_CITIES, pop_size=30, n_iterations=500, seed=42,
    )

    print("\n=== Result (FA on TSP) ===")
    print("Best tour:", best_tour)
    print("Best tour length:", best_len)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(best_hist, label="Best")
    axes[0].plot(mean_hist, label="Mean", alpha=0.7)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Tour length")
    axes[0].set_title("FA Convergence on TSP")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    tour_coords = cities[np.append(best_tour, best_tour[0])]
    axes[1].plot(tour_coords[:, 0], tour_coords[:, 1], "o-")
    for i, (x, y) in enumerate(cities):
        axes[1].annotate(str(i), (x, y), fontsize=8)
    axes[1].set_title(f"Best Tour Found (length={best_len:.2f})")

    plt.tight_layout()
    plt.show()