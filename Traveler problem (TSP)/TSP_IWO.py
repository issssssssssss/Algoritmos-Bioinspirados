"""
Invasive Weed Optimization (IWO) applied to the TSP.

Same random-key encoding rationale as fa_tsp.py -- seed dispersal
happens on continuous key vectors, decoded to tours via argsort. See
iwo_sphere.py for the full phase-by-phase explanation of the algorithm
itself (proportional seeding, shrinking Gaussian dispersal, competitive
exclusion).
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


def run_iwo(
    dist_matrix,
    n_cities,
    pop_size=30,
    n_iterations=500,
    min_seeds=0,
    max_seeds=5,
    exponent=2,
    sigma_initial=0.3,
    sigma_final=0.001,
    seed=None,
    verbose=True,
):
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)
    lo, hi = 0.0, 1.0
    pop_max = pop_size

    population = rng.uniform(lo, hi, size=(pop_size, n_cities))
    fitness = np.array([fitness_of_keys(ind, dist_matrix) for ind in population])

    best_history = []
    mean_history = []
    best_position_history = []

    for it in range(n_iterations):
        sigma_t = ((n_iterations - it) / n_iterations) ** exponent * \
            (sigma_initial - sigma_final) + sigma_final

        f_min, f_max = fitness.min(), fitness.max()
        all_seeds = []
        for i in range(len(population)):
            if f_max == f_min:
                n_seeds = max_seeds
            else:
                ratio = (f_max - fitness[i]) / (f_max - f_min)
                n_seeds = int(round(min_seeds + ratio * (max_seeds - min_seeds)))
            for _ in range(n_seeds):
                s = population[i] + rng.normal(0, sigma_t, size=n_cities)
                all_seeds.append(np.clip(s, lo, hi))

        if all_seeds:
            all_seeds = np.array(all_seeds)
            seed_fitness = np.array([fitness_of_keys(s, dist_matrix) for s in all_seeds])
            combined = np.vstack([population, all_seeds])
            combined_fitness = np.concatenate([fitness, seed_fitness])
        else:
            combined, combined_fitness = population, fitness

        order = np.argsort(combined_fitness)[:pop_max]
        population = combined[order]
        fitness = combined_fitness[order]

        best_idx = np.argmin(fitness)
        best_history.append(fitness[best_idx])
        mean_history.append(np.mean(fitness))
        best_position_history.append(decode_keys(population[best_idx]))
        evaluation_history.append(evaluation_count)

        if verbose and (it % 50 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best tour length: {best_history[-1]:.3f}")

    best_idx = np.argmin(fitness)
    best_tour = decode_keys(population[best_idx])
    run_iwo.evaluation_history = evaluation_history
    return best_tour, fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    N_CITIES = 25
    cities = generate_cities(n_cities=N_CITIES, seed=42)
    dmat = distance_matrix(cities)

    best_tour, best_len, best_hist, mean_hist, pos_hist = run_iwo(
        dmat, N_CITIES, pop_size=30, n_iterations=500, seed=42,
    )

    print("\n=== Result (IWO on TSP) ===")
    print("Best tour:", best_tour)
    print("Best tour length:", best_len)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(best_hist, label="Best")
    axes[0].plot(mean_hist, label="Mean", alpha=0.7)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Tour length")
    axes[0].set_title("IWO Convergence on TSP")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    tour_coords = cities[np.append(best_tour, best_tour[0])]
    axes[1].plot(tour_coords[:, 0], tour_coords[:, 1], "o-")
    for i, (x, y) in enumerate(cities):
        axes[1].annotate(str(i), (x, y), fontsize=8)
    axes[1].set_title(f"Best Tour Found (length={best_len:.2f})")

    plt.tight_layout()
    plt.show()