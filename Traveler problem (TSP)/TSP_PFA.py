"""
Paddy Field Algorithm (PFA) applied to the TSP.

Same random-key encoding rationale as fa_tsp.py -- seeding and
pollination happen on continuous key vectors, decoded to tours via
argsort. See pfa_sphere.py for the full phase-by-phase explanation of
the algorithm itself (selective seeding, proportional dispersal,
pollination, population control) and the caveat about PFA's lack of
one single standardized formula set.
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


def run_pfa(
    dist_matrix,
    n_cities,
    pop_size=30,
    n_iterations=500,
    selection_rate=0.3,
    max_seeds=5,
    sigma_initial=0.3,
    sigma_final=0.001,
    exponent=2,
    pollination_rate=0.1,
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

    n_select = max(1, int(selection_rate * pop_size))
    n_pollinate = max(1, int(pollination_rate * pop_size))

    for it in range(n_iterations):
        sigma_t = ((n_iterations - it) / n_iterations) ** exponent * \
            (sigma_initial - sigma_final) + sigma_final

        order = np.argsort(fitness)
        selected = order[:n_select]

        f_min, f_max = fitness[selected].min(), fitness[selected].max()
        seeds = []
        for rank, idx in enumerate(selected):
            if f_max == f_min:
                n_seeds = max_seeds
            else:
                ratio = (f_max - fitness[idx]) / (f_max - f_min)
                n_seeds = max(1, int(round(ratio * max_seeds)))
            for _ in range(n_seeds):
                s = population[idx] + rng.normal(0, sigma_t, size=n_cities)
                seeds.append(np.clip(s, lo, hi))
        seeds = np.array(seeds) if seeds else np.empty((0, n_cities))
        seed_fitness = np.array([fitness_of_keys(s, dist_matrix) for s in seeds]) if len(seeds) else np.empty(0)

        worst_idx = order[-n_pollinate:]
        for idx in worst_idx:
            donor = population[selected[rng.randint(len(selected))]]
            random_component = rng.uniform(lo, hi, size=n_cities)
            pollinated = 0.5 * donor + 0.5 * random_component
            population[idx] = np.clip(pollinated, lo, hi)
            fitness[idx] = fitness_of_keys(population[idx], dist_matrix)

        combined = np.vstack([population, seeds]) if len(seeds) else population
        combined_fitness = np.concatenate([fitness, seed_fitness]) if len(seeds) else fitness
        keep_order = np.argsort(combined_fitness)[:pop_size]
        population = combined[keep_order]
        fitness = combined_fitness[keep_order]

        best_idx = np.argmin(fitness)
        best_history.append(fitness[best_idx])
        mean_history.append(np.mean(fitness))
        best_position_history.append(decode_keys(population[best_idx]))
        evaluation_history.append(evaluation_count)

        if verbose and (it % 50 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best tour length: {best_history[-1]:.3f}")

    best_idx = np.argmin(fitness)
    best_tour = decode_keys(population[best_idx])
    run_pfa.evaluation_history = evaluation_history
    return best_tour, fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    N_CITIES = 25
    cities = generate_cities(n_cities=N_CITIES, seed=42)
    dmat = distance_matrix(cities)

    best_tour, best_len, best_hist, mean_hist, pos_hist = run_pfa(
        dmat, N_CITIES, pop_size=30, n_iterations=500, seed=42,
    )

    print("\n=== Result (PFA on TSP) ===")
    print("Best tour:", best_tour)
    print("Best tour length:", best_len)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(best_hist, label="Best")
    axes[0].plot(mean_hist, label="Mean", alpha=0.7)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Tour length")
    axes[0].set_title("PFA Convergence on TSP")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    tour_coords = cities[np.append(best_tour, best_tour[0])]
    axes[1].plot(tour_coords[:, 0], tour_coords[:, 1], "o-")
    for i, (x, y) in enumerate(cities):
        axes[1].annotate(str(i), (x, y), fontsize=8)
    axes[1].set_title(f"Best Tour Found (length={best_len:.2f})")

    plt.tight_layout()
    plt.show()