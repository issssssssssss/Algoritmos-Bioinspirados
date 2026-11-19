"""
Artificial Bee Colony (ABC) applied to the TSP.

Same random-key encoding rationale as gwo_tsp.py -- ABC's neighborhood
search operates on continuous key vectors, decoded to tours via argsort.
See abc_sphere.py for the full phase-by-phase explanation of the
algorithm itself (employed bees, onlooker bees, scout bees).
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


def run_abc(
    dist_matrix,
    n_cities,
    pop_size=30,
    n_iterations=500,
    limit=50,
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
    trial = np.zeros(pop_size, dtype=int)

    def fitness_transform(f):
        return 1.0 / (1.0 + f)  # tour length is always >= 0

    def neighbor_search(i):
        j = rng.randint(n_cities)
        k = rng.choice([idx for idx in range(pop_size) if idx != i])
        phi = rng.uniform(-1, 1)
        v = population[i].copy()
        v[j] = population[i][j] + phi * (population[i][j] - population[k][j])
        v = np.clip(v, lo, hi)
        return v

    best_history = []
    mean_history = []
    best_position_history = []
    best_ever_x = population[np.argmin(fitness)].copy()
    best_ever_f = fitness.min()

    for it in range(n_iterations):
        for i in range(pop_size):
            v = neighbor_search(i)
            fv = fitness_of_keys(v, dist_matrix)
            if fv < fitness[i]:
                population[i], fitness[i] = v, fv
                trial[i] = 0
            else:
                trial[i] += 1

        trans_fit = fitness_transform(fitness)
        probs = trans_fit / trans_fit.sum()
        for _ in range(pop_size):
            i = rng.choice(pop_size, p=probs)
            v = neighbor_search(i)
            fv = fitness_of_keys(v, dist_matrix)
            if fv < fitness[i]:
                population[i], fitness[i] = v, fv
                trial[i] = 0
            else:
                trial[i] += 1

        for i in range(pop_size):
            if trial[i] > limit:
                population[i] = rng.uniform(lo, hi, size=n_cities)
                fitness[i] = fitness_of_keys(population[i], dist_matrix)
                trial[i] = 0

        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_ever_f:
            best_ever_f = fitness[gen_best_idx]
            best_ever_x = population[gen_best_idx].copy()

        best_history.append(best_ever_f)
        mean_history.append(np.mean(fitness))
        best_position_history.append(decode_keys(best_ever_x))
        evaluation_history.append(evaluation_count)

        if verbose and (it % 50 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best-ever tour length: {best_ever_f:.3f}")

    best_tour = decode_keys(best_ever_x)
    run_abc.evaluation_history = evaluation_history
    return best_tour, best_ever_f, best_history, mean_history, best_position_history


if __name__ == "__main__":
    N_CITIES = 25
    cities = generate_cities(n_cities=N_CITIES, seed=42)
    dmat = distance_matrix(cities)

    best_tour, best_len, best_hist, mean_hist, pos_hist = run_abc(
        dmat, N_CITIES, pop_size=30, n_iterations=500, seed=42,
    )

    print("\n=== Result (ABC on TSP) ===")
    print("Best tour:", best_tour)
    print("Best tour length:", best_len)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(best_hist, label="Best-ever")
    axes[0].plot(mean_hist, label="Mean", alpha=0.7)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Tour length")
    axes[0].set_title("ABC Convergence on TSP")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    tour_coords = cities[np.append(best_tour, best_tour[0])]
    axes[1].plot(tour_coords[:, 0], tour_coords[:, 1], "o-")
    for i, (x, y) in enumerate(cities):
        axes[1].annotate(str(i), (x, y), fontsize=8)
    axes[1].set_title(f"Best Tour Found (length={best_len:.2f})")

    plt.tight_layout()
    plt.show()