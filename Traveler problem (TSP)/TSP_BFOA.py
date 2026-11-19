"""
Bacterial Foraging Optimization Algorithm (BFOA) applied to the TSP.

Same random-key encoding rationale as fa_tsp.py -- chemotaxis (tumble +
swim) happens on continuous key vectors, decoded to tours via argsort.
See bfoa_sphere.py for the full phase-by-phase explanation of the
algorithm itself (chemotaxis, swarming penalty, reproduction,
elimination-dispersal).
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


def compute_Jcc(X, population, d_attract=0.05, w_attract=2, h_repellant=0.05, w_repellant=50):
    diffs = population - X
    sq_dists = np.sum(diffs ** 2, axis=1)
    attract = -d_attract * np.exp(-w_attract * sq_dists)
    repel = h_repellant * np.exp(-w_repellant * sq_dists)
    return np.sum(attract + repel)


def run_bfoa(
    dist_matrix,
    n_cities,
    pop_size=30,
    n_iterations=500,   # total chemotaxis steps
    Nc=10,
    Ns=4,
    Nre=4,
    Ped=0.25,
    C=0.05,              # tumble/run step size in key-space [0,1]
    seed=None,
    verbose=True,
):
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)
    lo, hi = 0.0, 1.0
    step_size = C

    population = rng.uniform(lo, hi, size=(pop_size, n_cities))
    fitness = np.array([fitness_of_keys(ind, dist_matrix) for ind in population])
    health = np.zeros(pop_size)

    best_history = []
    mean_history = []
    best_position_history = []
    reproduction_count = 0

    for it in range(n_iterations):
        for i in range(pop_size):
            delta = rng.uniform(-1, 1, size=n_cities)
            delta = delta / (np.linalg.norm(delta) + 1e-12)

            current_cost = fitness[i] + compute_Jcc(population[i], population)
            new_pos = np.clip(population[i] + step_size * delta, lo, hi)
            new_cost = fitness_of_keys(new_pos, dist_matrix) + compute_Jcc(new_pos, population)

            if new_cost < current_cost:
                population[i], fitness[i] = new_pos, fitness_of_keys(new_pos, dist_matrix)
                for _ in range(Ns):
                    trial = np.clip(population[i] + step_size * delta, lo, hi)
                    trial_cost = fitness_of_keys(trial, dist_matrix) + compute_Jcc(trial, population)
                    if trial_cost < fitness[i] + compute_Jcc(population[i], population):
                        population[i], fitness[i] = trial, fitness_of_keys(trial, dist_matrix)
                    else:
                        break

            health[i] += fitness[i]

        best_idx = np.argmin(fitness)
        best_history.append(fitness[best_idx])
        mean_history.append(np.mean(fitness))
        best_position_history.append(decode_keys(population[best_idx]))
        evaluation_history.append(evaluation_count)

        if verbose and (it % 50 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best tour length: {best_history[-1]:.3f}")

        if (it + 1) % Nc == 0:
            order = np.argsort(health)
            n_keep = pop_size // 2
            survivors = order[:n_keep]
            weak = order[n_keep:]
            population[weak] = population[survivors]
            fitness[weak] = fitness[survivors]
            health[:] = 0
            reproduction_count += 1

            if reproduction_count % Nre == 0:
                for i in range(pop_size):
                    if rng.rand() < Ped:
                        population[i] = rng.uniform(lo, hi, size=n_cities)
                        fitness[i] = fitness_of_keys(population[i], dist_matrix)

    best_idx = np.argmin(fitness)
    best_tour = decode_keys(population[best_idx])
    run_bfoa.evaluation_history = evaluation_history
    return best_tour, fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    N_CITIES = 25
    cities = generate_cities(n_cities=N_CITIES, seed=42)
    dmat = distance_matrix(cities)

    best_tour, best_len, best_hist, mean_hist, pos_hist = run_bfoa(
        dmat, N_CITIES, pop_size=30, n_iterations=500, seed=42,
    )

    print("\n=== Result (BFOA on TSP) ===")
    print("Best tour:", best_tour)
    print("Best tour length:", best_len)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(best_hist, label="Best")
    axes[0].plot(mean_hist, label="Mean", alpha=0.7)
    axes[0].set_xlabel("Chemotaxis step")
    axes[0].set_ylabel("Tour length")
    axes[0].set_title("BFOA Convergence on TSP")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    tour_coords = cities[np.append(best_tour, best_tour[0])]
    axes[1].plot(tour_coords[:, 0], tour_coords[:, 1], "o-")
    for i, (x, y) in enumerate(cities):
        axes[1].annotate(str(i), (x, y), fontsize=8)
    axes[1].set_title(f"Best Tour Found (length={best_len:.2f})")

    plt.tight_layout()
    plt.show()