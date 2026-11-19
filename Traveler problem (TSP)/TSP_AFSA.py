"""
Artificial Fish Swarm Algorithm (AFSA) applied to the TSP.

Same random-key encoding rationale as fa_tsp.py -- AFSA's swarming/
following/preying behaviors operate on continuous key vectors, decoded
to tours via argsort. See afsa_sphere.py for the full phase-by-phase
explanation of the algorithm itself.
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


def run_afsa(
    dist_matrix,
    n_cities,
    pop_size=30,
    n_iterations=500,
    visual=0.3,
    step=0.1,
    try_number=5,
    crowding_factor=0.6,
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
        progress = it / n_iterations
        visual_t = visual * (1 - 0.9 * progress)
        step_t = step * (1 - 0.9 * progress)

        new_population = population.copy()

        for i in range(pop_size):
            Xi = population[i]
            dists = np.linalg.norm(population - Xi, axis=1)
            neighbor_idx = np.where((dists > 0) & (dists <= visual_t))[0]

            candidate_moves = []

            if len(neighbor_idx) > 0 and len(neighbor_idx) / pop_size < crowding_factor:
                center = population[neighbor_idx].mean(axis=0)
                if fitness_of_keys(center, dist_matrix) < fitness[i]:
                    direction = center - Xi
                    norm = np.linalg.norm(direction)
                    if norm > 1e-12:
                        move = Xi + step_t * rng.rand() * direction / norm
                        move = np.clip(move, lo, hi)
                        candidate_moves.append((move, fitness_of_keys(move, dist_matrix)))

            if len(neighbor_idx) > 0 and len(neighbor_idx) / pop_size < crowding_factor:
                best_neighbor = neighbor_idx[np.argmin(fitness[neighbor_idx])]
                if fitness[best_neighbor] < fitness[i]:
                    direction = population[best_neighbor] - Xi
                    norm = np.linalg.norm(direction)
                    if norm > 1e-12:
                        move = Xi + step_t * rng.rand() * direction / norm
                        move = np.clip(move, lo, hi)
                        candidate_moves.append((move, fitness_of_keys(move, dist_matrix)))

            if candidate_moves:
                best_move = min(candidate_moves, key=lambda m: m[1])
                new_population[i] = best_move[0]
                continue

            improved = False
            for _ in range(try_number):
                Xj = Xi + visual_t * rng.uniform(-1, 1, size=n_cities)
                Xj = np.clip(Xj, lo, hi)
                if fitness_of_keys(Xj, dist_matrix) < fitness[i]:
                    direction = Xj - Xi
                    norm = np.linalg.norm(direction)
                    move = Xi + step_t * rng.rand() * direction / (norm + 1e-12)
                    new_population[i] = np.clip(move, lo, hi)
                    improved = True
                    break

            if not improved:
                move = Xi + step_t * rng.uniform(-1, 1, size=n_cities)
                new_population[i] = np.clip(move, lo, hi)

        population = new_population
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
    run_afsa.evaluation_history = evaluation_history
    return best_tour, fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    N_CITIES = 25
    cities = generate_cities(n_cities=N_CITIES, seed=42)
    dmat = distance_matrix(cities)

    best_tour, best_len, best_hist, mean_hist, pos_hist = run_afsa(
        dmat, N_CITIES, pop_size=30, n_iterations=500, seed=42,
    )

    print("\n=== Result (AFSA on TSP) ===")
    print("Best tour:", best_tour)
    print("Best tour length:", best_len)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(best_hist, label="Best")
    axes[0].plot(mean_hist, label="Mean", alpha=0.7)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Tour length")
    axes[0].set_title("AFSA Convergence on TSP")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    tour_coords = cities[np.append(best_tour, best_tour[0])]
    axes[1].plot(tour_coords[:, 0], tour_coords[:, 1], "o-")
    for i, (x, y) in enumerate(cities):
        axes[1].annotate(str(i), (x, y), fontsize=8)
    axes[1].set_title(f"Best Tour Found (length={best_len:.2f})")

    plt.tight_layout()
    plt.show()