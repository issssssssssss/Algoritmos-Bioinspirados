"""
Artificial Fish Swarm Algorithm (AFSA) applied to the Rastrigin
benchmark function.

Same algorithm as afsa_sphere.py -- see that file for the full phase-by-
phase explanation (swarming, following, preying, random movement) and
the note about adding a shrinking visual/step schedule. Only the
objective function changes here.

Worth watching for: AFSA's "preying" phase (random trial points within
visual range, moving toward the first one that improves) gives it a
built-in local-optima-escape mechanism that's more stochastic than
GWO/WOA's leader-following -- each fish independently gambles on random
nearby points rather than all being pulled toward the same leader(s).
"""

import numpy as np
import matplotlib.pyplot as plt


evaluation_count = 0


def rastrigin(x, A=10):
    n = len(x)
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def evaluate(x):
    global evaluation_count
    evaluation_count += 1
    return rastrigin(x)


def evaluate_population(population):
    return np.array([evaluate(ind) for ind in population])


def run_afsa(
    n_dim=10,
    pop_size=30,
    n_iterations=300,
    bounds=(-5.12, 5.12),
    visual=2.0,
    step=0.5,
    try_number=5,
    crowding_factor=0.6,
    seed=None,
    verbose=True,
):
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)
    lo, hi = bounds

    population = rng.uniform(lo, hi, size=(pop_size, n_dim))
    fitness = evaluate_population(population)

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
                if evaluate(center) < fitness[i]:
                    direction = center - Xi
                    norm = np.linalg.norm(direction)
                    if norm > 1e-12:
                        move = Xi + step_t * rng.rand() * direction / norm
                        move = np.clip(move, lo, hi)
                        candidate_moves.append((move, evaluate(move)))

            if len(neighbor_idx) > 0 and len(neighbor_idx) / pop_size < crowding_factor:
                best_neighbor = neighbor_idx[np.argmin(fitness[neighbor_idx])]
                if fitness[best_neighbor] < fitness[i]:
                    direction = population[best_neighbor] - Xi
                    norm = np.linalg.norm(direction)
                    if norm > 1e-12:
                        move = Xi + step_t * rng.rand() * direction / norm
                        move = np.clip(move, lo, hi)
                        candidate_moves.append((move, evaluate(move)))

            if candidate_moves:
                best_move = min(candidate_moves, key=lambda m: m[1])
                new_population[i] = best_move[0]
                continue

            improved = False
            for _ in range(try_number):
                Xj = Xi + visual_t * rng.uniform(-1, 1, size=n_dim)
                Xj = np.clip(Xj, lo, hi)
                if evaluate(Xj) < fitness[i]:
                    direction = Xj - Xi
                    norm = np.linalg.norm(direction)
                    move = Xi + step_t * rng.rand() * direction / (norm + 1e-12)
                    new_population[i] = np.clip(move, lo, hi)
                    improved = True
                    break

            if not improved:
                move = Xi + step_t * rng.uniform(-1, 1, size=n_dim)
                new_population[i] = np.clip(move, lo, hi)

        population = new_population
        fitness = evaluate_population(population)

        best_idx = np.argmin(fitness)
        best_history.append(fitness[best_idx])
        mean_history.append(np.mean(fitness))
        best_position_history.append(population[best_idx].copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 30 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best: {best_history[-1]:.6f} | Mean: {mean_history[-1]:.6f}")

    best_idx = np.argmin(fitness)
    run_afsa.evaluation_history = evaluation_history
    return population[best_idx], fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, best_hist, mean_hist, pos_hist = run_afsa(
        n_dim=10, pop_size=30, n_iterations=300, bounds=(-5.12, 5.12), seed=42,
    )

    print("\n=== Result (AFSA on Rastrigin) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title("AFSA Convergence on Rastrigin Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()