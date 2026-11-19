"""
Whale Optimization Algorithm (WOA) applied to the Rastrigin benchmark
function.

Same algorithm as woa_sphere.py -- see that file for the full phase-by-
phase explanation (encircling, random-whale search, spiral bubble-net
attack). Only the objective function changes here.

Worth watching for: WOA's "search for prey" phase (moving toward a
RANDOM whale rather than the current best) is exactly the kind of
exploration multimodal landscapes need, so WOA often handles Rastrigin
relatively better than it handled sphere, compared to GWO/PSO which
rely more heavily on committing to a single best-so-far leader.
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


def run_woa(
    n_dim=10,
    pop_size=30,
    n_iterations=300,
    bounds=(-5.12, 5.12),
    b=1.0,
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
        best_idx = np.argmin(fitness)
        X_star = population[best_idx].copy()

        a = 2 - it * (2 / n_iterations)

        new_population = np.empty_like(population)
        for i in range(pop_size):
            X = population[i]
            r1, r2 = rng.rand(n_dim), rng.rand(n_dim)
            A = 2 * a * r1 - a
            C = 2 * r2
            p = rng.rand()

            if p < 0.5:
                if np.linalg.norm(A) < 1:
                    D = np.abs(C * X_star - X)
                    X_new = X_star - A * D
                else:
                    rand_idx = rng.randint(pop_size)
                    X_rand = population[rand_idx]
                    D = np.abs(C * X_rand - X)
                    X_new = X_rand - A * D
            else:
                l = rng.uniform(-1, 1, size=n_dim)
                D_prime = np.abs(X_star - X)
                X_new = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) + X_star

            new_population[i] = X_new

        population = np.clip(new_population, lo, hi)
        fitness = evaluate_population(population)

        best_idx = np.argmin(fitness)
        best_history.append(fitness[best_idx])
        mean_history.append(np.mean(fitness))
        best_position_history.append(population[best_idx].copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 30 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best: {best_history[-1]:.6f} | Mean: {mean_history[-1]:.6f}")

    best_idx = np.argmin(fitness)
    run_woa.evaluation_history = evaluation_history
    return population[best_idx], fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, best_hist, mean_hist, pos_hist = run_woa(
        n_dim=10, pop_size=30, n_iterations=300, bounds=(-5.12, 5.12), seed=42,
    )

    print("\n=== Result (WOA on Rastrigin) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title("WOA Convergence on Rastrigin Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()