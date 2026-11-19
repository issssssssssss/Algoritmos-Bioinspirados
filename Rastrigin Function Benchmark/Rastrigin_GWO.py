"""
Grey Wolf Optimizer (GWO) applied to the Rastrigin benchmark function.

Same algorithm as gwo_sphere.py -- see that file for the full phase-by-
phase explanation of the social hierarchy (alpha/beta/delta), encircling
update rule, and the exploration->exploitation decay of `a`. Only the
objective function changes here.

Worth watching for: GWO's position update always pulls every wolf
toward a BLEND of the 3 current best wolves. On a smooth landscape
(sphere) that blend reliably points toward the true optimum. On a
highly multimodal landscape like Rastrigin, alpha/beta/delta can easily
be three different local optima, so the averaged pull can point
somewhere between them rather than toward any real optimum -- a good
discussion point for why GWO's relative performance often drops on
multimodal benchmarks compared to unimodal ones.
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


def run_gwo(
    n_dim=10,
    pop_size=30,
    n_iterations=300,
    bounds=(-5.12, 5.12),
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
        order = np.argsort(fitness)
        alpha, beta, delta = population[order[0]], population[order[1]], population[order[2]]

        a = 2 - it * (2 / n_iterations)

        new_population = np.empty_like(population)
        for i in range(pop_size):
            X = population[i]
            X_updates = []
            for leader in (alpha, beta, delta):
                r1, r2 = rng.rand(n_dim), rng.rand(n_dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * leader - X)
                X_updates.append(leader - A * D)
            new_population[i] = np.mean(X_updates, axis=0)

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
    run_gwo.evaluation_history = evaluation_history
    return population[best_idx], fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, best_hist, mean_hist, pos_hist = run_gwo(
        n_dim=10, pop_size=30, n_iterations=300, bounds=(-5.12, 5.12), seed=42,
    )

    print("\n=== Result (GWO on Rastrigin) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title("GWO Convergence on Rastrigin Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()