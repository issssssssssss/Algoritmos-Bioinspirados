"""
Firefly Algorithm (FA) applied to the Rastrigin benchmark function.

Same algorithm as fa_sphere.py -- see that file for the full phase-by-
phase explanation (pairwise attraction, distance-based attractiveness
decay, randomness decay) and the note about scaling gamma to the search
space size. Only the objective function changes here.

Worth watching for: FA naturally maintains several attraction "clusters"
rather than collapsing onto one point (each firefly is only pulled
toward BRIGHTER fireflies, and there can be several separate bright
spots at once). On a multimodal landscape like Rastrigin this is
actually helpful -- different clusters can form around different local
optima, and the random-walk term (alpha*epsilon) gives each cluster a
chance to hop out.
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


def run_fa(
    n_dim=10,
    pop_size=30,
    n_iterations=300,
    bounds=(-5.12, 5.12),
    alpha=0.5,
    beta0=1.0,
    gamma=0.01,
    alpha_decay=0.97,
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

    alpha_t = alpha
    for it in range(n_iterations):
        alpha_t *= alpha_decay

        for i in range(pop_size):
            for j in range(pop_size):
                if fitness[j] < fitness[i]:
                    r = np.linalg.norm(population[i] - population[j])
                    beta = beta0 * np.exp(-gamma * r ** 2)
                    eps = alpha_t * (rng.rand(n_dim) - 0.5)
                    population[i] = population[i] + beta * (population[j] - population[i]) + eps
                    population[i] = np.clip(population[i], lo, hi)
                    fitness[i] = evaluate(population[i])

        best_idx = np.argmin(fitness)
        best_history.append(fitness[best_idx])
        mean_history.append(np.mean(fitness))
        best_position_history.append(population[best_idx].copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 30 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best: {best_history[-1]:.6f} | Mean: {mean_history[-1]:.6f}")

    best_idx = np.argmin(fitness)
    run_fa.evaluation_history = evaluation_history
    return population[best_idx], fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, best_hist, mean_hist, pos_hist = run_fa(
        n_dim=10, pop_size=30, n_iterations=300, bounds=(-5.12, 5.12), seed=42,
    )

    print("\n=== Result (FA on Rastrigin) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title("FA Convergence on Rastrigin Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()