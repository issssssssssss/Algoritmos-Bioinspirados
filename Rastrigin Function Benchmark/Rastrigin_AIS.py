"""
Artificial Immune System (AIS / CLONALG) applied to the Rastrigin
benchmark function.

Same algorithm as ais_sphere.py -- see that file for the full phase-by-
phase explanation (selection, proportional cloning, affinity
maturation/hypermutation, clone selection, receptor editing). Only the
objective function changes here.

Worth watching for: the RECEPTOR EDITING phase (replacing the worst
`replace_rate` fraction of the population with brand-new random
antibodies every single iteration) is a much more aggressive, constant
diversity injection than ABC's scout bees (which only trigger after a
source stagnates for `limit` trials) or IWO's population cap. This
makes AIS structurally well-suited to multimodal landscapes, since it
never lets the population fully homogenize.
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


def run_ais(
    n_dim=10,
    pop_size=30,
    n_iterations=300,
    bounds=(-5.12, 5.12),
    n_select=10,
    clones_per_antibody=5,
    beta=2.0,
    replace_rate=0.2,
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
    n_replace = max(1, int(replace_rate * pop_size))

    for it in range(n_iterations):
        order = np.argsort(fitness)
        selected = order[:n_select]

        for rank, idx in enumerate(selected):
            affinity_normalized = rank / (n_select - 1) if n_select > 1 else 0
            mutation_std = (0.01 + 0.5 * affinity_normalized) * (hi - lo)

            best_clone, best_clone_fit = population[idx].copy(), fitness[idx]
            for _ in range(clones_per_antibody):
                clone = population[idx] + rng.normal(0, mutation_std, size=n_dim)
                clone = np.clip(clone, lo, hi)
                clone_fit = evaluate(clone)
                if clone_fit < best_clone_fit:
                    best_clone, best_clone_fit = clone, clone_fit

            if best_clone_fit < fitness[idx]:
                population[idx] = best_clone
                fitness[idx] = best_clone_fit

        worst_idx = np.argsort(fitness)[-n_replace:]
        for idx in worst_idx:
            population[idx] = rng.uniform(lo, hi, size=n_dim)
            fitness[idx] = evaluate(population[idx])

        best_idx = np.argmin(fitness)
        best_history.append(fitness[best_idx])
        mean_history.append(np.mean(fitness))
        best_position_history.append(population[best_idx].copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 30 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best: {best_history[-1]:.6f} | Mean: {mean_history[-1]:.6f}")

    best_idx = np.argmin(fitness)
    run_ais.evaluation_history = evaluation_history
    return population[best_idx], fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, best_hist, mean_hist, pos_hist = run_ais(
        n_dim=10, pop_size=30, n_iterations=300, bounds=(-5.12, 5.12), seed=42,
    )

    print("\n=== Result (AIS on Rastrigin) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title("AIS (CLONALG) Convergence on Rastrigin Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()