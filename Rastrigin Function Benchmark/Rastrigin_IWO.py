"""
Invasive Weed Optimization (IWO) applied to the Rastrigin benchmark
function.

Same algorithm as iwo_sphere.py -- see that file for the full phase-by-
phase explanation (proportional seeding, shrinking Gaussian dispersal,
competitive exclusion). Only the objective function changes here.

Worth watching for: IWO has NO explicit random-restart/diversity
mechanism the way AIS/ABC/BFOA do -- every plant, however weak, still
gets at least `min_seeds` seeds, and competitive exclusion is the only
selection pressure. On a multimodal landscape this means IWO relies
entirely on its dispersal radius (sigma) staying large enough, for long
enough, to let seeds land in different basins before it shrinks too
much -- a real risk of premature convergence if sigma decays too fast
relative to the distance between Rastrigin's local optima (spaced 1
unit apart).
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


def run_iwo(
    n_dim=10,
    pop_size=30,
    n_iterations=300,
    bounds=(-5.12, 5.12),
    min_seeds=0,
    max_seeds=5,
    exponent=2,
    sigma_initial=0.5,
    sigma_final=0.001,
    seed=None,
    verbose=True,
):
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)
    lo, hi = bounds
    pop_max = pop_size

    population = rng.uniform(lo, hi, size=(pop_size, n_dim))
    fitness = evaluate_population(population)

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
                s = population[i] + rng.normal(0, sigma_t, size=n_dim)
                all_seeds.append(np.clip(s, lo, hi))

        if all_seeds:
            all_seeds = np.array(all_seeds)
            seed_fitness = evaluate_population(all_seeds)
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
        best_position_history.append(population[best_idx].copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 30 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best: {best_history[-1]:.6f} | Mean: {mean_history[-1]:.6f}")

    best_idx = np.argmin(fitness)
    run_iwo.evaluation_history = evaluation_history
    return population[best_idx], fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, best_hist, mean_hist, pos_hist = run_iwo(
        n_dim=10, pop_size=30, n_iterations=300, bounds=(-5.12, 5.12), seed=42,
    )

    print("\n=== Result (IWO on Rastrigin) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title("IWO Convergence on Rastrigin Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()