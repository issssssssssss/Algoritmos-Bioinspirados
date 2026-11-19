"""
Intelligent Water Drops (IWD) applied to the Rastrigin benchmark
function.

Same algorithm as iwd_sphere.py -- see that file for the full phase-by-
phase explanation and the note on adapting IWD from its native
graph-based formulation into continuous space (same kind of adaptation
ACOR made for ACO). Only the objective function changes here.

Worth watching for: the GLOBAL SOIL-DRIVEN CONVERGENCE phase (pulling
every drop toward the current best, weighted by the drop's own
velocity) is a single-attractor mechanism similar to PSO's gbest pull
or WOA's encircling phase. On Rastrigin, this is exactly the kind of
mechanism that risks collapsing the whole swarm toward the FIRST decent
local optimum found, before the candidate-generation phase (which still
carries some randomness) has had a chance to discover the true global
optimum.
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


def run_iwd(
    n_dim=10,
    pop_size=30,
    n_iterations=300,
    bounds=(-5.12, 5.12),
    n_candidates=5,
    av=1.0, bv=0.01, cv=1.0,
    init_vel=1.0,
    radius_initial=1.5, radius_final=0.01,
    pull_strength=0.05,
    seed=None,
    verbose=True,
):
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)
    lo, hi = bounds

    positions = rng.uniform(lo, hi, size=(pop_size, n_dim))
    velocities = np.full(pop_size, init_vel)
    fitness = evaluate_population(positions)

    best_history = []
    mean_history = []
    best_position_history = []

    for it in range(n_iterations):
        radius = radius_initial * (1 - it / n_iterations) + radius_final

        for i in range(pop_size):
            candidates = positions[i] + radius * rng.uniform(-1, 1, size=(n_candidates, n_dim))
            candidates = np.clip(candidates, lo, hi)
            cand_fitness = evaluate_population(candidates)

            soil = cand_fitness - cand_fitness.min() + 1e-6
            desirability = 1.0 / soil
            probs = desirability / desirability.sum()
            choice = rng.choice(n_candidates, p=probs)
            chosen_soil = soil[choice]

            velocities[i] = velocities[i] + av / (bv + cv * chosen_soil ** 2)
            positions[i] = candidates[choice]

        fitness = evaluate_population(positions)
        best_idx = np.argmin(fitness)
        X_best = positions[best_idx].copy()

        v_max = velocities.max()
        for i in range(pop_size):
            pull = pull_strength * (velocities[i] / v_max)
            positions[i] = positions[i] + pull * (X_best - positions[i])
            positions[i] = np.clip(positions[i], lo, hi)

        fitness = evaluate_population(positions)
        best_idx = np.argmin(fitness)

        best_history.append(fitness[best_idx])
        mean_history.append(np.mean(fitness))
        best_position_history.append(positions[best_idx].copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 30 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best: {best_history[-1]:.6f} | Mean: {mean_history[-1]:.6f}")

    best_idx = np.argmin(fitness)
    run_iwd.evaluation_history = evaluation_history
    return positions[best_idx], fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, best_hist, mean_hist, pos_hist = run_iwd(
        n_dim=10, pop_size=30, n_iterations=300, bounds=(-5.12, 5.12), seed=42,
    )

    print("\n=== Result (IWD on Rastrigin) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title("IWD Convergence on Rastrigin Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()