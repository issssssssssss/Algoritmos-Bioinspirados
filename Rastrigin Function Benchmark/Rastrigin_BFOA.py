"""
Bacterial Foraging Optimization Algorithm (BFOA) applied to the
Rastrigin benchmark function.

Same algorithm as bfoa_sphere.py -- see that file for the full phase-by-
phase explanation (chemotaxis with swarming penalty, reproduction,
elimination-dispersal). Only the objective function changes here.

Worth watching for: the ELIMINATION-DISPERSAL phase (each bacterium has
a small probability `Ped` of being randomly teleported elsewhere in the
search space) is BFOA's most direct multimodal-escape mechanism --
functionally similar to AIS's receptor editing or ABC's scout bees, but
triggered purely by chance rather than by tracking stagnation.
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


def compute_Jcc(X, population, d_attract=0.1, w_attract=0.2, h_repellant=0.1, w_repellant=10):
    diffs = population - X
    sq_dists = np.sum(diffs ** 2, axis=1)
    attract = -d_attract * np.exp(-w_attract * sq_dists)
    repel = h_repellant * np.exp(-w_repellant * sq_dists)
    return np.sum(attract + repel)


def run_bfoa(
    n_dim=10,
    pop_size=30,
    n_iterations=300,
    bounds=(-5.12, 5.12),
    Nc=10,
    Ns=4,
    Nre=4,
    Ped=0.25,
    C=0.1,
    seed=None,
    verbose=True,
):
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)
    lo, hi = bounds
    step_size = C * (hi - lo)

    population = rng.uniform(lo, hi, size=(pop_size, n_dim))
    fitness = evaluate_population(population)
    health = np.zeros(pop_size)

    best_history = []
    mean_history = []
    best_position_history = []

    reproduction_count = 0

    for it in range(n_iterations):
        for i in range(pop_size):
            delta = rng.uniform(-1, 1, size=n_dim)
            delta = delta / (np.linalg.norm(delta) + 1e-12)

            current_cost = fitness[i] + compute_Jcc(population[i], population)
            new_pos = np.clip(population[i] + step_size * delta, lo, hi)
            new_cost = evaluate(new_pos) + compute_Jcc(new_pos, population)

            if new_cost < current_cost:
                population[i], fitness[i] = new_pos, evaluate(new_pos)
                for _ in range(Ns):
                    trial = np.clip(population[i] + step_size * delta, lo, hi)
                    trial_cost = evaluate(trial) + compute_Jcc(trial, population)
                    if trial_cost < fitness[i] + compute_Jcc(population[i], population):
                        population[i], fitness[i] = trial, evaluate(trial)
                    else:
                        break

            health[i] += fitness[i]

        best_idx = np.argmin(fitness)
        best_history.append(fitness[best_idx])
        mean_history.append(np.mean(fitness))
        best_position_history.append(population[best_idx].copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 30 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best: {best_history[-1]:.6f} | Mean: {mean_history[-1]:.6f}")

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
                        population[i] = rng.uniform(lo, hi, size=n_dim)
                        fitness[i] = evaluate(population[i])

    best_idx = np.argmin(fitness)
    run_bfoa.evaluation_history = evaluation_history
    return population[best_idx], fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, best_hist, mean_hist, pos_hist = run_bfoa(
        n_dim=10, pop_size=30, n_iterations=300, bounds=(-5.12, 5.12), seed=42,
    )

    print("\n=== Result (BFOA on Rastrigin) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Chemotaxis step")
    plt.ylabel("Fitness (log scale)")
    plt.title("BFOA Convergence on Rastrigin Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()