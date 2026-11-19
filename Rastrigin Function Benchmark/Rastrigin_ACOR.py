"""
Ant Colony Optimization for continuous domains (ACOR) applied to the
Rastrigin benchmark function.

Same algorithm as acor_sphere.py -- see that file for the full phase-by-
phase explanation (archive as pheromone table, rank-based Gaussian
weights, solution construction, archive update) and the note on why
classic (combinatorial) ACO needed adapting for continuous problems.
Only the objective function changes here.

Worth watching for: ACOR's search radius (sigma) is tied to how spread
out the archive currently is. On Rastrigin, if the archive converges
too early around one local optimum, sigma shrinks and the colony loses
the ability to escape it -- similar failure mode to PSO's swarm
collapse, but arrived at through a different mechanism (archive spread
vs. velocity/gbest pull).
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


def run_acor(
    n_dim=10,
    archive_size=30,
    n_ants=30,
    n_iterations=300,
    bounds=(-5.12, 5.12),
    q=0.5,
    xi=0.85,
    seed=None,
    verbose=True,
):
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)
    lo, hi = bounds

    archive = rng.uniform(lo, hi, size=(archive_size, n_dim))
    archive_fitness = evaluate_population(archive)
    order = np.argsort(archive_fitness)
    archive, archive_fitness = archive[order], archive_fitness[order]

    ranks = np.arange(1, archive_size + 1)
    weights = (1.0 / (q * archive_size * np.sqrt(2 * np.pi))) * \
        np.exp(-(ranks - 1) ** 2 / (2 * q ** 2 * archive_size ** 2))
    probs = weights / weights.sum()

    best_history = []
    mean_history = []
    best_position_history = []

    for it in range(n_iterations):
        new_ants = np.empty((n_ants, n_dim))

        for a in range(n_ants):
            l = rng.choice(archive_size, p=probs)
            for j in range(n_dim):
                sigma = xi / (archive_size - 1) * np.sum(np.abs(archive[:, j] - archive[l, j]))
                new_ants[a, j] = rng.normal(archive[l, j], sigma if sigma > 0 else 1e-6)

        new_ants = np.clip(new_ants, lo, hi)
        new_fitness = evaluate_population(new_ants)

        combined = np.vstack([archive, new_ants])
        combined_fitness = np.concatenate([archive_fitness, new_fitness])
        order = np.argsort(combined_fitness)[:archive_size]
        archive, archive_fitness = combined[order], combined_fitness[order]

        best_history.append(archive_fitness[0])
        mean_history.append(np.mean(np.concatenate([archive_fitness, new_fitness])))
        best_position_history.append(archive[0].copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 30 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best: {best_history[-1]:.6f} | Mean: {mean_history[-1]:.6f}")

    run_acor.evaluation_history = evaluation_history
    return archive[0], archive_fitness[0], best_history, mean_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, best_hist, mean_hist, pos_hist = run_acor(
        n_dim=10, archive_size=30, n_ants=30, n_iterations=300,
        bounds=(-5.12, 5.12), seed=42,
    )

    print("\n=== Result (ACOR on Rastrigin) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title("ACOR Convergence on Rastrigin Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()