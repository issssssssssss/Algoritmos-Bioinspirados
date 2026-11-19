"""
Paddy Field Algorithm (PFA) applied to the Rastrigin benchmark function.

Same algorithm as pfa_sphere.py -- see that file for the full phase-by-
phase explanation (selective seeding, proportional dispersal,
pollination, population control) AND the caveat that PFA is much less
standardized in the literature than GWO/WOA/FA/IWO/AIS/BFOA -- treat
this as a faithful-to-structure but not "canonical formula" implementation.
Only the objective function changes here.

Worth watching for: the POLLINATION phase (replacing the worst plants
with a blend of a good plant and a totally random point) is PFA's
explicit long-range exploration mechanism, functionally similar to
AIS's receptor editing but directional (biased toward a known-good
plant) rather than fully random.
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


def run_pfa(
    n_dim=10,
    pop_size=30,
    n_iterations=300,
    bounds=(-5.12, 5.12),
    selection_rate=0.3,
    max_seeds=5,
    sigma_initial=0.5,
    sigma_final=0.001,
    exponent=2,
    pollination_rate=0.1,
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

    n_select = max(1, int(selection_rate * pop_size))
    n_pollinate = max(1, int(pollination_rate * pop_size))

    for it in range(n_iterations):
        sigma_t = ((n_iterations - it) / n_iterations) ** exponent * \
            (sigma_initial - sigma_final) + sigma_final

        order = np.argsort(fitness)
        selected = order[:n_select]

        f_min, f_max = fitness[selected].min(), fitness[selected].max()
        seeds = []
        for rank, idx in enumerate(selected):
            if f_max == f_min:
                n_seeds = max_seeds
            else:
                ratio = (f_max - fitness[idx]) / (f_max - f_min)
                n_seeds = max(1, int(round(ratio * max_seeds)))
            for _ in range(n_seeds):
                s = population[idx] + rng.normal(0, sigma_t, size=n_dim)
                seeds.append(np.clip(s, lo, hi))
        seeds = np.array(seeds) if seeds else np.empty((0, n_dim))
        seed_fitness = evaluate_population(seeds) if len(seeds) else np.empty(0)

        worst_idx = order[-n_pollinate:]
        for idx in worst_idx:
            donor = population[selected[rng.randint(len(selected))]]
            random_component = rng.uniform(lo, hi, size=n_dim)
            pollinated = 0.5 * donor + 0.5 * random_component
            population[idx] = np.clip(pollinated, lo, hi)
            fitness[idx] = evaluate(population[idx])

        combined = np.vstack([population, seeds]) if len(seeds) else population
        combined_fitness = np.concatenate([fitness, seed_fitness]) if len(seeds) else fitness
        keep_order = np.argsort(combined_fitness)[:pop_size]
        population = combined[keep_order]
        fitness = combined_fitness[keep_order]

        best_idx = np.argmin(fitness)
        best_history.append(fitness[best_idx])
        mean_history.append(np.mean(fitness))
        best_position_history.append(population[best_idx].copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 30 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best: {best_history[-1]:.6f} | Mean: {mean_history[-1]:.6f}")

    best_idx = np.argmin(fitness)
    run_pfa.evaluation_history = evaluation_history
    return population[best_idx], fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, best_hist, mean_hist, pos_hist = run_pfa(
        n_dim=10, pop_size=30, n_iterations=300, bounds=(-5.12, 5.12), seed=42,
    )

    print("\n=== Result (PFA on Rastrigin) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title("PFA Convergence on Rastrigin Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()