"""
Ant Colony Optimization for continuous domains (ACOR, Socha & Dorigo,
2008) applied to the Sphere benchmark function.

IMPORTANT ADAPTATION NOTE: classic ACO was designed for COMBINATORIAL
problems (like your TSP) -- ants deposit pheromone on graph EDGES they
travel, and future ants probabilistically follow stronger trails. That
mechanism doesn't exist for continuous variables (there's no "edge" to
lay pheromone on between two arbitrary real numbers). ACOR replaces the
pheromone TRAIL with a "pheromone-weighted ARCHIVE of solutions" and
sampling from Gaussian distributions built around them, but keeps the
same spirit: good solutions attract more future search, and search
intensity concentrates where the best ants have already been.

PHASES (repeated every iteration):

1. ARCHIVE (the "pheromone table")
   Maintain a table of the k best solutions found so far, sorted from
   best (rank 1) to worst (rank k). This table IS the pheromone
   information in ACOR -- instead of a continuous trail, it's a discrete
   set of high-quality reference points.

2. WEIGHT ASSIGNMENT (pheromone strength)
   Each archive solution gets a Gaussian-kernel weight based on its
   RANK (not raw fitness):
       w_i = (1 / (q*k*sqrt(2*pi))) * exp(-(i-1)^2 / (2*q^2*k^2))
   Lower rank (better solution) => higher weight => more pheromone.
   The parameter q controls exploitation vs exploration: small q makes
   the colony heavily favor the very best archive solutions; large q
   spreads attention more evenly across the whole archive.

3. SOLUTION CONSTRUCTION (ants build new solutions)
   Each of the m ants builds a new solution DIMENSION BY DIMENSION:
     a. Pick one archive solution l via roulette-wheel selection using
        the weights above (this is choosing WHICH pheromone trail to
        follow, like an ant choosing an edge in classic ACO).
     b. Sample the new value for that dimension from a Gaussian centered
        on the chosen solution's value in that dimension:
            x_new[j] ~ N(archive[l][j], sigma[j])
        where sigma[j] is the average distance from archive[l][j] to
        every OTHER archive solution's value in dimension j, scaled by a
        convergence-speed parameter xi. This means the search radius
        automatically SHRINKS as the archive converges (solutions get
        closer together), which is ACOR's natural exploration ->
        exploitation transition -- analogous to `a` decaying in
        GWO/WOA, but emergent from the archive's own spread rather than
        an explicit schedule.

4. ARCHIVE UPDATE (pheromone evaporation + reinforcement)
   Merge the new ants with the existing archive, keep only the k best
   overall, discard the rest. This simultaneously reinforces good
   regions (kept solutions) and evaporates bad ones (discarded), the
   same net effect as classic ACO's evaporation + deposit rules.
"""

import numpy as np
import matplotlib.pyplot as plt


evaluation_count = 0


def evaluate(x):
    global evaluation_count
    evaluation_count += 1
    return sphere(x)


def evaluate_population(population):
    return np.array([evaluate(ind) for ind in population])


def sphere(x):
    return np.sum(x ** 2)


def run_acor(
    n_dim=10,
    archive_size=30,     # k: size of the solution archive ("pheromone table")
    n_ants=30,            # m: new solutions constructed per iteration
    n_iterations=200,
    bounds=(-5.12, 5.12),
    q=0.5,                # locality/exploitation parameter (smaller = greedier)
    xi=0.85,               # convergence speed parameter (smaller = faster convergence, more exploitative)
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
            # 3a. choose a guiding archive solution via roulette wheel
            l = rng.choice(archive_size, p=probs)

            for j in range(n_dim):
                # 3b. Gaussian sampling around archive[l][j]
                sigma = xi / (archive_size - 1) * np.sum(np.abs(archive[:, j] - archive[l, j]))
                new_ants[a, j] = rng.normal(archive[l, j], sigma if sigma > 0 else 1e-6)

        new_ants = np.clip(new_ants, lo, hi)
        new_fitness = evaluate_population(new_ants)

        # --- 4. Archive update: merge, keep k best ---
        combined = np.vstack([archive, new_ants])
        combined_fitness = np.concatenate([archive_fitness, new_fitness])
        order = np.argsort(combined_fitness)[:archive_size]
        archive, archive_fitness = combined[order], combined_fitness[order]

        best_history.append(archive_fitness[0])
        mean_history.append(np.mean(np.concatenate([archive_fitness, new_fitness])))
        best_position_history.append(archive[0].copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 20 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best: {best_history[-1]:.8f} | Mean: {mean_history[-1]:.6f}")

    run_acor.evaluation_history = evaluation_history
    return archive[0], archive_fitness[0], best_history, mean_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, best_hist, mean_hist, pos_hist = run_acor(
        n_dim=10, archive_size=30, n_ants=30, n_iterations=200,
        bounds=(-5.12, 5.12), seed=42,
    )

    print("\n=== Result (ACOR on Sphere) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title("ACOR Convergence on Sphere Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()