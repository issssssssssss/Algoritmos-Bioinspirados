"""
Artificial Immune System (AIS) applied to the Sphere benchmark function,
using the Clonal Selection Algorithm formulation (CLONALG, de Castro &
Von Zuben, 2002) -- the standard AIS variant for optimization.

AIS models the immune system's response to antigens: antibodies
(candidate solutions) that recognize a threat well (good fitness) get
cloned and fine-tuned; weak antibodies are discarded and replaced.

FITNESS / "AFFINITY": affinity is just the inverse relationship to
fitness -- lower fitness = higher affinity, for a minimization problem.
We compare raw fitness values directly, same as sphere(x) everywhere
else.

PHASES (every iteration):

1. SELECTION
   Sort the whole antibody population by fitness. Select the n_select
   BEST antibodies (highest affinity) to undergo cloning -- weak
   antibodies are not cloned at all this round.

2. PROPORTIONAL CLONING
   Each of the n_select selected antibodies produces `clones_per_antibody`
   copies of itself. (In the full CLONALG formulation the number of
   clones is proportional to rank -- the very best antibody gets the
   most clones -- we use a fixed clone count per selected antibody here
   for simplicity, which is a common practical simplification.)

3. AFFINITY MATURATION (hypermutation)
   Each clone is mutated with Gaussian noise, but the mutation STRENGTH
   is INVERSELY proportional to the parent's affinity (rank):
       mutation_std = f(rank)   -- better rank (lower index) => smaller mutation
   This is the immune system's "somatic hypermutation" idea: cells that
   already bind well only need small refinements, while weaker cells
   need bigger, more exploratory jumps to have a chance of improving.

4. CLONE SELECTION
   For each parent, keep the best of its mutated clones. If that best
   clone beats the ORIGINAL parent's fitness, it replaces the parent in
   the population; otherwise the parent survives unchanged.

5. RECEPTOR EDITING (diversity introduction)
   The worst `replace_rate` fraction of the population is discarded and
   replaced with completely new random antibodies. This mimics the
   immune system continuously generating fresh antibody diversity, and
   is AIS's mechanism for escaping local optima / avoiding premature
   convergence -- functionally similar to ABC's scout bees.

Formulas/structure follow the canonical CLONALG algorithm.
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


def run_ais(
    n_dim=10,
    pop_size=30,
    n_iterations=200,
    bounds=(-5.12, 5.12),
    n_select=10,               # number of antibodies selected for cloning each iteration
    clones_per_antibody=5,
    beta=2.0,                  # controls how steeply mutation strength grows with worse rank
    replace_rate=0.2,          # fraction of population replaced by new random antibodies each iteration
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
        # --- 1. Selection ---
        order = np.argsort(fitness)
        selected = order[:n_select]

        # --- 2 & 3 & 4. Cloning, hypermutation, clone selection ---
        for rank, idx in enumerate(selected):
            affinity_normalized = rank / (n_select - 1) if n_select > 1 else 0  # 0=best, 1=worst
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

        # --- 5. Receptor editing: replace the worst antibodies ---
        worst_idx = np.argsort(fitness)[-n_replace:]
        for idx in worst_idx:
            population[idx] = rng.uniform(lo, hi, size=n_dim)
            fitness[idx] = evaluate(population[idx])

        best_idx = np.argmin(fitness)
        best_history.append(fitness[best_idx])
        mean_history.append(np.mean(fitness))
        best_position_history.append(population[best_idx].copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 20 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best: {best_history[-1]:.8f} | Mean: {mean_history[-1]:.6f}")

    best_idx = np.argmin(fitness)
    run_ais.evaluation_history = evaluation_history
    return population[best_idx], fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, best_hist, mean_hist, pos_hist = run_ais(
        n_dim=10, pop_size=30, n_iterations=200, bounds=(-5.12, 5.12), seed=42,
    )

    print("\n=== Result (AIS on Sphere) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title("AIS (CLONALG) Convergence on Sphere Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()