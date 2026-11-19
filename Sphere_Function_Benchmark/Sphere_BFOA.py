"""
Bacterial Foraging Optimization Algorithm (BFOA) applied to the Sphere
benchmark function (Passino, 2002).

BFOA models how E. coli bacteria forage for nutrients using 4 nested
phases -- this is the most deeply nested/structured algorithm in the
set: chemotaxis happens inside reproduction cycles, which happen inside
elimination-dispersal cycles.

FITNESS: same direct sphere(x) evaluation. Each bacterium also
accumulates a running "health" score (sum of fitness values it
experienced across a reproduction cycle) used only for the reproduction
phase -- lower accumulated health = fitter bacterium overall, not just
at a single point in time.

PHASES:

1. CHEMOTAXIS (tumble + swim) -- happens every step
   Each bacterium picks a random direction ("tumble") and takes one
   step of fixed size in that direction. If the new position is BETTER,
   the bacterium keeps "swimming" in that same direction for up to
   `Ns` further steps (as long as it keeps improving) -- this lets a
   bacterium that stumbles onto a good direction travel further along
   it in one go, rather than re-randomizing direction every single step.

2. SWARMING (cell-to-cell attraction/repulsion)
   Bacteria release chemical attractant/repellent signals: they're
   drawn toward other bacteria (to aggregate near good regions, like
   fireflies clustering) but also mildly repelled at very close range
   (to avoid over-crowding one spot). This is implemented as a penalty
   term J_cc(X) added to the raw fitness ONLY when deciding whether a
   chemotaxis move is an improvement -- the actual best-found value we
   report is always the true, unpenalized sphere(x).

3. REPRODUCTION -- happens every `Nc` chemotaxis steps
   Bacteria are sorted by their accumulated health (total fitness
   experienced during the chemotaxis phase, lower = better forager).
   The healthier HALF of the population survives and each survivor
   splits into two (its position is copied into a "sibling" slot,
   replacing one of the unhealthy half) -- population size stays
   constant, but poor foragers are eliminated.

4. ELIMINATION-DISPERSAL -- happens every `Nre` reproduction cycles
   Each bacterium independently has a small probability `Ped` of being
   "eliminated" (killed and replaced with a fresh, randomly-placed
   bacterium elsewhere in the search space). This models real-world
   events (e.g. a sudden environmental change) that can wipe out
   foragers in a good region and reseed elsewhere, which is BFOA's
   mechanism for maintaining global exploration and escaping local
   optima over the long run.
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


def compute_Jcc(X, population, d_attract=0.1, w_attract=0.2, h_repellant=0.1, w_repellant=10):
    """Cell-to-cell attraction-repulsion penalty for swarming behavior."""
    diffs = population - X
    sq_dists = np.sum(diffs ** 2, axis=1)
    attract = -d_attract * np.exp(-w_attract * sq_dists)
    repel = h_repellant * np.exp(-w_repellant * sq_dists)
    return np.sum(attract + repel)


def run_bfoa(
    n_dim=10,
    pop_size=30,
    n_iterations=200,   # total chemotaxis steps across the whole run
    bounds=(-5.12, 5.12),
    Nc=10,               # chemotactic steps per reproduction cycle
    Ns=4,                # max swim length
    Nre=4,               # reproduction cycles per elimination-dispersal event
    Ped=0.25,            # elimination-dispersal probability per bacterium
    C=0.1,               # basic tumble/run step size, as a fraction of the search range
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
        # --- 1 & 2. Chemotaxis with swarming penalty ---
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

        if verbose and (it % 20 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best: {best_history[-1]:.8f} | Mean: {mean_history[-1]:.6f}")

        # --- 3. Reproduction ---
        if (it + 1) % Nc == 0:
            order = np.argsort(health)
            n_keep = pop_size // 2
            survivors = order[:n_keep]
            weak = order[n_keep:]
            population[weak] = population[survivors]
            fitness[weak] = fitness[survivors]
            health[:] = 0
            reproduction_count += 1

            # --- 4. Elimination-dispersal ---
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
        n_dim=10, pop_size=30, n_iterations=200, bounds=(-5.12, 5.12), seed=42,
    )

    print("\n=== Result (BFOA on Sphere) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Chemotaxis step")
    plt.ylabel("Fitness (log scale)")
    plt.title("BFOA Convergence on Sphere Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()