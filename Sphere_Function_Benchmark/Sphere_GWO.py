"""
Grey Wolf Optimizer (GWO) applied to the Sphere benchmark function.

GWO (Mirjalili et al., 2014) mimics the leadership hierarchy and hunting
behavior of grey wolves. The population is split conceptually into a
hierarchy every iteration:

    alpha (best)  ->  beta (2nd best)  ->  delta (3rd best)  ->  omega (rest)

PHASES (repeated every iteration):

1. SOCIAL HIERARCHY UPDATE
   Evaluate all wolves, then re-identify alpha, beta, delta as simply the
   3 best solutions found so far in the current population. There's no
   persistent "leader" wolf object -- the roles are reassigned fresh each
   iteration based on current fitness.

2. ENCIRCLING PREY (position update)
   Every other wolf (including alpha/beta/delta themselves) updates its
   position by being pulled toward alpha, beta, AND delta simultaneously,
   then averaging the three pulls:
       D_alpha = |C1*alpha - X|,   X1 = alpha - A1*D_alpha
       D_beta  = |C2*beta  - X|,   X2 = beta  - A2*D_beta
       D_delta = |C3*delta - X|,   X3 = delta - A3*D_delta
       X_new = (X1 + X2 + X3) / 3
   A and C are randomized coefficient vectors that control how strongly
   the wolf is pulled and how much randomness is injected.

3. EXPLORATION -> EXPLOITATION TRANSITION
   The key control parameter is `a`, which decays LINEARLY from 2 to 0
   over the run. `A = 2*a*r1 - a` therefore shrinks in magnitude as `a`
   shrinks, meaning early iterations take large, exploratory steps
   (|A|>1 pushes wolves AWAY from leaders sometimes -- global search) and
   late iterations take small steps that hone in on the leaders' exact
   position (local search / attack).

There's no separate "scout" or "mutation" phase like GA -- exploration
in GWO comes entirely from the randomized A/C coefficients and the decay
of `a`.
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


def run_gwo(
    n_dim=10,
    pop_size=30,
    n_iterations=200,
    bounds=(-5.12, 5.12),
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

    for it in range(n_iterations):
        # --- 1. Identify alpha, beta, delta (the 3 fittest wolves) ---
        order = np.argsort(fitness)
        alpha, beta, delta = population[order[0]], population[order[1]], population[order[2]]

        # --- 2 & 3. Exploration/exploitation coefficient ---
        a = 2 - it * (2 / n_iterations)  # linearly 2 -> 0

        new_population = np.empty_like(population)
        for i in range(pop_size):
            X = population[i]
            X_updates = []
            for leader in (alpha, beta, delta):
                r1, r2 = rng.rand(n_dim), rng.rand(n_dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * leader - X)
                X_updates.append(leader - A * D)
            new_population[i] = np.mean(X_updates, axis=0)

        population = np.clip(new_population, lo, hi)
        fitness = evaluate_population(population)

        best_idx = np.argmin(fitness)
        best_history.append(fitness[best_idx])
        mean_history.append(np.mean(fitness))
        best_position_history.append(population[best_idx].copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 20 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best: {best_history[-1]:.8f} | Mean: {mean_history[-1]:.6f}")

    best_idx = np.argmin(fitness)
    run_gwo.evaluation_history = evaluation_history
    return population[best_idx], fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, best_hist, mean_hist, pos_hist = run_gwo(
        n_dim=10, pop_size=30, n_iterations=200, bounds=(-5.12, 5.12), seed=42,
    )

    print("\n=== Result (GWO on Sphere) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title("GWO Convergence on Sphere Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()