"""
Whale Optimization Algorithm (WOA) applied to the Sphere benchmark function.

WOA (Mirjalili & Lewis, 2016) mimics humpback whales' bubble-net hunting:
whales encircle prey and then attack along a shrinking, spiral path made
of rising bubbles. Every whale updates using ONE of three mechanisms
each iteration, chosen probabilistically:

PHASES (chosen per-whale, per-iteration):

1. ENCIRCLING PREY (exploitation, when |A| < 1)
   The whale moves toward the current best solution X*:
       D = |C*X* - X|
       X_new = X* - A*D
   Same mathematical form as GWO's leader-following, but here there's
   only ONE leader (the single best whale), not three.

2. SEARCH FOR PREY (exploration, when |A| >= 1)
   Instead of moving toward the best whale, the whale moves toward a
   RANDOMLY CHOSEN whale from the population:
       D = |C*X_rand - X|
       X_new = X_rand - A*D
   This is what stops the whole population collapsing onto the current
   best too early -- it's the main source of global exploration in WOA.

3. SPIRAL BUBBLE-NET ATTACK (exploitation, alternative to #1)
   Independently of the A-based choice above, with 50% probability the
   whale instead spirals toward the best solution, mimicking the
   logarithmic spiral shape of a real bubble-net:
       D' = |X* - X|
       X_new = D' * exp(b*l) * cos(2*pi*l) + X*
   where l is a random number in [-1, 1] and b is a constant defining
   the spiral's shape.

Each iteration, a coin flip (p) decides between mechanism {1 or 2}
(shrinking-encircle/search, governed by A) versus mechanism {3}
(spiral). `a` decays linearly from 2 to 0 exactly like in GWO, shrinking
the encircling radius over time and biasing later iterations toward
exploitation.
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


def run_woa(
    n_dim=10,
    pop_size=30,
    n_iterations=200,
    bounds=(-5.12, 5.12),
    b=1.0,  # shape constant of the logarithmic spiral
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
        best_idx = np.argmin(fitness)
        X_star = population[best_idx].copy()

        a = 2 - it * (2 / n_iterations)  # linearly 2 -> 0

        new_population = np.empty_like(population)
        for i in range(pop_size):
            X = population[i]
            r1, r2 = rng.rand(n_dim), rng.rand(n_dim)
            A = 2 * a * r1 - a
            C = 2 * r2
            p = rng.rand()

            if p < 0.5:
                if np.linalg.norm(A) < 1:
                    # Phase 1: encircle the best whale
                    D = np.abs(C * X_star - X)
                    X_new = X_star - A * D
                else:
                    # Phase 2: search using a random whale (exploration)
                    rand_idx = rng.randint(pop_size)
                    X_rand = population[rand_idx]
                    D = np.abs(C * X_rand - X)
                    X_new = X_rand - A * D
            else:
                # Phase 3: spiral bubble-net attack
                l = rng.uniform(-1, 1, size=n_dim)
                D_prime = np.abs(X_star - X)
                X_new = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) + X_star

            new_population[i] = X_new

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
    run_woa.evaluation_history = evaluation_history
    return population[best_idx], fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, best_hist, mean_hist, pos_hist = run_woa(
        n_dim=10, pop_size=30, n_iterations=200, bounds=(-5.12, 5.12), seed=42,
    )

    print("\n=== Result (WOA on Sphere) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title("WOA Convergence on Sphere Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()