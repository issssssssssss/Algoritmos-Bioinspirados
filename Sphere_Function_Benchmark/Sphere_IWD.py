"""
Intelligent Water Drops (IWD) applied to the Sphere benchmark function
(Shah-Hosseini, 2007).

IMPORTANT ADAPTATION NOTE -- same situation as ACO -> ACOR: classic IWD
was designed for GRAPH-based combinatorial problems (like TSP), where
water drops flow along edges, eroding SOIL as they pass (fast-flowing,
high-velocity drops erode more), and future drops preferentially follow
low-soil (well-worn, previously good) edges. There's no natural graph
of "edges" in a continuous space like sphere's, so this file adapts the
core IWD MECHANISM -- velocity that builds up when moving through
"easy" terrain, and soil that biases future choices -- into a continuous
setting, the same way ACOR adapted ACO's pheromone trails into an
archive + Gaussian sampling. This is a non-standard adaptation (there
isn't one single agreed "continuous IWD" the way there is for ACOR);
treat the specific formulas as a reasonable reconstruction of IWD's
spirit rather than a universally cited equation set.

FITNESS: same direct sphere(x) evaluation of each drop's position.

PHASES (every iteration):

1. CANDIDATE GENERATION ("which edge to flow down")
   Each drop currently at position X generates `n_candidates` random
   nearby points within a shrinking search radius -- standing in for
   the graph nodes a real IWD could flow toward.

2. SOIL-BASED PROBABILISTIC SELECTION
   Each candidate is assigned a "soil" value based on ITS fitness
   (worse fitness = more soil = harder to flow through, exactly
   inverted from how pheromone works in ACO, since IWD drops prefer
   LOW soil). The drop picks its next position via roulette-wheel
   selection weighted by 1/soil -- low-soil (good) candidates are more
   likely to be chosen, but it's still probabilistic, not greedy.

3. VELOCITY UPDATE (canonical IWD formula)
   The drop's velocity updates using IWD's standard formula:
       v_new = v + av / (bv + cv * soil_chosen^2)
   Lower soil on the chosen path => bigger velocity gain. A
   fast-flowing drop represents one that has been moving through
   consistently good terrain.

4. GLOBAL SOIL-DRIVEN CONVERGENCE (reinforcement)
   After all drops move, the current best drop's position is treated
   as the "most eroded" (most-flowed-through) point. Every drop is then
   pulled slightly toward it, with pull strength proportional to the
   drop's OWN velocity (relative to the fastest drop this round) --
   fast, high-velocity drops (which have been finding good terrain)
   commit to the flow toward the best-known point more strongly than
   slow ones, mimicking how eroded, low-soil channels naturally
   concentrate future water flow.
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


def run_iwd(
    n_dim=10,
    pop_size=30,           # number of water drops
    n_iterations=200,
    bounds=(-5.12, 5.12),
    n_candidates=5,
    av=1.0, bv=0.01, cv=1.0,     # velocity update constants
    init_vel=1.0,
    radius_initial=1.5, radius_final=0.01,   # candidate search radius decay
    pull_strength=0.05,           # how strongly drops get pulled toward the current best each iteration
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

        # --- 1 & 2 & 3. Candidate generation, soil-based selection, velocity update ---
        for i in range(pop_size):
            candidates = positions[i] + radius * rng.uniform(-1, 1, size=(n_candidates, n_dim))
            candidates = np.clip(candidates, lo, hi)
            cand_fitness = evaluate_population(candidates)

            soil = cand_fitness - cand_fitness.min() + 1e-6  # lower fitness -> lower soil
            desirability = 1.0 / soil
            probs = desirability / desirability.sum()
            choice = rng.choice(n_candidates, p=probs)
            chosen_soil = soil[choice]

            velocities[i] = velocities[i] + av / (bv + cv * chosen_soil ** 2)
            positions[i] = candidates[choice]

        fitness = evaluate_population(positions)
        best_idx = np.argmin(fitness)
        X_best = positions[best_idx].copy()

        # --- 4. Global soil-driven convergence: pull drops toward the best ---
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

        if verbose and (it % 20 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best: {best_history[-1]:.8f} | Mean: {mean_history[-1]:.6f}")

    best_idx = np.argmin(fitness)
    run_iwd.evaluation_history = evaluation_history
    return positions[best_idx], fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, best_hist, mean_hist, pos_hist = run_iwd(
        n_dim=10, pop_size=30, n_iterations=2000, bounds=(-5.12, 5.12), seed=42,
    )

    print("\n=== Result (IWD on Sphere) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title("IWD Convergence on Sphere Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()