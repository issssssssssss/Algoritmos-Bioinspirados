"""
Firefly Algorithm (FA) applied to the Sphere benchmark function.

FA (Yang, 2008) mimics how fireflies are attracted to brighter fireflies,
with attraction weakening with distance (light absorbed by air).

FITNESS / BRIGHTNESS:
    Brightness is just the inverse relationship to fitness: for a
    minimization problem, a LOWER fitness value means a BRIGHTER firefly.
    We never actually transform fitness into a separate "brightness"
    number -- we just compare fitness values directly (lower = brighter).

PHASES (every iteration):

1. PAIRWISE COMPARISON
   For every pair of fireflies (i, j), check whether j is brighter
   (better fitness) than i.

2. ATTRACTION-BASED MOVEMENT
   If j is brighter, firefly i moves toward j:
       r_ij = ||X_i - X_j||                     (distance between them)
       beta = beta0 * exp(-gamma * r_ij^2)       (attractiveness, decays with distance)
       X_i_new = X_i + beta*(X_j - X_i) + alpha*epsilon
   where epsilon is a random vector (adds a random walk component so
   fireflies don't just collapse onto each other), alpha controls how
   much randomness, and gamma controls how fast attractiveness fades
   with distance (large gamma = very local attraction; small gamma =
   fireflies attract each other across the whole space).

3. RANDOMNESS DECAY
   alpha typically shrinks slightly each iteration (alpha *= 0.97 here),
   reducing the random-walk component over time -- early iterations
   explore more, later iterations refine more precisely around the
   brightest fireflies.

IMPORTANT PRACTICAL NOTE: gamma must be scaled to the search space size.
In a 10-dimensional space with range ~10 per dimension, distances
between fireflies are naturally large (r^2 can easily be in the
hundreds), so with gamma=1 the term exp(-gamma*r^2) underflows to
essentially zero -- every firefly looks "too far away" to attract any
other, and the algorithm barely moves. Using a small gamma (default
0.01 here) keeps attraction meaningful across the actual scale of the
search space. This is a very common practical pitfall when implementing
FA from the textbook formula without adjusting for problem
dimensionality/range.

Every firefly is compared against every OTHER firefly each iteration
(O(pop_size^2) comparisons), which is what makes FA naturally maintain
several attraction "clusters" rather than collapsing onto a single best
point the way PSO's single gbest pull does.

Formulas are the canonical ones from Yang's original FA formulation.
"""

import numpy as np
import matplotlib.pyplot as plt

evaluation_count = 0


def evaluate(x):
    global evaluation_count
    evaluation_count += 1
    return sphere(x)

def sphere(x):
    return np.sum(x ** 2)


def run_fa(
    n_dim=10,
    pop_size=30,
    n_iterations=200,
    bounds=(-5.12, 5.12),
    alpha=0.5,       # randomness strength
    beta0=1.0,       # attractiveness at distance 0
    gamma=0.01,      # light absorption coefficient (scaled for a ~10-unit range search space)
    alpha_decay=0.97,
    seed=None,
    verbose=True,
):
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)
    lo, hi = bounds

    population = rng.uniform(lo, hi, size=(pop_size, n_dim))
    fitness = np.array([evaluate(ind) for ind in population])

    best_history = []
    mean_history = []
    best_position_history = []

    alpha_t = alpha
    for it in range(n_iterations):
        alpha_t *= alpha_decay

        for i in range(pop_size):
            for j in range(pop_size):
                if fitness[j] < fitness[i]:  # j is brighter than i
                    r = np.linalg.norm(population[i] - population[j])
                    beta = beta0 * np.exp(-gamma * r ** 2)
                    eps = alpha_t * (rng.rand(n_dim) - 0.5)
                    population[i] = population[i] + beta * (population[j] - population[i]) + eps
                    population[i] = np.clip(population[i], lo, hi)
                    fitness[i] = evaluate(population[i])

        best_idx = np.argmin(fitness)
        best_history.append(fitness[best_idx])
        mean_history.append(np.mean(fitness))
        best_position_history.append(population[best_idx].copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 20 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best: {best_history[-1]:.8f} | Mean: {mean_history[-1]:.6f}")

    best_idx = np.argmin(fitness)

    run_fa.evaluation_history = evaluation_history
    return population[best_idx], fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, best_hist, mean_hist, pos_hist = run_fa(
        n_dim=10, pop_size=30, n_iterations=200, bounds=(-5.12, 5.12), seed=42,
    )

    print("\n=== Result (FA on Sphere) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title("FA Convergence on Sphere Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()