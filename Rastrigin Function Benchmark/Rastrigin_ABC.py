"""
Artificial Bee Colony (ABC) applied to the Rastrigin benchmark function.

Same algorithm as abc_sphere.py -- see that file for the full phase-by-
phase explanation (employed bees, onlooker bees, scout bees). Only the
objective function changes here.

Worth watching for: the SCOUT BEE phase (abandoning any food source that
hasn't improved in `limit` trials, replacing it with a fresh random
point) is exactly the kind of forced-restart mechanism that helps on
multimodal landscapes -- it's ABC's explicit answer to "how do we escape
a local optimum" that GWO doesn't really have.
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


def run_abc(
    n_dim=10,
    pop_size=30,
    n_iterations=300,
    bounds=(-5.12, 5.12),
    limit=50,
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
    trial = np.zeros(pop_size, dtype=int)

    def fitness_transform(f):
        return np.where(f >= 0, 1.0 / (1.0 + f), 1.0 + np.abs(f))

    def neighbor_search(i):
        j = rng.randint(n_dim)
        k = rng.choice([idx for idx in range(pop_size) if idx != i])
        phi = rng.uniform(-1, 1)
        v = population[i].copy()
        v[j] = population[i][j] + phi * (population[i][j] - population[k][j])
        v = np.clip(v, lo, hi)
        return v

    best_history = []
    mean_history = []
    best_position_history = []
    best_ever_x = population[np.argmin(fitness)].copy()
    best_ever_f = fitness.min()

    for it in range(n_iterations):
        for i in range(pop_size):
            v = neighbor_search(i)
            fv = evaluate(v)
            if fv < fitness[i]:
                population[i], fitness[i] = v, fv
                trial[i] = 0
            else:
                trial[i] += 1

        trans_fit = fitness_transform(fitness)
        probs = trans_fit / trans_fit.sum()
        for _ in range(pop_size):
            i = rng.choice(pop_size, p=probs)
            v = neighbor_search(i)
            fv = evaluate(v)
            if fv < fitness[i]:
                population[i], fitness[i] = v, fv
                trial[i] = 0
            else:
                trial[i] += 1

        for i in range(pop_size):
            if trial[i] > limit:
                population[i] = rng.uniform(lo, hi, size=n_dim)
                fitness[i] = evaluate(population[i])
                trial[i] = 0

        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_ever_f:
            best_ever_f = fitness[gen_best_idx]
            best_ever_x = population[gen_best_idx].copy()

        best_history.append(best_ever_f)
        mean_history.append(np.mean(fitness))
        best_position_history.append(best_ever_x.copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 30 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best-ever: {best_ever_f:.6f} | Mean: {mean_history[-1]:.6f}")

    run_abc.evaluation_history = evaluation_history
    return best_ever_x, best_ever_f, best_history, mean_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, best_hist, mean_hist, pos_hist = run_abc(
        n_dim=10, pop_size=30, n_iterations=300, bounds=(-5.12, 5.12), seed=42,
    )

    print("\n=== Result (ABC on Rastrigin) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best-ever fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title("ABC Convergence on Rastrigin Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()